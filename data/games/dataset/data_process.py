import json
import os
import requests
import copy 
import random
from tqdm import tqdm
from collections import defaultdict

"""Preprocess Amazon review data and metadata for ReaEmb."""

def filter_low_rating(file_path, rating_scores):
    datas = []
    with open(file_path, 'r') as fp:
        for line in fp:
            data = json.loads(line.strip())
            if data['rating'] < rating_scores:
                continue
            else:
                datas.append((data['user_id'], data['parent_asin'], data['timestamp']))
    return datas

def clean_datas(datas):
    cleaned = []
    seen = set()
    for row in datas:
        if any(x is None or x == '' for x in row):
            continue
        if row in seen:
            continue
        seen.add(row)
        cleaned.append(row)
    return cleaned

def convert_to_sequence(datas):
    user_dict = defaultdict(list)
    for user_id, item_id, timestamp in datas:
        user_dict[user_id].append((item_id, timestamp))
    
    user_items = {}
    for user_id, interactions in user_dict.items():
        sorted_interactions = sorted(interactions, key=lambda x: x[1])
        item_sequence = [item_id for item_id, _ in sorted_interactions]
        user_items[user_id] = item_sequence

    return user_items

def filter_by_k_core(user_items, min_u_num=6, min_i_num=3, user_ts=10, item_ts=20):

    print(f"Calculating k-core (user>={min_u_num}, item>={min_i_num}) ...")
    user_items = {u: list(items) for u, items in user_items.items()}

    iteration = 0
    while True:
        user_count = {u: len(items) for u, items in user_items.items()}
        item_count = defaultdict(int)
        for items in user_items.values():
            for i in items:
                item_count[i] += 1

        ban_users = set(u for u, c in user_count.items() if c < min_u_num)
        ban_items = set(i for i, c in item_count.items() if c < min_i_num)

        if not ban_users and not ban_items:
            print(f"{sum(len(items) for items in user_items.values())} rows left in (u={min_u_num},i={min_i_num})-core")
            break

        before = sum(len(items) for items in user_items.values())
        # Remove low-frequency users and items.
        user_items = {u: items for u, items in user_items.items() if u not in ban_users}
        for u in list(user_items.keys()):
            # if user has banned items, remove the user
            if any(i in ban_items for i in user_items[u]):
                del user_items[u]
        dropped = before - sum(len(items) for items in user_items.values())
        print(f"\titeration {iteration}: {dropped} dropped interactions, "
              f"with {len(ban_users)} users banned and {len(ban_items)} items banned")
        iteration += 1

    # Build item statistics.
    item_set = set()
    item_count = defaultdict(int)
    total_interactions = 0
    for items in user_items.values():
        item_set.update(items)
        total_interactions += len(items)
        for i in items:
            item_count[i] += 1

    total_users = len(user_items)
    total_items = len(item_set)
    sparsity = 1 - total_interactions / (total_users * total_items) if total_users and total_items else 0

    # Tail-user statistics.
    tail_users = sum(1 for items in user_items.values() if len(items) < user_ts)
    tail_user_ratio = tail_users / total_users if total_users > 0 else 0

    # Tail-item statistics.
    tail_items = sum(1 for i in item_set if item_count[i] < item_ts)
    tail_item_ratio = tail_items / total_items if total_items > 0 else 0

    print(f"Total users: {total_users}")
    print(f"Total items: {total_items}")
    print(f"Total interactions: {total_interactions}")
    print(f"Sparsity: {sparsity*100:.2f}%")
    print(f"Tail user ratio: {tail_user_ratio*100:.2f}% ({tail_users}/{total_users})")
    print(f"Tail item ratio: {tail_item_ratio*100:.2f}% ({tail_items}/{total_items})")
    print(f"Average user interaction length: {total_interactions / total_users:.2f}" if total_users > 0 else "Average user interaction length: 0")
    print(f"Average item interaction count: {total_interactions / total_items:.2f}" if total_items > 0 else "Average item interaction count: 0")

    return user_items, item_set

def games_meta_data(file_path, item_set): # Extract metadata fields for items in item_set: title, features, description, average_rating, price, images, store, categories, details, parent_asin
    meta_data = {}
    with open(file_path, 'r') as fp:
        for line in fp:
            data = json.loads(line.strip())
            if data['parent_asin'] in item_set:
                meta_data[data['parent_asin']] = {
                    'main_category': data.get('main_category', ''),
                    'title': data.get('title', ''),
                    'features': data.get('features', []),
                    'description': data.get('description', []),
                    'price': data.get('price', ''),
                    'store': data.get('store', ''),
                    'categories': data.get('categories', []),
                    #'images': data.get('images', []),
                    'parent_asin': data.get('parent_asin', '')
                }
    # Check for items missing metadata
    missing_items = item_set - set(meta_data.keys())
    if missing_items:
        print(f"The following items are missing metadata: {missing_items}")
    else:
        print("All items have metadata")
    return meta_data

def map_ids(subset, meta_data): # Map the keys in subset, i.e., user IDs, to positive integers, and also map item IDs to positive integers. Record the user mapping and item mapping tables, and finally map the item IDs in meta_data to the new IDs
    user_map = {user: idx for idx, user in enumerate(subset.keys(), start=1)}
    item_map = {item: idx for idx, item in enumerate(meta_data.keys(), start=1)}
    new_subset = {user_map[user]: [item_map[item] for item in items] for user, items in subset.items()}
    new_meta_data = {item_map[item]: info for item, info in meta_data.items()}
    return new_subset, new_meta_data, user_map, item_map

def get_item_image(meta_data):
    error_image_id=[]
    save_dir = '../dataset/image/'
    os.makedirs(save_dir, exist_ok=True)
    for item, info in tqdm(meta_data.items(), desc="Downloading images progress"):
        # Check if image already exists
        img_path = os.path.join(save_dir, info.get('parent_asin', '') + '.jpg')
        if os.path.exists(img_path):
            continue
        # If image does not exist, download it
        images = info.get('images', [])
        if images and isinstance(images[0], dict):
            large_url = images[0].get('large', '')
            if large_url:
                try:
                    response = requests.get(large_url, timeout=10)
                    if response.status_code == 200:
                        with open(img_path, 'wb') as img_file:
                            img_file.write(response.content)
                    else:
                        error_image_id.append(info.get('parent_asin', ''))
                except Exception as e:
                    print(f"Failed to download image: {large_url}, error: {e}")
                    error_image_id.append(info.get('parent_asin', ''))
            else:
                error_image_id.append(info.get('parent_asin', ''))
        else:
            error_image_id.append(info.get('parent_asin', ''))
    if not error_image_id:
        print("All images downloaded successfully")
    else:
        print(f"Number of images with errors: {len(error_image_id)}")
        print(f"Image IDs with errors: {error_image_id}")
    return error_image_id


def _clean_text(value: str) -> str:
    return str(value).strip() if value is not None else ""

def process_item(item, max_features=4):
    title = item.get("title", "").strip()
    main_cat = (item.get("main_category") or "").strip()
    categories = ", ".join(item.get("categories", []))
    price = item.get("price", "")
    store = _clean_text(item.get("store", ""))
    parent_asin = item.get("parent_asin", "").strip()
    
    # Simplify features: keep up to max_features and deduplicate
    features = item.get("features", [])
    seen = set()
    filtered_features = []
    for f in features:
        short_f = f.strip().split(" - ")[0]  # Trim long explanatory parts
        if short_f not in seen:
            filtered_features.append(short_f)
            seen.add(short_f)
        if len(filtered_features) >= max_features:
            break
    
    features_str = " ".join(filtered_features)
    
    # Use first sentence of description as a summary
    description_list = item.get("description", [])
    description = description_list[0].split(".")[0] if description_list else ""
    if description:
        description += "."
    if features:
        features_str += "."
    
    # Combine main_category with categories, but skip duplication of main_category
    categories_list = [c.strip() for c in categories.split(",") if c.strip()]
    if main_cat:
        categories_list = [c for c in categories_list if c.lower() != main_cat.lower()]
        categories_list.insert(0, main_cat)
    category_str = ", ".join(categories_list)

    processed_text = {
        "title": title,
        "category": category_str,
        "price": price,
        "store": store,
        "features": features_str,
        "description": description,
        "parent_asin": parent_asin
    }
    return processed_text

def store_info(new_subset, new_meta_data, user_map, item_map):
    os.makedirs('../handled', exist_ok=True)
    os.makedirs('../dataset', exist_ok=True)
    with open('../handled/inter.txt', 'w') as f:
        for user, items in new_subset.items():
            for item in items:
                f.write(f"{user}"+' '+f"{item}\n")
    processed_items = []
    for _, info in new_meta_data.items():
        info_copy = copy.deepcopy(info)
        info_copy = process_item(info_copy, max_features=4)
        processed_items.append(info_copy)
    with open('../dataset/item_info.json', 'w') as f:
        json.dump(processed_items, f, ensure_ascii=False, indent=2)
    with open('../dataset/map.txt', 'w') as f:
        f.write(json.dumps(user_map, ensure_ascii=False) + '\n')
        f.write(json.dumps(item_map, ensure_ascii=False) + '\n')

def main():
    file_path = "./Video_Games.jsonl"
    rating_scores = 0.0
    datas = filter_low_rating(file_path, rating_scores)
    datas = clean_datas(datas)
    user_items = convert_to_sequence(datas)
    user_items, item_set = filter_by_k_core(user_items, min_u_num=5, min_i_num=3, user_ts=10, item_ts=20)
    file_path = "./meta_Video_Games.jsonl"
    meta_data = games_meta_data(file_path, item_set)
    subset, new_meta_data, user_map, item_map = map_ids(user_items, meta_data)
    # error_image_id = get_item_image(new_meta_data)
    store_info(subset, new_meta_data, user_map, item_map)

if __name__ == "__main__":
    main()

