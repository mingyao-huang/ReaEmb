import os
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def Yelp(date_min, date_max, rating_score): # take out inters in [date_min, date_max] and the score < rating_score
    datas = []
    data_flie = '../../yelp/dataset/yelp_academic_dataset_review.json'
    lines = open(data_flie, encoding="utf-8").readlines()
    for line in tqdm(lines):
        review = json.loads(line.strip())
        user = review['user_id']
        item = review['business_id']
        rating = review['stars']
        date = review['date']
        # filter out some examples
        if date < date_min or date > date_max or float(rating) <= rating_score:
            continue
        time = date.replace('-','').replace(':','').replace(' ','') 
        datas.append((user, item, int(time)))
    return datas

def filter_duplicate(datas):
    seen = set()
    filtered = []
    for entry in datas:
        if entry not in seen:
            seen.add(entry)
            filtered.append(entry)
    return filtered


def get_interaction(datas): # sort the interactions based on timestamp
    user_seq = {}
    for data in datas:
        user, item, time = data
        if user in user_seq:
            user_seq[user].append((item, time))
        else:
            user_seq[user] = []
            user_seq[user].append((item, time))

    for user, item_time in user_seq.items():
        item_time.sort(key=lambda x: x[1])  # sort interactions for each user by timestamp
        items = []
        for t in item_time:
            items.append(t[0])
        user_seq[user] = items
    return user_seq

def filter_by_k_core(user_items, min_u_num=6, min_i_num=3, user_ts=10, item_ts=20):

    print(f"Calculating k-core (user>={min_u_num}, item>={min_i_num}) ...")
    user_items = {u: list(items) for u, items in user_items.items()}  # deep copy to avoid modifying input

    iteration = 0
    while True:
        # Count interactions per user and per item
        user_count = {u: len(items) for u, items in user_items.items()}
        item_count = defaultdict(int)
        for items in user_items.values():
            for i in items:
                item_count[i] += 1
        # Find users and items that do not meet the thresholds
        ban_users = set(u for u, c in user_count.items() if c < min_u_num)
        ban_items = set(i for i, c in item_count.items() if c < min_i_num)

        if not ban_users and not ban_items:
            print(f"{sum(len(items) for items in user_items.values())} rows left in (u={min_u_num},i={min_i_num})-core")
            break

        # Filter out users and items
        before = sum(len(items) for items in user_items.values())
        # First remove users below threshold
        user_items = {u: items for u, items in user_items.items() if u not in ban_users}
        # Then remove users that have any banned items in their history
        for u in list(user_items.keys()):
            if any(i in ban_items for i in user_items[u]):
                del user_items[u]
        dropped = before - sum(len(items) for items in user_items.values())
        print(f"\titeration {iteration}: {dropped} dropped interactions, "
              f"with {len(ban_users)} users banned and {len(ban_items)} items banned")
        iteration += 1

    # Build the item_set
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

    # Tail user ratio
    tail_users = sum(1 for items in user_items.values() if len(items) < user_ts)
    tail_user_ratio = tail_users / total_users if total_users > 0 else 0

    # Tail item ratio
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

def filter_like_LLMEmb(user_items, min_u_num=6, min_i_num=3, user_ts=10, item_ts=20):
    print(f"Calculating LLMEmb-like filter (user>={min_u_num}, item>={min_i_num}) ...")
    user_count = {u: len(items) for u, items in user_items.items()}
    item_count = defaultdict(int)
    for items in user_items.values():
        for i in items:
            item_count[i] += 1
    new_user_items = {}
    for u, items in user_items.items():
        if user_count[u] < min_u_num:
            continue
        new_items = [i for i in items if item_count[i] >= min_i_num]
        new_user_items[u] = new_items
    user_items = new_user_items
    new_user_items = {}
    for u, items in user_items.items():
        if len(items) >= min_u_num:
            new_user_items[u] = items
    user_items = new_user_items
    
    # build item_set
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

    # tail user ratio
    tail_users = sum(1 for items in user_items.values() if len(items) < user_ts)
    tail_user_ratio = tail_users / total_users if total_users > 0 else 0

    # tail item ratio
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

def Yelp_meta(file_path, item_set):
    """
    Extract metadata for business IDs in item_set: name, categories, city, stars, review_count, is_open
    droped: address, state, postal_code, latitude, longitude, attributes, hours
    """
    meta_data = {}
    with open(file_path, 'r', encoding='utf-8') as fp:
        for line in fp:
            data = json.loads(line.strip())
            if data['business_id'] in item_set:
                meta_data[data['business_id']] = {
                    'business_id': data.get('business_id', ''),
                    'name': data.get('name', ''),
                    'categories': data.get('categories', ''),
                    'city': data.get('city', ''),
                    'stars': data.get('stars', ''),
                    'review_count': data.get('review_count', ''),
                    'is_open': data.get('is_open', '')
                }
    # Check for items missing metadata
    missing_items = item_set - set(meta_data.keys())
    if missing_items:
        print(f"The following items are missing metadata: {missing_items}")
    else:
        print("All items have metadata")
    return meta_data

def map_ids(subset, meta_data): # Map original user and item IDs to consecutive positive integers and remap meta_data accordingly
    user_map = {user: idx for idx, user in enumerate(subset.keys(), start=1)}
    item_map = {item: idx for idx, item in enumerate(meta_data.keys(), start=1)}
    new_subset = {user_map[user]: [item_map[item] for item in items] for user, items in subset.items()}
    new_meta_data = {item_map[item]: info for item, info in meta_data.items()}
    return new_subset, new_meta_data, user_map, item_map

def save_infos(new_subset, new_meta_data, user_map, item_map):
    with open('../handled/inter.txt', 'w', encoding='utf-8') as f:
        for user, items in new_subset.items():
            for item in items:
                f.write(f"{user}"+' '+f"{item}\n")
    processed_items = []
    for _, info in new_meta_data.items():
        processed_items.append(info)
    with open('../dataset/item_info.json', 'w') as f:
        json.dump(processed_items, f, ensure_ascii=False, indent=2)
    with open('../dataset/map.txt', 'w', encoding='utf-8') as f:
        # Write mapping tables as JSON, one mapping per line
        f.write(json.dumps(user_map, ensure_ascii=False) + '\n')
        f.write(json.dumps(item_map, ensure_ascii=False) + '\n')

def main():
    rating_score = 3.0
    user_core=5
    item_core=3
    user_ts = 10
    item_ts = 18
    file_path = "../../yelp/dataset/yelp_academic_dataset_business.json"
    # 1. Read Yelp data
    date_max = '2024-12-31 00:00:00'
    date_min = '2000-01-01 00:00:00'
    datas = Yelp(date_min, date_max, rating_score)

    # 2. Build user-item interaction sequences
    datas = filter_duplicate(datas)  # remove duplicate interactions
    user_items = get_interaction(datas)
    print(f'Raw data has been processed! Lower than {rating_score} are deleted!')

    # 3. k-core filtering
    user_items, item_set = filter_by_k_core(user_items, min_u_num=user_core, min_i_num=item_core, user_ts=user_ts, item_ts=item_ts)
    print(f'User {user_core}-core complete! Item {item_core}-core complete!')

    # 4. Extract metadata and attributes
    print('Begin extracting meta infos...')
    meta_infos = Yelp_meta(file_path, item_set)

    # 5. ID mapping
    user_items, meta_infos, user_map, item_map = map_ids(user_items, meta_infos)

    # 6. Save processed results
    save_infos(user_items, meta_infos, user_map, item_map)
    print('All data has been processed and saved!')

if __name__ == "__main__":
    main()