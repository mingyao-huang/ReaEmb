import os
import json
import copy
from tqdm import tqdm


def save_data(data_path, data):
    os.makedirs("../handled", exist_ok=True)
    with open(os.path.join("../handled", data_path), "w", encoding="utf-8") as writer:
        for meta_data in data:
            writer.write(json.dumps(meta_data, ensure_ascii=False) + "\n")


def build_item_prompts():
	data = json.load(open("./item_info.json", "r"))

	example_dict = {}
	for item_dict in tqdm(data):
		example_dict.update(item_dict)

	instruction = "The CD item has the following attributes: \n "

	item_data = {}
	for item_dict in tqdm(data):
		item_prompt = copy.deepcopy(instruction)
		item_id = None
		for key, value in item_dict.items():
			if key in ["fit", "also_buy", "also_view", "similar_item", "imageURL", "imageURLHighRes"]:
				continue
			elif key in ["parent_asin"]:
				item_id = value
			elif key in ["description", "features"]:
				attri_str = value
				if attri_str == "":
					attri_str = "none, "
				attri_str = attri_str.replace("\n", " ").replace(";", ".")
				if len(attri_str) > 100:
					attri_str = attri_str[:100]
				while attri_str and attri_str[-1] == ";":
					attri_str = attri_str[:-1]
				attri_prompt = key + " is " + attri_str + "; "
				item_prompt += attri_prompt
			else:
				if isinstance(value, str) and len(value) > 100:
					value = value[:100]
				attri_prompt = key + " is " + str(value).replace("\n", " ").replace(";", ".") + "; "
				item_prompt += attri_prompt
		if item_id:
			item_data[item_id] = item_prompt[:-2]
		else:
			raise ValueError("No item id")

	json.dump(item_data, open("../handled/item_info.json", "w", encoding="utf-8"), ensure_ascii=False)

	with open("./map.txt", "r", encoding="utf-8") as file_obj:
		lines = file_obj.readlines()
	id_map = json.loads(lines[1])

	json_data = []
	for key, value in item_data.items():
		json_data.append({"input": value, "target": "", "item": key, "item_id": id_map[key]})

	save_data("item_info.jsonline", json_data)


if __name__ == "__main__":
	build_item_prompts()