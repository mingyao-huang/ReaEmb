import os
import pickle
import jsonlines
import pandas as pd
import numpy as np
import json
import copy
from tqdm import tqdm


def save_data(data_path, data):
	"""Write list of dicts as jsonl under ./handled."""
	with jsonlines.open(os.path.join("../handled", data_path), "w") as writer:
		for meta_data in data:
			writer.write(meta_data)


def build_item_prompts():
	data = json.load(open("./item_info.json", "r"))

	example_dict = {}
	for item_dict in tqdm(data):
		example_dict.update(item_dict)

	instruction = "The item has the following attributes: \n "

	item_data = {}
	for item_dict in tqdm(data):
		item_prompt = copy.deepcopy(instruction)
		item_id = None
		for key, value in item_dict.items():
			if key in ["fit", "also_buy", "also_view", "similar_item", "imageURL", "imageURLHighRes"]:
				continue
			elif key in ["business_id"]:
				item_id = value
			else:
				if isinstance(value, str) and len(value) > 100:
					value = value[:100]
				attri_prompt = key + " is " + str(value).replace("\n", " ").replace(";", ".") + "; "
				item_prompt += attri_prompt
		if item_id:
			item_data[item_id] = item_prompt[:-2]
		else:
			raise ValueError("No item id")

	json.dump(item_data, open("../handled/item_str_0722.json", "w"))

	with open("./map.txt", "r", encoding="utf-8") as file_obj:
		lines = file_obj.readlines()
	id_map = json.loads(lines[1])

	json_data = []
	for key, value in item_data.items():
		json_data.append({"input": value, "target": "", "item": key, "item_id": id_map[key]})

	save_data("item_str_0722.jsonline", json_data)


if __name__ == "__main__":
	build_item_prompts()