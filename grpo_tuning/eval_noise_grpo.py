import os
import sys
import json
from typing import List

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))

import fire
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from model import QwenRSEmbForGRPO


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _encode_with_flags(tokenizer, text: str, bos: bool, eos: bool) -> List[int]:
	tokens = tokenizer.encode(text)
	bos_id = tokenizer.bos_token_id
	eos_id = tokenizer.eos_token_id
	while tokens and bos_id is not None and tokens[0] == bos_id:
		tokens = tokens[1:]
	while tokens and eos_id is not None and tokens[-1] == eos_id:
		tokens = tokens[:-1]
	if bos and bos_id is not None:
		tokens = [bos_id] + tokens
	if eos and eos_id is not None:
		tokens = tokens + [eos_id]
	return tokens


def _load_item_prompts(dataset: str) -> List[str]:
	item_file = os.path.join(_PROJECT_ROOT, "data", dataset, "handled", "item_info.jsonline")
	prompts: List[str] = []
	with open(item_file, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			obj = json.loads(line)
			raw = obj.get("input", "")
			parts = raw.split("\n", 1)
			prompt = parts[1] if len(parts) > 1 else raw
			prompts.append(prompt)
	return prompts


def _mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
	mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
	summed = (hidden_states * mask).sum(dim=1)
	denom = mask.sum(dim=1).clamp_min(1.0)
	return summed / denom


def _load_state_dict(model_dir: str) -> dict:
	safetensors_path = os.path.join(model_dir, "model.safetensors")
	bin_path = os.path.join(model_dir, "pytorch_model.bin")
	if os.path.exists(safetensors_path):
		try:
			from safetensors.torch import load_file
			return load_file(safetensors_path)
		except Exception as exc:
			raise RuntimeError(f"Failed to load {safetensors_path}") from exc
	if os.path.exists(bin_path):
		return torch.load(bin_path, map_location="cpu")
	raise FileNotFoundError("No model.safetensors or pytorch_model.bin found in model_path.")


def _infer_base_model_from_config(model_dir: str) -> str:
	config_path = os.path.join(model_dir, "config.json")
	if not os.path.exists(config_path):
		return ""
	with open(config_path, "r", encoding="utf-8") as f:
		cfg = json.load(f)
	name_or_path = cfg.get("_name_or_path", "")
	return name_or_path


def _is_wrapper_state_dict(state_dict: dict) -> bool:
	for k in state_dict.keys():
		if k.startswith("model.") or k.startswith("r3_attn") or k.startswith("cls_head"):
			return True
	return False


def _filter_mismatched_state_dict(model: torch.nn.Module, state_dict: dict) -> dict:
	"""Drop state-dict entries whose shapes do not match the current model."""
	model_state = model.state_dict()
	filtered = {}
	for k, v in state_dict.items():
		if k in model_state and model_state[k].shape == v.shape:
			filtered[k] = v
	return filtered


def _pool_item_embeds(model, outputs, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
	last_hidden = outputs.hidden_states[-1]
	pool_type = getattr(model, "pool_type", "last")
	if pool_type == "avg":
		return _mean_pool(last_hidden, attention_mask)
	if pool_type == "last":
		seq_len = attention_mask.size(1)
		mask = attention_mask.bool()
		rev = torch.flip(mask, dims=[1])
		rev_idx = torch.argmax(rev.int(), dim=1)
		last_idx = (seq_len - 1 - rev_idx).to(attention_mask.device)
		batch_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
		if hasattr(model, "thought_token_id"):
			thought_id = int(model.thought_token_id)
		else:
			thought_id = int(getattr(model.config, "thought_token_id", -1))
		if thought_id < 0:
			raise ValueError("thought_token_id not found; cannot validate pool_type='last'.")
		last_token_ids = input_ids[batch_idx, last_idx]
		if not torch.all(last_token_ids.eq(thought_id)):
			raise ValueError("pool_type='last' expects last valid token to be thought_id.")
		return last_hidden[batch_idx, last_idx]
	return _mean_pool(last_hidden, attention_mask)


def extract_item_embs(
	dataset: str = "games",
	model_path: str = None,
	base_model: str = "../qwen2.5_0.5b",
	user_prefix: str = "### Instruction:\n Please provide the item description. The item information is: ",
	cutoff_len: int = 256,
	batch_size: int = 4,
	r3_think: bool = True,
	pool_type: str = "avg",
	end_k: int = -1,
	output_json: str = None,
):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if model_path is None:
		model_path = f"./output_grpo/{dataset}/"
	if output_json is None:
		output_json = f"./grpo_emb/{dataset}/json/item_embs_grpo.jsonl"

	tokenizer = AutoTokenizer.from_pretrained(
		model_path,
		trust_remote_code=True,
		fix_mistral_regex=True,
	)
	if "<|Thought|>" not in tokenizer.additional_special_tokens:
		additional_special_tokens = tokenizer.additional_special_tokens
		additional_special_tokens.append("<|Thought|>")
		tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})

	thought_token_id = tokenizer.convert_tokens_to_ids("<|Thought|>")
	if thought_token_id is None or thought_token_id == tokenizer.unk_token_id:
		raise ValueError("Failed to resolve <|Thought|> token id.")

	state_dict = _load_state_dict(model_path)
	if base_model is None and _is_wrapper_state_dict(state_dict):
		inferred = _infer_base_model_from_config(model_path)
		if inferred:
			base_model = inferred
			print(f"Inferred base_model from config: {base_model}")
		else:
			raise ValueError("base_model is required because model_path contains wrapper-only weights.")

	init_path = base_model if base_model else model_path
	model = QwenRSEmbForGRPO.from_pretrained(
		init_path,
		R3_think=r3_think,
		pool_type=pool_type,
		thought_token_id=int(thought_token_id),
		thought_end_k=end_k,
		torch_dtype=torch.bfloat16,
	)
	if len(tokenizer) != model.get_input_embeddings().num_embeddings:
		model.resize_token_embeddings(len(tokenizer))
	if base_model or _is_wrapper_state_dict(state_dict):
		state_dict = _filter_mismatched_state_dict(model, state_dict)
		missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
		if missing_keys:
			print(f"Missing keys: {missing_keys[:5]}...")
		if unexpected_keys:
			print(f"Unexpected keys: {unexpected_keys[:5]}...")

		for name, param in model.named_parameters():
			if torch.isnan(param).any():
				print(f"Warning: NaN in parameter {name}")
				break
	model.to(device)
	model.eval()

	torch.set_grad_enabled(False)

	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.pad_token_id = tokenizer.eos_token_id
	tokenizer.padding_side = "left"

	prompts = _load_item_prompts(dataset)
	os.makedirs(os.path.dirname(output_json), exist_ok=True)

	with open(output_json, "w", encoding="utf-8") as fout:
		with torch.no_grad():
			for start in tqdm(range(0, len(prompts), batch_size), desc="Extracting item embeddings"):
				batch_prompts = prompts[start : start + batch_size]
				input_ids_list = []
				valid_indices = []
				for i, prompt in enumerate(batch_prompts):
					tokens = _encode_with_flags(tokenizer, user_prefix, bos=True, eos=False)
					tokens += _encode_with_flags(tokenizer, prompt, bos=False, eos=False)
					tokens += _encode_with_flags(tokenizer, "<|Thought|>", bos=False, eos=False)
					if len(tokens) == 0 or tokens[-1] != thought_token_id:
						print(f"Warning: Invalid tokens for prompt {start + i}: no <|Thought|> at end")
						continue
					input_ids_list.append(tokens[-cutoff_len:])
					valid_indices.append(i)

				padded = tokenizer.pad(
					{"input_ids": input_ids_list},
					padding=True,
					return_tensors="pt",
				)
				input_ids = padded["input_ids"].to(device)
				attention_mask = padded["attention_mask"].to(device)

				outputs = model(
					input_ids=input_ids,
					attention_mask=attention_mask,
					output_hidden_states=True,
					return_dict=True,
				)

				if torch.isnan(outputs.hidden_states[-1]).any():
					print(f"Warning: NaN detected in hidden_states at batch {start}")
					print(f"Input IDs shape: {input_ids.shape}")
					print(f"Sample input IDs: {input_ids[0][:10]}")
					continue
				embs = _pool_item_embeds(model, outputs, input_ids, attention_mask)
				embs = embs.float().detach().cpu().numpy()

				for i, emb in enumerate(embs):
					original_i = valid_indices[i]
					item_id = start + original_i + 1
					record = {"item_id": item_id, "hidden_state": emb.tolist()}
					fout.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
	fire.Fire(extract_item_embs)
