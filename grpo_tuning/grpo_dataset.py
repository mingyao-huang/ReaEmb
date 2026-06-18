import json
import os
import random
from collections import defaultdict
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def apply_chat_template(tokenizer, text: str, system_prompt: str, user_prefix: str, add_generation_prompt: bool = True) -> str:
    prompt_text = f"{user_prefix}\n{text}" if user_prefix else text
    if not getattr(tokenizer, "chat_template", None):
        return f"System:\n{system_prompt}\n\nUser:\n{prompt_text}" if system_prompt else f"User:\n{prompt_text}"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt_text})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )

def tokenize_chat_text_generation(
    tokenizer,
    text: str,
    system_prompt: str,
    user_prefix: str,
    max_source_length: int,
    add_generation_prompt: bool = True,
):
    """Lightweight prompt tokenization for GRPO dataset/eval.

    Keeps only logic needed by GRPO data construction:
    - build chat text with system/user messages
    - tokenize and truncate to max length
    - append required <|Thought|> reasoning token
    """
    base_templated = apply_chat_template(
        tokenizer,
        text,
        system_prompt,
        user_prefix,
        add_generation_prompt=add_generation_prompt,
    )

    encoded = tokenizer(base_templated, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Reserve 1 slot: required <|Thought|> token
    # Old behavior (with eos):
    max_source_len = max(max_source_length - 2, 1)
    # max_source_len = max(max_source_length - 1, 1)
    if len(input_ids) > max_source_len:
        input_ids = input_ids[:max_source_len]
        attention_mask = attention_mask[:max_source_len]

    thought_token_id = tokenizer.convert_tokens_to_ids("<|Thought|>")
    if thought_token_id is None or thought_token_id == tokenizer.unk_token_id:
        raise ValueError("Failed to resolve <|Thought|> token id for GRPO dataset tokenization.")

    input_ids = input_ids + [int(thought_token_id)]
    attention_mask = attention_mask + [1]
    # Old behavior (with eos):
    input_ids = input_ids + [tokenizer.eos_token_id]
    attention_mask = attention_mask + [1]
    return input_ids, [int(x) for x in attention_mask]


class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos_id: int = self.tokenizer.bos_token_id
        self.eos_id: int = self.tokenizer.eos_token_id

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.tokenizer.encode(s)
        while t[0] == self.bos_id:
            t = t[1:]
        while t[-1] == self.eos_id:
            t = t[:-1]

        if bos and self.bos_id is not None:
            t = [self.bos_id] + t
        if eos and self.eos_id is not None:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.tokenizer.decode(t)


def get_item_inter_frequency(
    inter_file: str,
    num_items: int,
    candidate_num: int,
    pos_num: int,
    seed: int = 0,
) -> Dict[int, List[int]]:
    """Build candidate item list for each item.

    Returns a dict: {item_id: [pos_1, ..., pos_pos_num, neg_1, ..., neg_k]}.
    """
    rng = random.Random(seed)
    inter_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    user_items: Dict[int, List[int]] = defaultdict(list)

    with open(inter_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            user_id, item_id = int(parts[0]), int(parts[1])
            if 1 <= item_id <= num_items:
                user_items[user_id].append(item_id)
            else:
                print(f"Warning: item id {item_id} out of range [1, {num_items}], skipping")

    for items in user_items.values():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                item_i = items[i]
                item_j = items[j]
                if item_i != item_j:
                    inter_counts[item_i][item_j] += 1
                    inter_counts[item_j][item_i] += 1

    all_items = list(range(1, num_items+1))
    candidates: Dict[int, List[int]] = {}

    for item_id in tqdm(range(num_items), desc="Building candidates"):
        neighbors_raw = inter_counts.get(item_id + 1, {})
        neighbors = {}
        for k, v in neighbors_raw.items():
            if 1 <= k <= num_items:
                neighbors[k] = v
            else:
                print(f"Warning: item id {k} out of range [1, {num_items}], skipping")
        pos_items = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)
        if pos_num == 1:
            pos_list = [x[0] for x in pos_items[:pos_num]]
        else:
            pos_list = [x for x in pos_items[:pos_num]]
        interacted = set(neighbors.keys())
        interacted.add(item_id + 1)
        neg_pool = [x for x in all_items if x not in interacted]
        rng.shuffle(neg_pool)
        neg_need = max(candidate_num - len(pos_list), 0)
        neg_list = neg_pool[:neg_need]

        # If still short, sample from remaining items (excluding itself and pos_list)
        if len(neg_list) < neg_need:
            fallback_pool = [x for x in all_items if x != item_id + 1 and x not in pos_list]
            rng.shuffle(fallback_pool)
            extra = fallback_pool[: max(neg_need - len(neg_list), 0)]
            neg_list.extend(extra)
        if pos_num > 1:
            neg_list = [(x,-1) for x in neg_list]

        candidates[item_id] = pos_list + neg_list

    return candidates


class GRPODataset(Dataset):
    def __init__(
        self,
        args,
        tokenizer,
        max_len: int = 512,
        sample: int = -1,
        seed: int = 0,
    ):
        self.args = args
        self.hf_tokenizer = tokenizer
        self.max_len = max_len
        self.sample = sample
        self.seed = seed
        self.user_prefix = getattr(args, "user_prefix", "")
        self.system_prompt = getattr(args, "system_prompt", "")
        self.dataset = args.dataset
        self.candidate_num = args.candidate_num
        self.pos_num = args.pos_num

        self.item_prompts = self._load_item_prompts()
        inter_file = os.path.join(_PROJECT_ROOT, "data", self.dataset, "handled", "inter.txt")
        self.candidates = get_item_inter_frequency(
            inter_file=inter_file,
            num_items=len(self.item_prompts),
            candidate_num=self.candidate_num,
            pos_num=self.pos_num,
            seed=seed,
        )

        self.item_indices = list(range(len(self.item_prompts)))
        if sample and sample > 0 and sample < len(self.item_indices):
            rng = random.Random(seed)
            rng.shuffle(self.item_indices)
            self.item_indices = self.item_indices[:sample]

        self.get_inputs()

    def __len__(self):
        return len(self.inputs)

    def _load_item_prompts(self) -> List[str]:
        item_file = os.path.join(_PROJECT_ROOT, "data", self.dataset, "handled", "item_info.jsonline")
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
        print("The first three prompts:\n", prompts[:3])
        return prompts

    def _encode_prompt(self, prompt: str):
        input_ids, attention_mask = tokenize_chat_text_generation(
            tokenizer=self.hf_tokenizer,
            text=prompt,
            system_prompt=self.system_prompt,
            user_prefix=self.user_prefix,
            max_source_length=1024,
            add_generation_prompt=False,
        )
        return input_ids, attention_mask

    def pre(self, item_id: int):
        prompt = self.item_prompts[item_id]
        tokens, attention_mask = self._encode_prompt(prompt)

        if self.pos_num == 1:
            candidate_ids = self.candidates[item_id]
        else:
            candidate_ids, frequency = zip(*self.candidates[item_id])
        candidate_input_ids = []
        candidate_attention_mask = []
        for cand_id in candidate_ids:
            cand_prompt = self.item_prompts[cand_id - 1]
            cand_tokens, cand_mask = self._encode_prompt(cand_prompt)
            candidate_input_ids.append(cand_tokens[-self.max_len :])
            candidate_attention_mask.append(cand_mask[-self.max_len :])

        if self.pos_num == 1:
            return {
                "item_id": item_id+1,
                "input_ids": tokens[-self.max_len :],
                "attention_mask": attention_mask[-self.max_len :],
                "candidate_ids": candidate_ids,
                "candidate_input_ids": candidate_input_ids,
                "candidate_attention_mask": candidate_attention_mask,
            }
        else:
            return {
                "item_id": item_id+1,
                "input_ids": tokens[-self.max_len :],
                "attention_mask": attention_mask[-self.max_len :],
                "candidate_ids": candidate_ids,
                "candidate_frequencies": frequency,
                "candidate_input_ids": candidate_input_ids,
                "candidate_attention_mask": candidate_attention_mask,
            }


    def get_inputs(self):
        inputs = []
        for idx in tqdm(self.item_indices):
            inputs.append(self.pre(idx))
        self.inputs = inputs

    def get_inputs_list(self):
        return self.inputs

    def __getitem__(self, idx):
        return self.inputs[idx]


class GRPOEvalDataset(Dataset):
    def __init__(
        self,
        args,
        tokenizer,
        max_len: int = 512,
        sample: int = -1,
        seed: int = 0,
    ):
        self.args = args
        self.hf_tokenizer = tokenizer
        self.max_len = max_len
        self.sample = sample
        self.seed = seed
        self.user_prefix = getattr(args, "user_prefix", "")
        self.system_prompt = getattr(args, "system_prompt", "")
        self.dataset = args.dataset

        self.item_prompts = self._load_item_prompts()
        self.item_indices = list(range(len(self.item_prompts)))
        if sample and sample > 0 and sample < len(self.item_indices):
            rng = random.Random(seed)
            rng.shuffle(self.item_indices)
            self.item_indices = self.item_indices[:sample]

        self.get_inputs()

    def __len__(self):
        return len(self.inputs)

    def _load_item_prompts(self) -> List[str]:
        item_file = os.path.join(_PROJECT_ROOT, "data", self.dataset, "handled", "item_info.jsonline")
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
        print("The first three prompts:\n", prompts[:3])
        return prompts

    def _encode_prompt(self, prompt: str):
        input_ids, attention_mask = tokenize_chat_text_generation(
            tokenizer=self.hf_tokenizer,
            text=prompt,
            system_prompt=self.system_prompt,
            user_prefix=self.user_prefix,
            max_source_length=1024,
            add_generation_prompt=False,
        )
        return input_ids, attention_mask

    def pre(self, item_id: int):
        prompt = self.item_prompts[item_id]
        tokens, attention_mask = self._encode_prompt(prompt)

        return {
            "item_id": item_id + 1,
            "inputs_ids": tokens[-self.max_len :],
            "attention_mask": attention_mask[-self.max_len :],
        }

    def get_inputs(self):
        inputs = []
        for idx in tqdm(self.item_indices):
            inputs.append(self.pre(idx))
        self.inputs = inputs

    def get_inputs_list(self):
        return self.inputs

    def __getitem__(self, idx):
        return self.inputs[idx]