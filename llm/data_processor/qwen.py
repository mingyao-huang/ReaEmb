import numpy as np
from tqdm import tqdm
import copy
import random


def resolve_token_id(tokenizer, token: str):
    try:
        token_id = tokenizer.convert_tokens_to_ids(token)
    except KeyError:
        return None
    if isinstance(token_id, list):
        token_id = token_id[0]
    if token_id is None or token_id == tokenizer.unk_token_id:
        return None
    return int(token_id)


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



def tokenize_chat_text_generation(tokenizer, text, data_args, model_args=None, add_generation_prompt: bool = True):
    """Tokenize an item prompt for embedding extraction and training."""
    base_templated = apply_chat_template(
        tokenizer,
        text,
        getattr(data_args, "system_prompt", ""),
        getattr(data_args, "user_prefix", ""),
        add_generation_prompt=add_generation_prompt,
    )

    append_r3_thought = bool(getattr(model_args, "R3_think", False)) if model_args is not None else False

    encoded = tokenizer(base_templated, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Reserve space for optional <|Thought|> plus eos.
    reserved = 1 + (1 if append_r3_thought else 0)
    max_source_len = data_args.max_source_length - reserved
    if len(input_ids) > max_source_len:
        input_ids = input_ids[:max_source_len]
        attention_mask = attention_mask[:max_source_len]


    if append_r3_thought:
        thought_id = resolve_token_id(tokenizer, "<|Thought|>")
        if thought_id is not None:
            input_ids = input_ids + [thought_id]
            attention_mask = attention_mask + [1]
    

    attention_bool = [bool(m) for m in attention_mask]

    input_ids = input_ids + [tokenizer.eos_token_id]
    attention_bool = attention_bool + [True]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    labels = [pad_id] * (len(input_ids) - 1) + [tokenizer.eos_token_id]
    if getattr(data_args, "ignore_pad_token_for_loss", False):
        labels = [(l if l != pad_id else -100) for l in labels]

    return input_ids, labels, attention_bool


class qwen_train_mask(object):
    def __init__(self, data_args, model_args, tokenizer) -> None:
        self.data_args = data_args
        self.model_args = model_args
        self.prompt_column = "input"
        self.response_column = "target"
        self.regression_column = getattr(data_args, "regression_column", "score")
        self.tokenizer = tokenizer
        self.user_prefix = getattr(data_args, "user_prefix", "")
    def __call__(self, examples):
        max_seq_length = self.data_args.max_source_length + self.data_args.max_target_length
        model_inputs = {
            "chosen_ids": [],
            "rejected_ids": [],
            "chosen_labels": [],
            "rejected_labels": [],
            "chosen_mask": [],
            "rejected_mask": [],
        }
        if self.model_args.mse_loss:
            model_inputs.update(
                {
                    "full_ids": [],
                    "full_labels": [],
                    "full_mask": [],
                }
            )
            if self.model_args.mse_loss:
                model_inputs["regression_labels"] = []

        for i in range(len(examples[self.prompt_column])):
            if examples[self.prompt_column][i]:
                query, answer = examples[self.prompt_column][i], examples[self.response_column][i]
                chosen, rejected = dropout_feature(query, self.data_args.dropout_ratio)
                while chosen == rejected:
                    chosen, rejected = dropout_feature(query, self.data_args.dropout_ratio)
                if chosen == rejected:
                    raise ValueError("After dropout, chosen and rejected are still identical.")
                full_query = query
                if self.user_prefix:
                    parts = query.split("\n", 1)
                    if len(parts) > 1:
                        full_query = parts[1]
                    else:
                        raise ValueError("The input query must two lines when user_prefix is set.")
                    
                chosen_input_ids, chosen_labels, chosen_mask = tokenize_chat_text_generation(
                    self.tokenizer,
                    chosen,
                    self.data_args,
                    model_args=self.model_args,
                    add_generation_prompt=False,
                )
                rejected_input_ids, rejected_labels, rejected_mask = tokenize_chat_text_generation(
                    self.tokenizer,
                    rejected,
                    self.data_args,
                    model_args=self.model_args,
                    add_generation_prompt=False,
                )
                full_input_ids, full_labels, full_mask = tokenize_chat_text_generation(
                    self.tokenizer,
                    full_query,
                    self.data_args,
                    model_args=self.model_args,
                    add_generation_prompt=False,
                )

                model_inputs["chosen_ids"].append(chosen_input_ids)
                model_inputs["rejected_ids"].append(rejected_input_ids)
                model_inputs["chosen_labels"].append(chosen_labels)
                model_inputs["rejected_labels"].append(rejected_labels)
                model_inputs["chosen_mask"].append(chosen_mask)
                model_inputs["rejected_mask"].append(rejected_mask)
                if self.model_args.mse_loss:
                    model_inputs["full_ids"].append(full_input_ids)
                    model_inputs["full_labels"].append(full_labels)
                    model_inputs["full_mask"].append(full_mask)
                    if self.model_args.mse_loss:
                        if self.regression_column in examples:
                            reg_value = examples[self.regression_column][i]
                        elif "group" in examples:
                            reg_value = examples["group"][i]
                        else:
                            raise ValueError(
                                f"`{self.regression_column}` or `group` field is required in the dataset when mse_loss is enabled."
                            )
                        if reg_value is None:
                            raise ValueError("Found None regression value while mse_loss is enabled.")
                        model_inputs.setdefault("regression_labels", []).append(float(reg_value))

        return model_inputs


def dropout_feature(item_str, ratio):
    """Shuffle/drop features and rebuild with provided user_prefix replacing original instruction."""
    parts = item_str.split("\n", 1)
    instruction = parts[0]
    feat_part = parts[1] if len(parts) > 1 else ""
    feat_list = feat_part.split(";") if feat_part else []

    feat_list_1 = copy.deepcopy(feat_list)
    feat_list_2 = copy.deepcopy(feat_list)
    random.shuffle(feat_list_1)
    random.shuffle(feat_list_2)

    dropout_N = int(len(feat_list) * ratio)
    for _ in range(min(dropout_N, len(feat_list_1))):
        feat_list_1.pop()
    for _ in range(min(dropout_N, len(feat_list_2))):
        feat_list_2.pop()
    item_str_1 = ""
    item_str_2 = ""
    if len(feat_list) > 0:
        for val in feat_list_1:
            if val:
                item_str_1 += val + ";"
        for val in feat_list_2:
            if val:
                item_str_2 += val + ";"
    return item_str_1[:-1], item_str_2[:-1]



class qwen_eval_mask(object):
    def __init__(self, data_args, model_args, tokenizer) -> None:
        self.data_args = data_args
        self.model_args = model_args
        self.prompt_column = "input"
        self.response_column = "target"
        self.history_column = None
        self.tokenizer = tokenizer
        self.user_prefix = getattr(data_args, "user_prefix", "")
    def __call__(self, examples):
        max_seq_length = self.data_args.max_source_length + self.data_args.max_target_length
        model_inputs = {
            "input_ids": [],
            "labels": [],
            "attention_mask": []
        }

        for i in range(len(examples[self.prompt_column])):
            if examples[self.prompt_column][i]:
                query, answer = examples[self.prompt_column][i], examples[self.response_column][i]
                input_ids, _, attention_mask = tokenize_chat_text_generation(
                    self.tokenizer,
                    query,
                    self.data_args,
                    model_args=self.model_args,
                    add_generation_prompt=False,
                )

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(input_ids)
                model_inputs["attention_mask"].append(attention_mask)

        return model_inputs
