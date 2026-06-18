import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Union

import transformers
from accelerate.utils import gather
from transformers import DataCollatorForSeq2Seq

from trl import GRPOTrainer
from trl.trainer.grpo_trainer import selective_log_softmax


def swap_adjacent_blocks(x, k):
	original_shape = x.shape
	x_2d = x.view(-1, k)
	n = x_2d.size(0)
	if n < 2:
		return x  # If only one block, no swap needed
	indices = torch.arange(n).view(-1, 2).flip(1).reshape(-1)
	return x_2d[indices].view(original_shape)

class NoiseGRPOEmbTrainer(GRPOTrainer):

	def __init__(self, prefix_allowed_tokens_fn, candidate_num: int, pos_num: int, noise_scale: float = 0.1, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)
		self.ref_model = self.model
		self.candidate_num = candidate_num
		self.pos_num = pos_num
		self.noise_scale = noise_scale

		data_collator = DataCollatorForSeq2Seq(
			self.processing_class, pad_to_multiple_of=8, return_tensors="pt", padding=True
		)

		def data_collate_fn(batch):
			base_batch = data_collator([
				{"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]} for b in batch
			])

			candidate_input_ids = [c for b in batch for c in b["candidate_input_ids"]]
			candidate_attention_mask = [c for b in batch for c in b["candidate_attention_mask"]]
			cand_padded = self.processing_class.pad(
				{"input_ids": candidate_input_ids, "attention_mask": candidate_attention_mask},
				padding=True,
				return_tensors="pt",
				pad_to_multiple_of=8,
			)
			bsz = len(batch)
			cand_num = len(batch[0]["candidate_input_ids"])
			base_batch["candidate_input_ids"] = cand_padded["input_ids"].view(bsz, cand_num, -1)
			base_batch["candidate_attention_mask"] = cand_padded["attention_mask"].view(bsz, cand_num, -1)
			base_batch["candidate_ids"] = torch.tensor([b["candidate_ids"] for b in batch], dtype=torch.long)
			if self.pos_num > 1:
				candidate_frequencies = [c for b in batch for c in b["candidate_frequencies"]]
				base_batch["candidate_frequencies"] = torch.tensor(candidate_frequencies, dtype=torch.long).view(bsz, cand_num)
			return base_batch

		self.data_collator = data_collate_fn
		self.prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
		self.generation_config = transformers.GenerationConfig(
			max_new_tokens=self.max_completion_length,
			do_sample=False,
			temperature=self.generation_config.temperature,
			pad_token_id=self.processing_class.pad_token_id,
		)

	def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
		logits = model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			logits_to_keep=logits_to_keep + 1,
		).logits
		logits = logits[:, :-1, :]
		input_ids = input_ids[:, -logits_to_keep:]
		logits = logits[:, -logits_to_keep:]
		logits = logits / self.temperature
		return selective_log_softmax(logits, input_ids)

	def my_get_per_token_logps(self, model, input_ids, inputs_embeds, attention_mask, logits_to_keep): # ?
		if hasattr(model, "module"):
			model = model.module
		model_to_call = model.model  # Use the underlying AutoModelForCausalLM

		outputs = model_to_call(
			input_ids=None,
			inputs_embeds=inputs_embeds,
			attention_mask=attention_mask,
			output_attentions=False,
			output_hidden_states=False,
			use_cache=False,
			logits_to_keep=logits_to_keep + 1,
		)
		logits = outputs.logits
		# logits = logits[:, :-1, :]  # Remove this line as inputs_embeds already provides the correct seq_len
		# if logits.size(0) != input_ids.size(0):
		# 	min_bsz = min(logits.size(0), input_ids.size(0))
		# 	logits = logits[:min_bsz]
		# 	input_ids = input_ids[:min_bsz]
		input_ids = input_ids[:, -logits_to_keep:]
		logits = logits[:, -logits_to_keep:]
		logits = logits / self.temperature
		# print("logits shape:", logits.shape)
		# print("input_ids shape:", input_ids.shape)
		return selective_log_softmax(logits, input_ids)

	def _mean_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
		mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
		summed = (hidden_states * mask).sum(dim=1) 
		denom = mask.sum(dim=1).clamp_min(1.0)
		return summed / denom

	def _get_item_embeds(self, model, input_ids, attention_mask, inputs_embeds=None):
		if inputs_embeds is not None:
			outputs = model(
				input_ids=None,
				attention_mask=attention_mask,
				inputs_embeds=inputs_embeds,
				output_hidden_states=True,
				use_cache=False,
				return_dict=True,
			)
		else:
			outputs = model(
				input_ids=input_ids,
				attention_mask=attention_mask,
				output_hidden_states=True,
				use_cache=False,
				return_dict=True,
			)
		last_hidden = outputs.hidden_states[-1]
		pool_type = getattr(model, "pool_type", "avg")
		if pool_type == "avg":
			return self._mean_pool(last_hidden, attention_mask)
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
				raise NotImplementedError("thought_token_id not found in model.")
			last_token_ids = input_ids[batch_idx, last_idx]
			if not torch.all(last_token_ids.eq(thought_id)):
				raise ValueError("pool_type='last' expects last valid token to be thought_id.")
			return last_hidden[batch_idx, last_idx]
		return self._mean_pool(last_hidden, attention_mask)

	def _generate_and_score_completions(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
		device = self.accelerator.device

		# print(f"Before _prepare_inputs: batch_size = {inputs['input_ids'].shape[0]}")
		prompt_inputs = super(GRPOTrainer, self)._prepare_inputs(inputs)
		# print(f"After _prepare_inputs: batch_size = {prompt_inputs['input_ids'].shape[0]}")
		input_ids = prompt_inputs["input_ids"]
		attention_mask = prompt_inputs["attention_mask"]
		candidate_input_ids = prompt_inputs["candidate_input_ids"]
		candidate_attention_mask = prompt_inputs["candidate_attention_mask"]

		batch_size = input_ids.size(0)
		num_prompts = batch_size // self.num_generations

		with torch.no_grad():
			original_embeds = self.model.generate_embs(input_ids, attention_mask)
			if hasattr(self.model, "thought_token_id"):
				thought_id = int(self.model.thought_token_id)
			else:
				raise NotImplementedError("thought_token_id not found in model.")
			where_thought_ids = torch.nonzero(input_ids == thought_id)
			noise = torch.randn(
				(batch_size, original_embeds.size(-1)), device=self.model.device
			).mul(self.noise_scale)
			if num_prompts > 0:
				for i in range(num_prompts):
					noise[i * self.num_generations, :] = 0
			original_embeds[torch.arange(batch_size), where_thought_ids[:, 1]] += noise

		logits_to_keep = input_ids.size(1)
		with torch.no_grad():
			old_per_token_logps = self.my_get_per_token_logps(
				self.model, input_ids, original_embeds, attention_mask, logits_to_keep
			)
			if self.beta == 0.0:
				ref_per_token_logps = None
			else:
				ref_per_token_logps = self.my_get_per_token_logps(
					self.ref_model, input_ids, original_embeds, attention_mask, logits_to_keep
				)

		with torch.no_grad():
			item_embs = self._get_item_embeds(
				self.model, input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=original_embeds
			)

			cand_bsz, cand_num, cand_len = candidate_input_ids.shape
			cand_ids_flat = candidate_input_ids.view(-1, cand_len)
			cand_mask_flat = candidate_attention_mask.view(-1, cand_len)
			cand_embs_flat = self._get_item_embeds(
				self.model, input_ids=cand_ids_flat, attention_mask=cand_mask_flat, inputs_embeds=None
			)
			cand_embs = cand_embs_flat.view(cand_bsz, cand_num, -1)

			# cand_norm = F.normalize(cand_embs, dim=-1)
			# sims = torch.einsum("bd,bkd->bk", item_norm, cand_norm)
			sims = torch.einsum("bd,bkd->bk", item_embs, cand_embs)
			if self.pos_num > 1:

				candidate_frequencies = inputs["candidate_frequencies"]  # shape: (batch_size, cand_num)
                # print("candidate_frequencies:", candidate_frequencies)

				pos_mask = candidate_frequencies.ne(-1)
				pos_frequencies = candidate_frequencies.float() * pos_mask.float()
				sum_pos = pos_frequencies.sum(dim=1, keepdim=True).clamp_min(1e-8)
				ratio = pos_frequencies / sum_pos
				neg_count = (1 - pos_mask.float()).sum(dim=1, keepdim=True).clamp_min(1e-8)
				rewards = torch.zeros((batch_size,), device=device, dtype=torch.float32)
				order = torch.argsort(sims, dim=1, descending=True)
				for i in range(batch_size):
					for rank, index in enumerate(order[i], start=0):
						if pos_mask[i, index]:
							rewards[i] += ratio[i][index] * (1.0 / (1.0 + torch.log2(torch.tensor(rank + 2.0, device=device))))
			else:
				order = torch.argsort(sims, dim=1, descending=True)
				pos_rank = torch.argmax((order == 0).to(torch.int64), dim=1)
				rewards = 1.0 / (1.0 + torch.log2(pos_rank.to(torch.float32) + 2.0))

		rewards = gather(rewards)
		# print("rewards shape:", rewards.shape)
		print(f"Sample rewards: {rewards[:16].cpu().numpy()}")  # Print first 16 rewards for debugging

		mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
		std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
		mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
		std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)

		temp_rewards = rewards.clone().view(-1, self.num_generations)
		xx = temp_rewards[:, 0].unsqueeze(1).expand_as(temp_rewards).reshape(-1)
		xx = swap_adjacent_blocks(xx, self.num_generations)
		advantages = rewards - xx.mean()
		advantages = advantages / (torch.norm(advantages) + 1e-6)

		process_slice = slice(
			self.accelerator.process_index * batch_size,
			(self.accelerator.process_index + 1) * batch_size,
		)
		advantages = advantages[process_slice]

		return {
			"input_ids": input_ids,
			"attention_mask": attention_mask,
			"old_per_token_logps": old_per_token_logps,
			"ref_per_token_logps": ref_per_token_logps,
			"advantages": advantages,
			"original_embeds": original_embeds,
			"noise": noise,
		}

	def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
		if return_outputs:
			raise ValueError("The GRPOTrainer does not support returning outputs")

		input_ids = inputs["input_ids"]
		attention_mask = inputs["attention_mask"]
		logits_to_keep = input_ids.size(1)

		noise = inputs["noise"]
		embeds = self.model.generate_embs(input_ids, attention_mask)
		if hasattr(self.model, "thought_token_id"):
			thought_id = int(self.model.thought_token_id)
		else:
			raise NotImplementedError("thought_token_id not found in model.")
		where_thought_ids = torch.nonzero(input_ids == thought_id)
		batch_size = input_ids.size(0)
		embeds[torch.arange(batch_size), where_thought_ids[:, 1]] += noise

		per_token_logps = self.my_get_per_token_logps(
			model, input_ids, embeds, attention_mask, logits_to_keep
		)

		if self.beta != 0.0:
			ref_per_token_logps = inputs["ref_per_token_logps"]
			per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

		advantages = inputs["advantages"]
		old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
		coef_1 = torch.exp(per_token_logps - old_per_token_logps)
		coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
		per_token_loss1 = coef_1 * advantages.unsqueeze(1)
		per_token_loss2 = coef_2 * advantages.unsqueeze(1)
		per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
		if self.beta != 0.0:
			per_token_loss = per_token_loss + self.beta * per_token_kl
		loss = (per_token_loss * attention_mask).sum() / attention_mask.sum()

		mode = "eval" if self.control.should_evaluate else "train"

		if self.beta != 0.0:
			mean_kl = (per_token_kl * attention_mask).sum() / attention_mask.sum()
			self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

		is_clipped = (per_token_loss1 < per_token_loss2).float()
		clip_ratio = (is_clipped * attention_mask).sum() / attention_mask.sum()
		self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
		return loss
