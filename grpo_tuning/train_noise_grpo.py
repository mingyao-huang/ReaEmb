import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.dirname(current_dir))

import fire
import torch
import trl
from datasets import Dataset as HFDataset
from transformers import AutoTokenizer

from model import Qwen3RSEmbForGRPO
from grpo_dataset import GRPODataset
from grpo_trainer import NoiseGRPOEmbTrainer

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")

def train(
	base_model: str = "../qwen2.5_0.5b",
	lora_path: str = "../saved/lora-0123_no_think_w_R3/checkpoint-2000",
	output_dir: str = "./output_grpo/",
	dataset: str = "fashion",
	user_prefix: str = "### Instruction:\n Please provide the item description. The item information is: ",
	system_prompt: str = "You are a specialist who summarizes product texts into concise, high-quality item descriptions.",
	candidate_num: int = 10,
	pos_num: int = 1,
	cutoff_len: int = 128,
	sample: int = -1,
	seed: int = 42,
	lr: float = 6e-6,
	end_k: int = -1,
	num_epochs: int = 1,
	batch_size: int = 8,
	micro_batch_size: int = 1,
	num_generations: int = 8,
	beta: float = 0.00,
	num_iterations: int = 2,
	epsilon: float = 0.2,
	epsilon_high: float = 0.28,
	noise_scale: float = 0.1,
	max_steps: int = -1,
	max_completion_length: int = 1,
	use_vllm: bool = False,
	vllm_gpu_memory_utilization: float = 0.7,
	r3_think: bool = True,
	pool_type: str = "last",
	resume_from_checkpoint: str = None,
	skip_train: bool = False,
	save_steps: int = 200,
	save_total_limit: int = 2,
	local_rank: int = 0,
):
	device_map = "auto"
	print("micro_batch_size=", micro_batch_size,"\n")
	gradient_accumulation_steps = batch_size // micro_batch_size
	world_size = int(os.environ.get("WORLD_SIZE", 1))
	ddp = world_size != 1
	if ddp:
		gradient_accumulation_steps = gradient_accumulation_steps // world_size

	assert batch_size == micro_batch_size * gradient_accumulation_steps * world_size

	tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
	if "<|Thought|>" not in tokenizer.additional_special_tokens:
		additional_special_tokens = tokenizer.additional_special_tokens
		additional_special_tokens.append("<|Thought|>")
		tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})

	thought_token_id = tokenizer.convert_tokens_to_ids("<|Thought|>")
	if thought_token_id is None or thought_token_id == tokenizer.unk_token_id:
		raise ValueError("Failed to resolve <|Thought|> token id for GRPO.")

	model = Qwen3RSEmbForGRPO.from_pretrained(
		base_model,
		R3_think=r3_think,
		pool_type=pool_type,
		thought_token_id=int(thought_token_id),
		thought_end_k=end_k,
		torch_dtype=torch.bfloat16,
	)

	if len(tokenizer) != model.get_input_embeddings().num_embeddings:
		model.resize_token_embeddings(len(tokenizer))

	if lora_path:
		try:
			from llm.lora_cls import PeftModelForCLS
		except Exception as exc:
			raise RuntimeError("peft is required to load LoRA adapters. Please install peft.") from exc
		# Load LoRA adapter weights from SFT and merge into base model
		model = PeftModelForCLS.from_pretrained(model, lora_path, is_trainable=False)
		model = model.merge_and_unload()

	# Freeze base LM weights
	if hasattr(model, "model"):
		for param in model.model.parameters():
			param.requires_grad = False
		if hasattr(model.model, "lm_head"):
			for param in model.model.lm_head.parameters():
				param.requires_grad = False
		if hasattr(model.model, "cls_head"):
			for param in model.model.cls_head.parameters():
				param.requires_grad = False
	else:
		for param in model.parameters():
			param.requires_grad = False
		if hasattr(model, "lm_head"):
			for param in model.lm_head.parameters():
				param.requires_grad = False
		if hasattr(model, "cls_head"):
			for param in model.cls_head.parameters():
				param.requires_grad = False

	def count_trainable_params(model):
		return sum(p.numel() for p in model.parameters() if p.requires_grad)
	trainable_params = count_trainable_params(model)
	print(f"Trainable parameters: {trainable_params}")
	trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
	print(f"Trainable param names (first 10): {trainable_names[:10]}")

	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.pad_token_id = tokenizer.eos_token_id
	tokenizer.padding_side = "left"

	if skip_train:
		process_rank = int(os.environ.get("LOCAL_RANK", local_rank))
		if process_rank in (-1, 0):
			os.makedirs(output_dir, exist_ok=True)
			model.config.use_cache = False
			model.save_pretrained(output_dir, safe_serialization=True)
			tokenizer.save_pretrained(output_dir)
			state_dict = model.state_dict()
			torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
			print("skip_train=True: saved initialized model and tokenizer to {}".format(output_dir))
		return

	class Args:
		pass

	args = Args()
	args.system_prompt = system_prompt
	args.user_prefix = user_prefix
	args.dataset = dataset
	args.candidate_num = candidate_num
	args.pos_num = pos_num

	train_data = GRPODataset(
		args=args,
		tokenizer=tokenizer,
		max_len=cutoff_len,
		sample=sample,
		seed=seed,
	)
	val_data = GRPODataset(
		args=args,
		tokenizer=tokenizer,
		max_len=cutoff_len,
		sample=sample,
		seed=seed,
	)

	hf_train_dataset = HFDataset.from_dict({k: [v[k] for v in train_data] for k in train_data[0].keys()})
	hf_val_dataset = HFDataset.from_dict({k: [v[k] for v in val_data] for k in val_data[0].keys()})

	def reward_placeholder(*_args, **_kwargs):
		return 0

	trainer = NoiseGRPOEmbTrainer(
		prefix_allowed_tokens_fn=None,
		candidate_num=candidate_num,
		pos_num=pos_num,
		noise_scale=noise_scale,
		model=model,
		reward_funcs=[reward_placeholder],
		args=trl.GRPOConfig(
			warmup_steps=200,
			num_generations=num_generations,
			max_prompt_length=cutoff_len,
			temperature=1.0,
			max_completion_length=max_completion_length,
			use_vllm=use_vllm,
			vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
			learning_rate=lr,
			beta=beta,
			num_iterations=num_iterations,
			epsilon=epsilon,
			epsilon_high=epsilon_high,
			num_train_epochs=num_epochs,
			max_steps=max_steps,
			output_dir=output_dir,
			per_device_train_batch_size=micro_batch_size,
			per_device_eval_batch_size=micro_batch_size,
			seed=seed,
			gradient_accumulation_steps=gradient_accumulation_steps,
			lr_scheduler_type="cosine",
			eval_strategy="no",
			save_steps=save_steps,
			save_total_limit=save_total_limit,
			load_best_model_at_end=False,
			ddp_find_unused_parameters=True,
			report_to="none",
			bf16=True,
			logging_steps=1,
			disable_tqdm=False,
		),
		train_dataset=hf_train_dataset,
		eval_dataset=hf_val_dataset,
		processing_class=tokenizer,
	)

	model.config.use_cache = False

	checkpoint = resume_from_checkpoint
	train_result = trainer.train(resume_from_checkpoint=checkpoint)
	metrics = train_result.metrics
	metrics["train_samples"] = len(hf_train_dataset)
	trainer.log_metrics("train", metrics)
	trainer.save_metrics("train", metrics)
	trainer.save_state()
	model.save_pretrained(output_dir, safe_serialization=True)
	tokenizer.save_pretrained(output_dir)

	# Save full state_dict to ensure all parameters are saved
	state_dict = model.state_dict()
	torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))


if __name__ == "__main__":
	fire.Fire(train)
