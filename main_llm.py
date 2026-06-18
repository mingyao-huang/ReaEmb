import os
import json
import torch
from datasets import load_dataset
from datasets import builder as hf_builder
from datasets import filesystems as hf_filesystems

# Patch datasets' filesystem helper so LocalFileSystem protocols expressed as tuples
# (e.g., ('file', 'local')) are treated as local rather than "remote".
_original_is_remote_fs = hf_filesystems.is_remote_filesystem


def _patched_is_remote_filesystem(fs):
    protocol = getattr(fs, "protocol", None)
    if isinstance(protocol, (tuple, list, set)) and "file" in protocol:
        return False
    return _original_is_remote_fs(fs)


hf_filesystems.is_remote_filesystem = _patched_is_remote_filesystem
hf_builder.is_remote_filesystem = _patched_is_remote_filesystem

from llm.peft import (
    LoraConfig,
    PeftModel,
)
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from transformers import AutoTokenizer
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from llm.qwen3 import QwenRSEmb
from llm.trainer_seq2seq import MedRecTrainer
from llm.lora_cls import PeftModelForCLS
from llm.arguments import DataTrainingArguments, ModelArguments
from llm.data_processor.qwen import qwen_train_mask, qwen_eval_mask
from llm.data_processor.collator import LongestSequenceMaskCollator, PairwiseDataCollatorWithPadding


# save model for PeftModel
class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.is_world_process_zero:
            print('+++++++++++++++++save call back++++++++++++++++')
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )
            kwargs["model"].save_pretrained(checkpoint_folder)

            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)
            return control
        

def train():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if getattr(model_args, "mse", None) is not None:
        model_args.mse_loss = bool(model_args.mse)
    device_map = "auto"

    ## Load Tokenizer ##
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token is not None else tokenizer.eos_token
    if tokenizer.pad_token_id is None and tokenizer.pad_token is not None:
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = "left"  # define the padding direction

    # Optional R3-style latent thought token
    if getattr(model_args, "R3_think", False):
        additional_special_tokens = list(getattr(tokenizer, "additional_special_tokens", []) or [])
        if "<|Thought|>" not in additional_special_tokens:
            additional_special_tokens.append("<|Thought|>")
            num_added_toks = tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
            print(f"[Info] R3_think=True: added {num_added_toks} special tokens (including <|Thought|>)")
        thought_token_id = tokenizer.convert_tokens_to_ids("<|Thought|>")
        if thought_token_id is None or thought_token_id == tokenizer.unk_token_id:
            raise ValueError("R3_think=True but <|Thought|> token id could not be resolved")

    ## Load Model ##
    if model_args.model_choice == "rsemb":

        if training_args.bf16:
            model_dtype = torch.bfloat16
        elif training_args.fp16:
            model_dtype = torch.float16
        else:
            model_dtype = torch.float32

        model = QwenRSEmb.from_pretrained(
            model_args.model_name_or_path,
            pool_type=model_args.pool_type,
            R3_think=getattr(model_args, "R3_think", False),
            thought_token_id=int(thought_token_id) if getattr(model_args, "R3_think", False) else -1,
            tau=model_args.tau,
            mse_loss=model_args.mse_loss,
            mse_alpha=model_args.mse_alpha,
            num_groups=model_args.num_groups,
            torch_dtype=model_dtype,
        ).to(dtype=model_dtype)
        if model_args.mse_loss:
            print("[Info] Using contrastive + regression loss (mse_loss=True).")
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id

        # Ensure embedding table matches tokenizer after adding <|Thought|>
        if getattr(model_args, "R3_think", False):
            model.resize_token_embeddings(len(tokenizer))

        if model_args.peft_path is not None:
            # Resume training.
            if training_args.resume_from_checkpoint is not None:
                model = PeftModelForCLS.from_pretrained(model, model_args.peft_path, is_trainable=True)
            else:
                model = PeftModelForCLS.from_pretrained(model, model_args.peft_path, is_trainable=False)
        else:
            if training_args.do_train:
                # Load Lora Config
                print("[Info] Training with LoRA on base model.")

                modules_to_save = []
                if getattr(model_args, "modules_to_save", None):
                    modules_to_save = [m.strip() for m in str(model_args.modules_to_save).split(",") if m.strip()]
                if getattr(model_args, "R3_think", False):
                    modules_to_save = list(dict.fromkeys(modules_to_save + ["r3_attn"]))

                peft_config = LoraConfig(
                    r=model_args.lora_rank,
                    lora_alpha=model_args.lora_alpha,
                    target_modules=model_args.trainable.split(","),
                    lora_dropout=model_args.lora_dropout,
                    modules_to_save=modules_to_save if modules_to_save else None,
                    task_type="SEQ_CLS",
                )
                model = PeftModelForCLS(model, peft_config)  # LoRA-wrapped Qwen encoder
            else:
                print("[Info] Testing with base model only (no LoRA).")

    else:

        raise ValueError("No such LLM model")

    

    if training_args.do_train:
        for name, param in model.named_parameters():
            if "head_attn" in name:
                param.requires_grad = True
            if "tau" in name:
                try:
                    param.requires_grad = True
                except:
                    pass
            if "item_wte" in name:
                param.requires_grad = True
            if "projector" in name:
                param.requires_grad = True
            if "cls_head" in name:
                param.requires_grad = True

            if getattr(model_args, "R3_think", False) and "r3_attn" in name:
                param.requires_grad = True

    # model.print_trainable_parameters()

    ## Load Dataset ##
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file

    ds_kwargs = {
        "data_files": data_files,
        "cache_dir": model_args.cache_dir,
    }
    if model_args.use_auth_token:
        ds_kwargs["use_auth_token"] = True
    raw_datasets = load_dataset("json", **ds_kwargs)
    print("raw_datasets: ", raw_datasets)

    if training_args.do_train:
        target_dataset = raw_datasets["train"]
    elif training_args.do_eval:
        target_dataset = raw_datasets["eval"]
    elif training_args.do_predict:
        target_dataset = raw_datasets["test"]
    
    if training_args.do_train:
        preprocess_func = qwen_train_mask(data_args, model_args, tokenizer)
        data_collator = PairwiseDataCollatorWithPadding(tokenizer, mse_loss=model_args.mse_loss)

    else:
        preprocess_func = qwen_eval_mask(data_args, model_args, tokenizer)
        data_collator = LongestSequenceMaskCollator(tokenizer)

    with training_args.main_process_first(desc="Dataset map pre-processing"):
        target_dataset = target_dataset.map(
            preprocess_func,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            desc="Running tokenizer on prediction dataset",
            load_from_cache_file=False,
        )
    target_dataset.set_format("torch")

    training_args.remove_unused_columns = False

    ## Set Trainer ##
    trainer = MedRecTrainer(
        model=model,
        args=training_args,
        train_dataset=target_dataset if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
        callbacks=([SavePeftModelCallback] if isinstance(model, PeftModel) else None),
    )

    # Print trainable parameter summary so we can confirm LoRA params are active.
    if training_args.do_train and trainer.is_world_process_zero():
        total_params = 0
        trainable_params = 0
        lora_params = []
        for n, p in model.named_parameters():
            num = p.numel()
            total_params += num
            if p.requires_grad:
                trainable_params += num
                if "lora_" in n:
                    lora_params.append(n)
        print(f"[Info] total params: {total_params}, trainable params: {trainable_params}")
        print(f"[Info] example trainable param names (up to 20): {lora_params[:20]}")

        # # Quick, safe diagnostic: use a tiny in-memory DataLoader (num_workers=0) to avoid blocking heavy IO
        # try:
        #     if trainer.train_dataset is None or len(trainer.train_dataset) == 0:
        #         raise RuntimeError("No training dataset available for diagnostics")
        #     ds = trainer.train_dataset.select(range(min(2, len(trainer.train_dataset))))
        #     dl = torch.utils.data.DataLoader(ds, batch_size=min(2, len(ds)), collate_fn=trainer.data_collator, num_workers=0)
        #     batch = next(iter(dl))
        #     # move tensors to model device
        #     batch = {k: (v.to(trainer.model.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        #     print(f"[Diag] batch keys/shapes: {{k: (v.shape if isinstance(v, torch.Tensor) else type(v)) for k,v in batch.items()}}")
        #     loss, outputs = trainer.compute_loss(trainer.model, batch, return_outputs=True)
        #     print(f"[Diag] sample loss: {loss.item()}")
        #     pooled = outputs.get("logits")
        #     if pooled is not None:
        #         print(f"[Diag] pooled shape: {pooled.shape}, mean: {pooled.mean().item():.6e}, std: {pooled.std().item():.6e}")
        #         pair_batch = pooled.size(0) // 2
        #         chosen = pooled[:pair_batch]
        #         rejected = pooled[pair_batch : pair_batch * 2]
        #         print(f"[Diag] chosen mean/std: {chosen.mean().item():.6e}/{chosen.std().item():.6e}")
        #         print(f"[Diag] rejected mean/std: {rejected.mean().item():.6e}/{rejected.std().item():.6e}")
        #     # backward
        #     trainer.model.zero_grad()
        #     loss.backward()
        #     lora_grad_nonzero = 0
        #     max_grad = 0.0
        #     for n, p in trainer.model.named_parameters():
        #         if "lora_" in n:
        #             if p.grad is not None and p.grad.norm().item() > 0:
        #                 lora_grad_nonzero += 1
        #                 gnorm = float(p.grad.norm().item())
        #                 if gnorm > max_grad:
        #                     max_grad = gnorm
        #     print(f"[Diag] LoRA nonzero grads: {lora_grad_nonzero}, max grad norm: {max_grad:.6e}")
        # except Exception as e:
        #     print(f"[Diag] diagnostics failed: {e}")

    ## Train Model
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        if hasattr(model, "gen_model") and model.gen_model is not None:
            model.gen_model.gradient_checkpointing_enable()
            model.gen_model.enable_input_require_grads()
        else:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_state()


    ## Evaluation ##
    results = {}

    if training_args.do_predict:

        if model_args.model_choice == "rsemb":

            list_test_samples = []
            with open(data_args.test_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = json.loads(line)
                    list_test_samples.append(line)

            # start_time = time.time()
            with torch.no_grad():
                predict_results = trainer.predict(
                    target_dataset,
                    metric_key_prefix="predict",
                )
            # end_time = time.time()

            if trainer.is_world_process_zero():
                predictions = predict_results.predictions
                assert len(predictions) == len(list_test_samples)
                hidden_states = predict_results.label_ids

                output_prediction_file = os.path.join(training_args.output_dir, model_args.output_file)

                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    for idx, p in enumerate(predictions):
                        samp = list_test_samples[idx]
                        samp["hidden_states"] = hidden_states[idx].astype(float).tolist()
                        samp["target"] = p.astype(float).tolist()
                        res = json.dumps(samp, ensure_ascii=False)
                        writer.write(f"{res}\n")

                results = None

    return results


if __name__ == "__main__":

    train()
