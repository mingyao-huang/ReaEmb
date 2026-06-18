from typing import Optional, List, Union, Tuple
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PreTrainedModel

from llm.r3_latent_thought import R3LatentThoughtAttention


class Qwen3RSEmb(PreTrainedModel):
    """HuggingFace-compatible wrapper around ``AutoModelForCausalLM`` that provides
    a sequence embedding interface and is compatible with PEFT and HF save/load.

    - Inherits from ``PreTrainedModel`` so it can be passed to PEFT helpers that
      expect a HF model.
    - Internally contains an ``AutoModelForCausalLM`` as ``self.model`` and uses
      ``outputs.hidden_states[-1]`` for pooling.
    """

    def __init__(
        self,
        config,
        model: Optional[AutoModelForCausalLM] = None,
        pool_type: str = "last",
        R3_think: bool = False,
        thought_token_id: int = -1,
        thought_end_k: int = -1,
        tau: float = 3.0,
        mse_loss: bool = False,
        mse_alpha: float = 1.0,
        num_groups: int = 5,
    ) -> None:
        super().__init__(config)
        # allow passing either a prepared AutoModelForCausalLM or none (load later)
        self.model = model
        self.pool_type = pool_type
        print(f"Pooling type: {self.pool_type}")

        self.R3_think = bool(R3_think)
        self.thought_token_id = int(thought_token_id)
        self.thought_end_k = int(thought_end_k)
        self.r3_attn = None
        if self.R3_think:
            if self.thought_token_id < 0:
                raise ValueError("R3_think=True requires a valid thought_token_id")
            self.r3_attn = R3LatentThoughtAttention(self.config.hidden_size, end_k=self.thought_end_k)
            if self.model is not None:
                self.r3_attn = self.r3_attn.to(self.model.dtype)

        self.tau = nn.Parameter(torch.tensor([float(tau)], dtype=torch.float32))
        if self.model is not None:
            self.tau = nn.Parameter(self.tau.to(self.model.dtype))

        self.num_groups = int(num_groups)

        self.mse_loss_enabled = bool(mse_loss)
        self.mse_alpha = float(mse_alpha)


        cls_out_dim = 1
        # if model provided, use its config, otherwise rely on passed config
        self.config = model.config if model is not None else config
        self.cls_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.config.hidden_size, cls_out_dim),
        )
        if self.model is not None:
            self.cls_head = self.cls_head.to(self.model.dtype)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        pool_type: str = "last",
        R3_think: bool = False,
        thought_token_id: int = -1,
        thought_end_k: int = -1,
        tau: float = 3.0,
        mse_loss: bool = False,
        mse_alpha: float = 1.0,
        num_groups: int = 5,
        **model_kwargs,
    ):
        """Convenience constructor: load an AutoModelForCausalLM and wrap it.

        Wrapper-specific arguments are consumed here and not
        forwarded into the underlying model's ``from_pretrained`` call. Remaining
        keyword arguments are passed to ``AutoModelForCausalLM.from_pretrained``.
        """
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, trust_remote_code=True, **model_kwargs
        )
        config = model.config

        try:
            setattr(config, "attn_implementation", "eager")
            # internal field used by Transformers' modeling_utils for the check
            setattr(config, "_attn_implementation_internal", "eager")
        except Exception:
            pass

        return cls(
            config=config,
            model=model,
            pool_type=pool_type,
            R3_think=R3_think,
            thought_token_id=thought_token_id,
            thought_end_k=thought_end_k,
            tau=tau,
            mse_loss=mse_loss,
            mse_alpha=mse_alpha,
            num_groups=num_groups,
        )

    def gradient_checkpointing_enable(self, *args, **kwargs):
        """Enable gradient checkpointing on the underlying model if supported."""
        if hasattr(self.model, "gradient_checkpointing_enable"):
            return self.model.gradient_checkpointing_enable(*args, **kwargs)
        raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")

    def enable_input_require_grads(self):
        """Ensure input embeddings require gradients (used by PEFT/LoRA)."""
        # Prefer delegating to underlying model if it provides a helper
        if hasattr(self.model, "enable_input_require_grads"):
            return self.model.enable_input_require_grads()
        emb = self.get_input_embeddings()
        if emb is None:
            return
        for p in emb.parameters():
            p.requires_grad = True

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def _inject_r3_thought_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        past_key_values,
        use_cache: Optional[bool],
        output_attentions: Optional[bool],
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """Run a first forward pass to get hidden states, then replace <|Thought|> embeddings.

        Returns a full-sequence inputs_embeds tensor (B, L, H) if <|Thought|> is present; otherwise None.
        """
        if not self.R3_think or self.r3_attn is None:
            return None
        if input_ids is None:
            return None
        if attention_mask is None:
            return None

        # Locate <|Thought|> per sample (expect exactly one, but handle missing safely)
        mask = input_ids.eq(self.thought_token_id)
        has = mask.any(dim=1)
        if not bool(has.any().item()):
            return None

        bsz, seq_len = input_ids.shape
        # Find the last <|Thought|> position.
        rev = torch.flip(mask, dims=[1])
        rev_idx = torch.argmax(rev.int(), dim=1)
        thought_pos = (seq_len - 1 - rev_idx).to(input_ids.device)
        thought_pos = torch.where(has, thought_pos, torch.full_like(thought_pos, -1))

        # First pass: get last hidden states from the base model.
        # NOTE: Under DeepSpeed ZeRO stage 1/2, using the same parameters in two autograd graphs
        # within a single training step can trigger "gradient computed twice" reduction asserts.
        # We only need hidden states as features to build the latent thought vector, so we run
        # this pass without gradients. Gradients still flow to r3_attn through the second pass.
        with torch.no_grad():
            mid = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )
        last_hidden = mid.hidden_states[-1]

        # Build embeddings and replace <|Thought|> token embedding with latent thought vector
        embed_layer = self.model.get_input_embeddings()
        input_embs = embed_layer(input_ids)
        # Under PEFT/DeepSpeed, inputs_embeds may be marked as a leaf tensor requiring grads.
        # Cloning avoids an in-place write on a leaf Variable.
        input_embs = input_embs.clone()
        thought_vec = self.r3_attn(last_hidden, attention_mask, thought_pos)

        batch_idx = torch.arange(bsz, device=input_ids.device)
        valid = thought_pos.ge(0) # (B,)
        if bool(valid.any().item()):
            input_embs[batch_idx[valid], thought_pos[valid]] = thought_vec[valid].to(input_embs.dtype)
        return input_embs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        group_labels: Optional[torch.LongTensor] = None,
        regression_labels: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        grpo: bool = False,
        **kwargs,
    ) -> dict:
        """Forward that uses the model's hidden states for pooling and contrastive loss.

        Important: this forward will explicitly request output_hidden_states=True to
        ensure the last transformer hidden layer is available as `outputs.hidden_states[-1]`.
        """
        return_dict = return_dict if return_dict is not None else True

        # Optional R3-style: generate a latent thought embedding for <|Thought|> then re-run model.
        if input_ids is not None and inputs_embeds is None and self.R3_think:
            injected = self._inject_r3_thought_embeddings(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs,
            )
            if injected is not None:
                inputs_embeds = injected
                input_ids = None

        # Force returning hidden states for stable embedding extraction
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        if grpo:
            return outputs
        # For AutoModelForCausalLM outputs.hidden_states is a tuple of layer outputs; last index is last layer
        last_hidden = outputs.hidden_states[-1]

        # compute batch size and sequence lengths robustly to left/right padding
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if getattr(self.config, "pad_token_id", None) is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")

        if getattr(self.config, "pad_token_id", None) is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                pad_id = self.config.pad_token_id
                mask = input_ids.ne(pad_id)
                rev_mask = torch.flip(mask, dims=[1])
                rev_idx = torch.argmax(rev_mask.int(), dim=1)
                seq_len = input_ids.size(1)
                sequence_lengths = (seq_len - 1 - rev_idx).to(input_ids.device)
            else:
                sequence_lengths = -1

        pooled_mask = attention_mask
        # if not getattr(self, "_printed_last_hidden_pool_debug", False):
        #     print("----------First Batch last_hidden (before _get_pool_emb) ----------")
        #     print("last_hidden shape:", last_hidden.shape)
        #     print("last_hidden[0]:", last_hidden[0])
        pooled_logits = self._get_pool_emb(last_hidden, sequence_lengths, pooled_mask)
        # if not getattr(self, "_printed_last_hidden_pool_debug", False):
        #     print("----------First Batch last_hidden (after _get_pool_emb) ----------")
        #     print("last_hidden shape:", pooled_logits.shape)
        #     print("last_hidden[0]:", pooled_logits[0])
        #     self._printed_last_hidden_pool_debug = True

        total_batch = pooled_logits.size(0)
        if self.mse_loss_enabled:
            if total_batch % 3 != 0:
                raise ValueError("Expected 3x samples per item when classification/regression is enabled.")
            pair_batch = total_batch // 3
            chosen_logits = pooled_logits[:pair_batch]
            rejected_logits = pooled_logits[pair_batch : pair_batch * 2]
            cls_inputs = pooled_logits[pair_batch * 2 : pair_batch * 3]
            cls_logits = self.cls_head(cls_inputs)
        else:
            if total_batch % 2 != 0:
                raise ValueError("Expected 2x samples per item when classification/regression is not enabled.")
            pair_batch = total_batch // 2
            chosen_logits, rejected_logits = pooled_logits.split(pair_batch, dim=0)

            # Simple debug: print first 3 input_ids rows for chosen and rejected, only once per model instance
            # if not getattr(self, "_pairwise_debug_printed", False):
            #     with torch.no_grad():
            #         n_print = min(3, pair_batch)
            #         if input_ids is not None:
            #             chosen_ids = input_ids[:pair_batch][:n_print].cpu().tolist()
            #             rejected_ids = input_ids[pair_batch:pair_batch*2][:n_print].cpu().tolist()
            #             print(f"[Pairwise Debug] chosen input_ids (first {n_print}): {chosen_ids}")
            #             print(f"[Pairwise Debug] rejected input_ids (first {n_print}): {rejected_ids}")
            #     self._pairwise_debug_printed = True

            cls_logits = None

        loss = None
        contrastive_loss_value = None
        mse_loss_value = None
        if labels is not None:
            loss_fct = Contrastive_Loss(self.tau)
            contrastive_loss = loss_fct(chosen_logits, rejected_logits)
            contrastive_loss_value = contrastive_loss.detach()
            if self.mse_loss_enabled and cls_logits is not None and regression_labels is not None:
                reg_targets = regression_labels[pair_batch * 2 : pair_batch * 3]
                valid_mask = reg_targets != -100
                if valid_mask.any():
                    reg_loss = F.mse_loss(cls_logits.squeeze(-1)[valid_mask], reg_targets[valid_mask])
                    mse_loss_value = reg_loss.detach()
                    loss = (contrastive_loss + self.mse_alpha * reg_loss) / (1.0 + self.mse_alpha)
                else:
                    loss = contrastive_loss
            else:
                loss = contrastive_loss

        return {
            "loss": loss,
            "logits": pooled_logits,
            "past_key_values": outputs.past_key_values,
            "hidden_states": pooled_logits,
            "attentions": outputs.attentions,
            "contrastive_loss": contrastive_loss_value,
            "mse_loss": mse_loss_value,
        }

    def _get_pool_emb(self, hidden_states, sequence_lengths, pooled_mask):
        """hidden_states: (batch_size, seq_len, hidden_size)
        sequence_lengths: (batch_size,) - indices of last valid token per sequence
        pooled_mask: (batch_size, seq_len) - mask for valid tokens (1)"""
        """get the logits according to pool type"""
        if pooled_mask is None:
            pooled_mask = torch.ones(hidden_states.shape[:2], device=hidden_states.device, dtype=hidden_states.dtype)
        if self.pool_type == "last":
            pooled_emb = hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), sequence_lengths]
        elif self.pool_type == "avg":
            pooled_emb = (
                torch.sum(hidden_states * pooled_mask.unsqueeze(-1), dim=1)
                / torch.sum(pooled_mask, dim=1).unsqueeze(-1)
            )
        else:
            raise ValueError(f"Unsupported pool type: {self.pool_type}")

        return pooled_emb

class Contrastive_Loss(nn.Module):

    def __init__(self, tau=1) -> None:
        super().__init__()

        self.temperature = tau

    def forward(self, X, Y):
        safe_tau = torch.clamp(self.temperature, min=1e-3)
        logits = (X @ Y.T) / safe_tau
        X_similarity = Y @ Y.T
        Y_similarity = X @ X.T
        targets = F.softmax((X_similarity + Y_similarity) / (2 * safe_tau), dim=-1)
        X_loss = self.cross_entropy(logits, targets, reduction="none")
        Y_loss = self.cross_entropy(logits.T, targets.T, reduction="none")
        loss = (Y_loss + X_loss) / 2.0
        return loss.mean()

    def cross_entropy(self, preds, targets, reduction="none"):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")
        

class Contrastive_Loss_2(nn.Module):

    def __init__(self, tau: float = 0.5) -> None:
        super().__init__()
        self.tau = max(float(tau), 1e-3)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        normalized_X = F.normalize(X, dim=-1)
        normalized_Y = F.normalize(Y, dim=-1)

        logits_XY = normalized_X @ normalized_Y.T / self.tau
        logits_YX = normalized_Y @ normalized_X.T / self.tau

        labels = torch.arange(normalized_X.size(0), device=normalized_X.device)
        loss_XY = F.cross_entropy(logits_XY, labels)
        loss_YX = F.cross_entropy(logits_YX, labels)

        return 0.5 * (loss_XY + loss_YX)