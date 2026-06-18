from dataclasses import dataclass
from typing import Any, List, Dict, Sequence, Tuple
import torch
import transformers
from transformers import DataCollatorForSeq2Seq

IGNORE_INDEX = -100


@dataclass
class LongestSequenceMaskCollator(object):
    """Collate tokenized examples into padded tensors."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        input_ids, labels, attention_mask = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "attention_mask"))
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=False
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )


@dataclass
class PairwiseDataCollatorWithPadding(LongestSequenceMaskCollator):

    tokenizer: transformers.PreTrainedTokenizer
    mse_loss: bool = False

    r"""Data collator for pairwise contrastive data."""

    def __call__(self, features: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        r"""Pads batched data to the longest sequence in the batch.

        Without regression, generate 2 * n examples: chosen then rejected. With regression,
        append the full input as a third group for each example.
        """
        # Build separate lists for chosen/rejected/(full) so we produce grouped batches
        chosen_list = []
        rejected_list = []
        full_list = []
        regression_labels: List[float] = []

        use_third = self.mse_loss

        # Collect entries; for classification/regression keep full_list values separately so we can
        # construct label tensors in the same grouped order as concatenated_features below.
        full_regression_values: List[float] = []

        for feature in features:
            # collect individual entries
            chosen_list.append(
                {
                    "input_ids": feature["chosen_ids"],
                    "attention_mask": feature["chosen_mask"],
                    "labels": feature["chosen_labels"],
                }
            )
            rejected_list.append(
                {
                    "input_ids": feature["rejected_ids"],
                    "attention_mask": feature["rejected_mask"],
                    "labels": feature["rejected_labels"],
                }
            )
            if use_third:
                full_list.append(
                    {
                        "input_ids": feature["full_ids"],
                        "attention_mask": feature["full_mask"],
                        "labels": feature["full_labels"],
                    }
                )
                if self.mse_loss:
                    r = feature.get("regression_labels", -100.0)
                    r = float(r) if r is not None else -100.0
                    full_regression_values.append(r)

        # concatenate grouped lists: all chosen, then all rejected, then all full (if present)
        concatenated_features = chosen_list + rejected_list
        if use_third:
            concatenated_features += full_list

        batch = super().__call__(concatenated_features)

        # Build regression tensors consistent with the grouped order.
        if self.mse_loss:
            if use_third:
                batch["regression_labels"] = torch.tensor(
                    [-100.0] * len(chosen_list) + [-100.0] * len(rejected_list) + [float(r) for r in full_regression_values],
                    dtype=torch.float,
                )
            else:
                batch["regression_labels"] = torch.tensor([-100.0] * batch["input_ids"].size(0), dtype=torch.float)

        return batch
    

