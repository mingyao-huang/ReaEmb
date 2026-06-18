import torch
import pickle
import numpy as np
import torch.nn as nn

from models.BaseModel import BaseSeqModel


class PoolRec(BaseSeqModel):
    """Sequence model that aggregates item embeddings via simple mean pooling."""

    def __init__(self, user_num, item_num, device, args):
        super().__init__(user_num, item_num, device, args)

        if args.emb_argu:
            print("Loading LLM-based item embeddings from {}".format(args.llm_emb_path))
            llm_item_emb = pickle.load(open(args.llm_emb_path, "rb"))
            if isinstance(llm_item_emb, list):
                llm_item_emb = np.asarray(llm_item_emb, dtype=np.float32)
            llm_item_emb = np.insert(llm_item_emb, 0, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
            llm_item_emb = np.insert(llm_item_emb, -1, values=np.zeros((1, llm_item_emb.shape[1])), axis=0)
            self.item_emb = nn.Embedding.from_pretrained(torch.Tensor(llm_item_emb))
        else:
            self.item_emb = nn.Embedding(self.item_num + 2, args.hidden_size, padding_idx=0)
        if args.freeze_emb:
            self.item_emb.weight.requires_grad = False
        else:
            self.item_emb.weight.requires_grad = True
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        self.loss_func = nn.BCEWithLogitsLoss()

        self.linear = nn.Sequential(
            nn.Linear(self.item_emb.embedding_dim, args.hidden_size)
        )

        if args.emb_argu:
            self.filter_init_modules = ["item_emb"]
        self._init_weights()

    def _get_embedding(self, log_seqs):
        return self.linear(self.item_emb(log_seqs))

    def log2feats(self, log_seqs):
        seqs = self._get_embedding(log_seqs)
        mask = (log_seqs != 0).unsqueeze(-1)
        masked_seqs = seqs * mask
        lengths = mask.sum(dim=1).clamp(min=1)
        pooled = masked_seqs.sum(dim=1) / lengths

        return pooled # (bs, hidden_size)

    def forward(self, seq, pos, neg, positions, **kwargs):
        log_feats = self.log2feats(seq).unsqueeze(1) # (bs, 1, hidden_size)

        pos_embs = self._get_embedding(pos.unsqueeze(1))
        neg_embs = self._get_embedding(neg)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        pos_labels = torch.ones_like(pos_logits, device=self.dev)
        neg_labels = torch.zeros_like(neg_logits, device=self.dev)

        indices = (pos != 0)
        pos_loss = self.loss_func(pos_logits[indices], pos_labels[indices])
        neg_loss = self.loss_func(neg_logits[indices], neg_labels[indices])
        loss = pos_loss + neg_loss

        return loss

    def predict(self, seq, item_indices, positions, **kwargs):
        user_emb = self.log2feats(seq)
        item_embs = self._get_embedding(item_indices)
        logits = item_embs.matmul(user_emb.unsqueeze(-1)).squeeze(-1)
        return logits # (bs, item_num)

    def get_user_emb(self, seq, positions, **kwargs):
        return self.log2feats(seq) # (bs, hidden_size)