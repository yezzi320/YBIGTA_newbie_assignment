import torch  # type: ignore
from torch import nn, Tensor, LongTensor  # type: ignore
from torch.optim import Adam  # type: ignore
from torch.optim.lr_scheduler import CosineAnnealingLR  # type: ignore
import torch.nn.functional as F  # type: ignore

from transformers import PreTrainedTokenizer  # type: ignore
from typing import Literal, Union
from tqdm import tqdm  # type: ignore
import numpy as np  # type: ignore


class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        window_size: int,
        method: Literal["cbow", "skipgram"]
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.output_embeddings = nn.Embedding(vocab_size, d_model)
        self.window_size = window_size
        self.method = method
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.register_buffer("neg_cache", torch.randint(0, vocab_size, (100000,), dtype=torch.long))

    def embeddings_weight(self) -> Tensor:
        return self.embeddings.weight.detach()

    def fit(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int
    ) -> None:
        batch_size = 512
        num_negative = 10
        used_corpus_size = 10000

        optimizer = Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

        print("=" * 50)
        print("Word2Vec 학습 시작")
        print(f"방법: {self.method.upper()}")
        print(f"윈도우 크기: {self.window_size}")
        print(f"임베딩 차수: {self.d_model}")
        print(f"코퍼스 크기: {used_corpus_size} 문장")
        print("=" * 50)

        print("\n데이터 준비 중...")
        corpus_subset = corpus[:used_corpus_size]
        all_contexts, all_targets = self._prepare_data_batch(corpus_subset, tokenizer)
        print(f"총 {len(all_contexts)}개의 학습 샘플 준비 완료!")

        device = self.embeddings.weight.device

        if self.method == "cbow":
            max_context_len = max(len(ctx) for ctx in all_contexts if isinstance(ctx, list)) if all_contexts else 0
            padded_contexts = []
            for ctx in all_contexts:
                if isinstance(ctx, list):
                    padded_ctx = ctx + [0] * (max_context_len - len(ctx))
                    padded_contexts.append(padded_ctx)
                else:
                    padded_contexts.append([ctx])
            contexts_tensor = torch.tensor(padded_contexts, dtype=torch.long).to(device)
        else:
            contexts_tensor = torch.tensor(all_contexts, dtype=torch.long).to(device)

        targets_tensor = torch.tensor(all_targets, dtype=torch.long).to(device)

        dataset_size = len(contexts_tensor)
        total_batches = (dataset_size + batch_size - 1) // batch_size

        print(f"\n학습 시작 (배치 크기: {batch_size}, 총 배치: {total_batches})")
        print("=" * 50)

        for epoch in range(num_epochs):
            total_loss = 0.0
            print(f"\nEpoch {epoch+1}/{num_epochs} 시작")
            print(f"현재 학습률: {scheduler.get_last_lr()[0]:.6f}")

            perm = torch.randperm(dataset_size)
            contexts_shuffled = contexts_tensor[perm]
            targets_shuffled = targets_tensor[perm]

            for i in tqdm(range(0, dataset_size, batch_size), desc=f"Epoch {epoch+1}", leave=False):
                batch_contexts = contexts_shuffled[i:i+batch_size]
                batch_targets = targets_shuffled[i:i+batch_size]

                optimizer.zero_grad()

                if self.method == "cbow":
                    loss = self._train_cbow(batch_contexts, batch_targets)
                else:
                    loss = self._train_skipgram(batch_contexts, batch_targets, num_negative)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()

            scheduler.step()
            print(f"Epoch {epoch+1} 종료 | 평균 손실: {total_loss/total_batches:.4f}")

        print("\nWord2Vec 학습 완료!")
        print("=" * 50)

    def _prepare_data_batch(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer
    ) -> tuple[list[Union[list[int], int]], list[int]]:
        all_contexts: list[Union[list[int], int]] = []
        all_targets: list[int] = []
        seen = set()

        for text in tqdm(corpus, desc="텍스트 처리"):
            text = text.strip()
            if len(text) == 0 or text.isspace():
                continue
            tokens = tokenizer(text, add_special_tokens=False, truncation=True, max_length=256)["input_ids"]
            tokens = [tid for tid in tokens if tid != tokenizer.pad_token_id]
            if len(tokens) < 3:
                continue

            if self.method == "cbow":
                contexts, targets = self._create_cbow_data(tokens)
            else:
                contexts, targets = self._create_skipgram_data(tokens)  # type: ignore[assignment]

            for c, t in zip(contexts, targets):
                key = (tuple(c), t) if isinstance(c, list) else (c, t)
                if key not in seen:
                    seen.add(key)
                    # Skip-gram에서는 c가 int, CBOW에서는 c가 list[int]
                    all_contexts.append(c)  # type: ignore[arg-type]
                    all_targets.append(t)

        return all_contexts, all_targets

    def _create_cbow_data(self, tokens: list[int]) -> tuple[list[list[int]], list[int]]:
        contexts: list[list[int]] = []
        targets: list[int] = []
        for i in range(len(tokens)):
            target = tokens[i]
            context = [tokens[j] for j in range(i - self.window_size, i + self.window_size + 1)
                       if j != i and 0 <= j < len(tokens)]
            if len(context) > 0:
                contexts.append(context)
                targets.append(target)
        return contexts, targets

    def _create_skipgram_data(self, tokens: list[int]) -> tuple[list[int], list[int]]:
        contexts: list[int] = []
        targets: list[int] = []
        for i in range(len(tokens)):
            center_word = tokens[i]
            for j in range(i - self.window_size, i + self.window_size + 1):
                if j != i and 0 <= j < len(tokens):
                    contexts.append(center_word)
                    targets.append(tokens[j])
        return contexts, targets

    def _train_cbow(self, contexts: LongTensor, targets: LongTensor) -> Tensor:
        mask = (contexts != 0).float()
        context_embeddings = self.embeddings(contexts)
        masked_embeddings = context_embeddings * mask.unsqueeze(-1)
        context_embeddings = masked_embeddings.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)
        context_embeddings = F.normalize(context_embeddings, dim=1)
        target_embeddings = self.output_embeddings(targets)
        logits = torch.matmul(context_embeddings, target_embeddings.T)
        return F.cross_entropy(logits, targets)

    def _train_skipgram(self, contexts: LongTensor, targets: LongTensor, num_negative: int = 5) -> Tensor:
        center_embeddings = self.embeddings(contexts)
        output_weight = self.output_embeddings.weight

        pos_embeddings = F.embedding(targets, output_weight)
        pos_score = torch.sum(center_embeddings * pos_embeddings, dim=1)
        pos_loss = -F.logsigmoid(pos_score)

        neg_indices = torch.randint(
            0, self.neg_cache.size(0),
            (contexts.size(0) * num_negative,),
            device=contexts.device
        )
        neg_samples = self.neg_cache[neg_indices].view(contexts.size(0), num_negative)

        neg_embeddings = F.embedding(neg_samples, output_weight)
        center_expanded = center_embeddings.unsqueeze(1)
        neg_score = torch.sum(neg_embeddings * (-center_expanded), dim=2)
        neg_loss = -F.logsigmoid(neg_score).sum(1)

        return (pos_loss + neg_loss).mean()
