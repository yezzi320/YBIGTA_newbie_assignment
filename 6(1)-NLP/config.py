from typing import Literal


device = "cpu"
d_model = 256

# Word2Vec
window_size = 5
method: Literal["cbow", "skipgram"] = "skipgram"
lr_word2vec = 1e-03
num_epochs_word2vec = 5

# GRU
hidden_size = 256
num_classes = 4
lr = 2e-03  # 더 안정적인 학습률
num_epochs = 200  # 더 많은 에포크
batch_size = 64  # 더 큰 배치 크기