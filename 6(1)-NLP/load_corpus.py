# 구현하세요!
from datasets import load_dataset

def load_corpus() -> list[str]:
    corpus: list[str] = []
    # 구현하세요 !
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    for item in dataset["train"]:
        text = item["text"].strip()
        if text:
            corpus.append(text)
            
    return corpus
