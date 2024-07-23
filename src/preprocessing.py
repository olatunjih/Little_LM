from transformers import AutoTokenizer
from nlpaug.augmenter.word import SynonymAug
import torch

class DataPreprocessor:
    def __init__(self, tokenizer_name, max_length=128):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def tokenize(self, texts):
        return self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

    def augment_data(self, texts):
        aug = SynonymAug(aug_src='wordnet')
        return [aug.augment(text) for text in texts]
