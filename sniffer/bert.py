from transformers import BertTokenizer, BertModel
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

"""
Modified Code from @marcellusruben
- https://github.com/marcellusruben/medium-resources/blob/main/Text_Classification_BERT/bert_medium.ipynb
"""

class Dataset(torch.utils.data.Dataset):
    """
    Custom Torch class for input text data
    """
    def __init__(self, df, tokenizer):
        self.labels = [label for label in df['label']]
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y
