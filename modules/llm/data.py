import torch
from torch.utils.data import Dataset

class AptamerDataset(Dataset):
    """
    Dataset class for aptamer data and pre-training GPT model
    """
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


class FineTuneDataset(Dataset):
    """
    Dataset class for fine-tuning GPT model
    """
    def __init__(self, sequences, targets, tokenizer, max_length=6):
        self.sequences = sequences
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_seq = self.sequences[idx]
        target_seq = self.targets[idx]

        input_tokens = self.tokenizer.encode(input_seq, return_tensors='pt', max_length=self.max_length, truncation=True)
        target_tokens = self.tokenizer.encode(target_seq, return_tensors='pt', max_length=self.max_length, truncation=True)

        return input_tokens.squeeze(0), target_tokens.squeeze(0)