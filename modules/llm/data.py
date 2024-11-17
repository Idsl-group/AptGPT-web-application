import random
import torch
from torch.utils.data import Dataset

class DataUtils:
    def __init__(self, data):
        self.data = data

    def augment_dataframe(self, sequence_column=0, percentage=0.35):
        num_to_augment = int(len(self.data) * percentage)

        # Randomly select indices to augment
        indices_to_augment = random.sample(range(len(self.data)), num_to_augment)

        # Copy the original dataframe to avoid modifying the original one
        df_augmented = self.data.copy()

        # Apply augmentation to the selected rows
        df_augmented.iloc[indices_to_augment, sequence_column] = df_augmented.iloc[
            indices_to_augment, sequence_column].apply(self.augment_sequence)

        return df_augmented

    @staticmethod
    def augment_sequence(seq):
        length = len(seq)
        # Randomly decide how many nucleotides to remove (1 to 5)
        num_to_remove = random.randint(1, 5)

        # Remove from back if less than 3 nucleotides to remove
        if num_to_remove < 3:
            # Remove num_to_remove nucleotides randomly from anywhere in the sequence
            new_seq = seq[:-num_to_remove]

        # Remove from front and back if more than or equal to 3 nucleotides to remove
        else:
            # Choosing whether to remove from front or back
            if random.choice([True, False]):
                new_seq = seq[1:-num_to_remove + 1]
            else:
                new_seq = seq[num_to_remove - 1:length - 1]

        return new_seq


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