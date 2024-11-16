import itertools
from tokenizers import Tokenizer, models, decoders, pre_tokenizers
from transformers import PreTrainedTokenizerFast


class AptamerTokenizer:
    """
    Tokenizer for DNA sequences
    """

    def __init__(self):
        self.vocab = None
        self.tokenizer = None

    def save_new_tokenizer(self, base_tokens, permutations=1, save_path="./tokens/dna_tokenizer.json"):
        """
        Save a new tokenizer with the given base tokens and permutations
        """
        # Vocab building
        self.build_vocab(base_tokens, permutations)

        # Encode lambda function for torch Tokenizer class
        token_to_id = {token: idx for idx, token in enumerate(self.vocab)}

        self.tokenizer = Tokenizer(models.WordLevel(vocab=token_to_id,  unk_token="<unk>"))
        self.tokenizer.add_tokens(self.vocab)
        self.tokenizer._pre_tokenizer = pre_tokenizers.Whitespace()
        self.tokenizer._decoder = decoders.WordPiece()

        self.tokenizer.save(save_path)

    def build_vocab(self, base_tokens, permutations=1):
        """
        Build a vocabulary from the base tokens and permutations
        """
        special_tokens = ["<pad>", "<bos>", "<eos>"]
        self.vocab = base_tokens + special_tokens

        if permutations > 1:
            for i in range(1, permutations + 1):
                self.vocab += ["".join(p) for p in itertools.product(base_tokens, repeat=i)]

        return self.vocab

    @staticmethod
    def load_tokenizer(path="./tokens/dna_tokenizer.json"):
        """
        Load a tokenizer from a file
        """
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=path, vocab_file=None)
        tokenizer.pad_token = "<pad>"
        tokenizer.bos_token = "<bos>"
        tokenizer.eos_token = "<eos>"
        # tokenizer.add_tokens(self.vocab)
        return tokenizer


# ---------------------------------------------------------------------------------------------------------------------- #
# Usage
# tokenizer = AptamerTokenizer()
# tokenizer.save_new_tokenizer(["A", "C", "G", "T"], 1)
# tokenizer = tokenizer.load_tokenizer()
# print(f"Vocabulary size: {len(tokenizer)}")
# print(f"Tokenizer vocab: {tokenizer.get_vocab()}")
