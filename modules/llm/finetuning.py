import RNA
import datetime
import math
import torch
import pandas as pd
from skbio import DNA
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from .data import FineTuneDataset
from .tokenizer import AptamerTokenizer
from transformers import GPT2LMHeadModel, AdamW
from torch.utils.data import DataLoader
from skbio.alignment import local_pairwise_align_ssw
device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SFT_AptGPT:
    """
    Class to fine-tune GPT2 for aptamer generation.
    """
    def __init__(self, tokenizer, model_path, epochs=5, learning_rate=1e-5, batch_size=16, max_length=30, save_path='./', reference_sequence='', score_weight=0.99999):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        self.score_weight = score_weight
        self.reference_sequence = reference_sequence
        self.tokenizer = tokenizer
        self.save_path = save_path
        self.model_path = model_path

        # Load GPT2 model
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(device)

    def train(self, targets):
        # Set up optimizer and input-target data
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        inputs = ["<bos>"] * 1000
        references = [self.reference_sequence] * 1000

        # Build dataloader
        dataset = FineTuneDataset(inputs, references, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Train model
        self.model.train()
        for epoch in tqdm(range(self.epochs)):
            losses = []
            for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
                loss = None
                generation_loss = []
                # Pass through model and generate sequences
                input_ids, outputs = self.forward(input_ids.to(device), target_ids.to(device))
                # Get decoded sequences for loss calculation
                generated_sequences = [self.tokenizer.decode(input_id[1:], skip_special_tokens=True).replace(' ', '') for input_id in input_ids]

                try:
                    loss = self.calculate_reward(generated_sequences, targets).mean()
                    generation_loss.append(loss.item())
                except Exception as e:
                    print(e)
                    continue

                optimizer.zero_grad()
                outputs.loss = ((1 -self.score_weight) * outputs.loss) + (self.score_weight * loss)
                outputs.loss.backward()
                optimizer.step()

                losses.append(np.mean(generation_loss))

            print(f"Epoch: {epoch}, Loss: {np.mean(losses)}\n")

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.model.save_pretrained(self.save_path + f"/apt_gpt_model_finetuned_{timestamp}_epochs_{self.epochs}_lr_{self.learning_rate}_bs_{self.batch_size}")
        print("Model trained and saved to disk.", self.save_path)

        return self.model

    def forward(self, input_ids, target_ids):
        output = None

        for _ in range(self.max_length):
            partial_target = target_ids[:, :_+1]
            output = self.model(input_ids, labels=partial_target)
            logits = output.logits

            next_token_logits = logits[:, -1, :]
            next_token = torch.multinomial(nn.functional.softmax(next_token_logits, dim=-1), num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=-1)

        return input_ids, output

    def generate(self, prompt, num_sequences, temperature=0.7, top_k=30, top_p=0.95):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(device)
        output_ids = self.model.generate(
            input_ids,
            max_length=self.max_length,
            temperature=temperature,
            num_return_sequences=num_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
        )
        output_sequences = [self.tokenizer.decode(output_id, skip_special_tokens=True) for output_id in output_ids]

        return output_sequences

    def load_model(self, model_path=None):
        if model_path:
            self.model_path = model_path

        self.model = GPT2LMHeadModel.from_pretrained(self.model_path).to(device)
        return self.model

    def calculate_reward(self, generated_sequences, targets):
        rewards = []
        for seq in generated_sequences:
            alignment_score = max(self.compute_alignment_score(seq, targets))
            _, _, hairpins, _, energy = self.count_rna_structures_viennarna(seq)

            # Activation functions for the rewards
            alignment_reward = torch.nn.MSELoss()(torch.tensor([alignment_score]), torch.tensor([1.0]))
            mfe_reward = 1 / (1 + math.exp(2 * (energy - (-5))))
            hairpin_reward = max(1 - (0.05 * hairpins), 0)

            # Merge all rewards
            reward = 0.1 * alignment_reward + 0.45 * mfe_reward + 0.45 * hairpin_reward
            rewards.append(reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        return rewards

    @staticmethod
    def count_rna_structures_viennarna(sequence):
        # Fold the RNA sequence to get the secondary structure
        fold_compound = RNA.fold_compound(sequence)
        structure, mfe = fold_compound.mfe()

        # Use RNAeval to evaluate the secondary structure for hairpins, loops, and stems
        hairpin_count = 0
        loop_count = 0
        forward_stem_count = 0
        backward_stem_count = 0

        # Get secondary structure details
        for i, char in enumerate(structure):
            if char == '.':
                continue
            elif char == '(':
                forward_stem_count += 1
            elif char == ')':
                backward_stem_count += 1
            elif char == '.':  # Loops or unpaired bases
                loop_count += 1

        # Hairpins are represented as `(...)` or `(...)`
        hairpin_count = structure.count('(')  # Simple heuristic for now

        return forward_stem_count, backward_stem_count, hairpin_count, loop_count, mfe

    @staticmethod
    def compute_alignment_score(seq, targets):
        total = len(targets)
        clusters = targets.cluster_id.unique()
        scores = []

        for cluster in clusters:
            group = targets[targets.cluster_id == cluster]
            factor = len(group) / total
            seq_target = group.sequence.iloc[0]

            _, score, _ = local_pairwise_align_ssw(DNA(seq), DNA(seq_target))
            max_possible_score = len(seq) + len(seq_target)
            normalized_score = score / max_possible_score
            scores.append(factor * normalized_score)

        return scores


# ---------------------------------------------------------------------------------------------------------------------#
# Usage
# tokenizer = AptamerTokenizer().load_tokenizer()
# apt_gpt = SFT_AptGPT(
#     tokenizer,
#     epochs=5,
#     learning_rate=1e-5,
#     batch_size=16,
#     max_length=30,
#     model_path='/home/rtulluri/AptGPT-web-application/model/apt_gpt_model_20241115_175841_epochs_200_batch_32_layers_16_heads_16',
#     save_path='/home/rtulluri/AptGPT-web-application/model',
#     reference_sequence="GGGTCTGTAATGGATTGTTCTCAACCAACT"
# )
#
# targets = pd.read_csv("/home/rtulluri/DAPTEV_Model/data/start/guan06_vae_trainer.csv")['Sequences'].values
# apt_gpt.train(targets)


# ---------------------------------------------------------------------------------------------------------------------#
# Usage of pre-finetuned model
# tokenizer = AptamerTokenizer().load_tokenizer()
# apt_gpt = SFT_AptGPT(tokenizer, model_path='/home/rtulluri/DAPTEV_Model/llm_trained_models/finetuned-gpt2-dna-model_20240919_214426_epochs_20_Qloss_guan11')
# print(apt_gpt.generate("<bos>", 5))