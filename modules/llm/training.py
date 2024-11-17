import torch
import datetime
import pandas as pd
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from .tokenizer import AptamerTokenizer
from .data import AptamerDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AptGPT:
    
    def __init__(self, max_positions, embedding_size, num_layers, num_heads, batch_size_per_device, epochs, output_directory, tokenizer):
        # Set hyperparameters
        self.max_positions = max_positions  # 40
        self.embedding_size = embedding_size  # 256
        self.num_layers = num_layers  # 16
        self.num_heads = num_heads  # 16
        self.batch_size_per_device = batch_size_per_device # 32
        self.epochs = epochs # 200
        self.output_directory = output_directory

        # Set GPT configurations
        self.gpt_config = GPT2Config(
            vocab_size=tokenizer.vocab_size,  # Set vocabulary size to match the tokenizer
            n_positions=max_positions,  # Define the max length of positions
            n_embd=embedding_size,  # Embedding size (adjust as needed)
            n_layer=num_layers,  # Number of hidden layers
            n_head=num_heads,  # Number of attention heads
        )

        # Load GPT model
        self.model = GPT2LMHeadModel(self.gpt_config).to(device)
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Set to True for masked language models
        )

    def train(self, dataset: AptamerDataset):
        """
        Train the model
        :param dataset: AptamerDataset
        """
        # Set training arguments
        training_args = TrainingArguments(
            output_dir=self.output_directory + "/runs",  # output directory
            num_train_epochs=self.epochs,  # total number of training epochs
            per_device_train_batch_size=self.batch_size_per_device,  # batch size per device during training
            save_steps=10_000,  # number of updates steps before checkpoint saves
            save_total_limit=2,  # limit the total amount of checkpoints
            prediction_loss_only=True  # only compute the prediction loss
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=self.data_collator,
            train_dataset=dataset,
        )

        trainer.train()
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        trainer.save_model(
            self.output_directory + f"/apt_gpt_model_{timestamp}_epochs_{self.epochs}_batch_{self.batch_size_per_device}_layers_{self.num_layers}_heads_{self.num_heads}")
        print("Model trained and saved to disk.", self.output_directory)
        
        return self.model


# ----------------------------------------------------------------------------------------------------------------------#
# Usage
# Instantiate the AptGPT class
# tokenizer = AptamerTokenizer().load_tokenizer()
# apt_gpt = AptGPT(max_positions=30,
#                  embedding_size=256,
#                  num_layers=16,
#                  num_heads=16,
#                  batch_size_per_device=32,
#                  epochs=200,
#                  output_directory="/home/rtulluri/AptGPT-web-application/model",
#                  tokenizer=tokenizer)
# # Read data set
# dataset = pd.read_csv('/home/rtulluri/DAPTEV_Model/data/start/guan05_vae_trainer_augmented.csv', index_col=0)
# dataset = dataset['Sequences'].tolist()
# # Tokenize dataset and convert to AptamerDataset
# tokenized_dataset = tokenizer(dataset, padding='max_length', truncation=True, max_length=30, return_tensors='pt')
# train_dataset = AptamerDataset(tokenized_dataset)
# # Train the model
# apt_gpt.train(train_dataset)
