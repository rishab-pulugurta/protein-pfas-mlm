# dataset.py

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class PolymerProteinDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=512):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        polymer_smile = self.data.iloc[idx]['Polymer Smile']
        protein_smile = self.data.iloc[idx]['Protein Smile']

        # Concatenate with [SEP] token
        combined_smile = f"{polymer_smile} [SEP] {protein_smile}"
        encoding = self.tokenizer(combined_smile, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # Mask the protein SMILES portion
        sep_token_index = (input_ids == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
        input_ids[sep_token_index+1:] = self.tokenizer.mask_token_id

        labels = encoding['input_ids'].squeeze().clone()
        labels[:sep_token_index+1] = -100  # Don't compute loss for polymer and separator

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
