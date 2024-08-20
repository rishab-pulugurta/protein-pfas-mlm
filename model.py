import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer

class PolymerProteinModel(nn.Module):
    def __init__(self, model_name, unfreeze_layers=3, use_lora=False):
        super(PolymerProteinModel, self).__init__()
        self.base_model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Add separator token
        special_tokens_dict = {'sep_token': '[SEP]'}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.base_model.resize_token_embeddings(len(self.tokenizer))

        if use_lora:
            self.apply_lora()
        else:
            self.unfreeze_layers(unfreeze_layers)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def generate_protein_smile(self, polymer_smile, max_length=512):
        self.eval()
        device = next(self.parameters()).device

        # Tokenize polymer SMILES
        polymer_encoding = self.tokenizer(polymer_smile, return_tensors="pt", max_length=max_length//2, 
                                          truncation=True, padding="max_length")
        input_ids = polymer_encoding['input_ids'].to(device)
        attention_mask = polymer_encoding['attention_mask'].to(device)

        # Add [SEP] token
        sep_token_id = self.tokenizer.sep_token_id
        input_ids = torch.cat([input_ids, torch.tensor([[sep_token_id]]).to(device)], dim=1)
        attention_mask = torch.cat([attention_mask, torch.tensor([[1]]).to(device)], dim=1)

        # Add mask tokens for protein SMILES
        protein_length = max_length - input_ids.shape[1]
        mask_tokens = torch.full((1, protein_length), self.tokenizer.mask_token_id, device=device)
        input_ids = torch.cat([input_ids, mask_tokens], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones((1, protein_length), device=device)], dim=1)

        # Generate protein SMILES
        with torch.no_grad():
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs.logits

        # Unmask tokens
        predicted_tokens = torch.argmax(predictions[0, input_ids.shape[1]-protein_length:], dim=-1)
        generated_smile = self.tokenizer.decode(predicted_tokens, skip_special_tokens=True)

        return generated_smile