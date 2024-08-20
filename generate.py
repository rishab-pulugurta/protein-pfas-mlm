# generate.py
import torch
import argparse
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from model import PolymerProteinModel
from config import *
import os
import json

def load_tokenizer(checkpoints_dir):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(checkpoints_dir, "tokenizer.json"))
    
    if os.path.exists(os.path.join(checkpoints_dir, "special_tokens_map.json")):
        with open(os.path.join(checkpoints_dir, "special_tokens_map.json"), "r") as f:
            special_tokens_map = json.load(f)
        tokenizer.add_special_tokens(special_tokens_map)

    if os.path.exists(os.path.join(checkpoints_dir, "added_tokens.json")):
        with open(os.path.join(checkpoints_dir, "added_tokens.json"), "r") as f:
            added_tokens = json.load(f)
        tokenizer.add_tokens(list(added_tokens.keys()))
    
    return tokenizer

def generate_protein_smile(polymer_smile, model, tokenizer, num_amino_acids, max_length=512):
    """
    Generates a protein SMILES string given a polymer SMILES string.
    Args:
    polymer_smile (str): Polymer SMILES string.
    model (PolymerProteinModel): Fine-tuned model.
    tokenizer (AutoTokenizer): Tokenizer for encoding SMILES strings.
    num_amino_acids (int): Number of amino acids desired in the protein.
    max_length (int): Maximum length of the tokenized sequences.
    Returns:
    protein_smile (str): Generated protein SMILES string.
    """
    model.eval()
    device = next(model.parameters()).device

    # Tokenize polymer SMILES
    polymer_encoding = tokenizer(polymer_smile, return_tensors='pt', max_length=max_length//2,
                                 truncation=True, padding='max_length')
    input_ids = polymer_encoding['input_ids'].to(device)
    attention_mask = polymer_encoding['attention_mask'].to(device)

    # Add [SEP] token
    sep_token_id = tokenizer.sep_token_id
    input_ids = torch.cat([input_ids, torch.tensor([[sep_token_id]]).to(device)], dim=1)
    attention_mask = torch.cat([attention_mask, torch.tensor([[1]]).to(device)], dim=1)

    # Add mask tokens for protein SMILES
    protein_length = min(num_amino_acids, max_length - input_ids.shape[1])
    mask_tokens = torch.full((1, protein_length), tokenizer.mask_token_id, device=device)
    input_ids = torch.cat([input_ids, mask_tokens], dim=1)
    attention_mask = torch.cat([attention_mask, torch.ones((1, protein_length), device=device)], dim=1)

    # Generate protein SMILES
    with torch.no_grad():
        outputs = model.base_model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = outputs.logits

    # Unmask tokens
    predicted_tokens = torch.argmax(predictions[0, input_ids.shape[1]-protein_length:], dim=-1)
    generated_sequence = tokenizer.decode(predicted_tokens, skip_special_tokens=True)

    return generated_sequence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a protein SMILES string given a polymer SMILES string.")
    parser.add_argument('--polymer_smile', type=str, required=True, help="Polymer SMILES string.")
    parser.add_argument('--num_amino_acids', type=int, required=True, help="Number of amino acids desired in the protein.")
    parser.add_argument('--checkpoints_dir', type=str, default="/workspace/rtp/CF-PepMLM/checkpoints", help="Path to the main checkpoints directory.")
    parser.add_argument('--model_checkpoint', type=str, default="checkpoint-2170", help="Name of the specific model checkpoint directory.")
    args = parser.parse_args()

    # Load the tokenizer from the main checkpoints directory
    tokenizer = load_tokenizer(args.checkpoints_dir)

    # Initialize the model
    model = PolymerProteinModel(MODEL_NAME, UNFREEZE_LAYERS, USE_LORA)
    
    # Load the model weights from the specific checkpoint directory
    model_path = os.path.join(args.checkpoints_dir, args.model_checkpoint, "pytorch_model.bin")
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    generated_protein = generate_protein_smile(args.polymer_smile, model, tokenizer, args.num_amino_acids)
    print(f"Input Polymer SMILE: {args.polymer_smile}")
    print(f"Generated Protein SMILE: {generated_protein}")