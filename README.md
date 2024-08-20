# CF-PepMLM: C-F Polymer-Peptide SMILES MLM Model

This repository contains code to train a model using a pre-trained SMILES transformer, ChemBERTa, to generate peptides that bind to C-F polymers. We train the  model by concatenating polymer SMILES strings with peptide SMILES strings and using a masking objective to fully mask the peptide sequence for de novo peptide generation.

## Model Architecture

The model architecture is based on the ChemBERTa model with options for using LoRA and unfreezing specific layers for fine-tuning.

### Loss Function

The loss function used is the standard cross-entropy loss for masked language modeling. The loss is computed only on the peptide sequence, excluding the polymer and the separator token.

Mathematically, the loss $$ \mathcal{L} $$ is defined as:

$$
\mathcal{L} = -\sum_{t=1}^{T} y_t \log(\hat{y}_t)
$$

## Configuration

The configuration file `config.py` contains all the necessary hyperparameters for training and fine-tuning.

## Training

To train the model, run:

```bash
python train.py
``````

## Training

To generate new peptides for an input polymer, run:

```bash
python generate.py --polymer_smile "YOUR_POLYMER_SMILES" --num_amino_acids N
``````

## Dependencies

- PyTorch
- Transformers
- Pandas
- WandB
