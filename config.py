# config.py

# Model parameters
MODEL_NAME = "DeepChem/ChemBERTa-77M-MLM"
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 5e-5
UNFREEZE_LAYERS = 3  # Number of layers to unfreeze (last N layers)
USE_LORA = False  # Whether to use LoRA or not

# Training hyperparameters
OPTIMIZER = 'AdamW'  # Optimizer to use
WEIGHT_DECAY = 0.01  # Weight decay for regularization
WARMUP_STEPS = 500  # Warmup steps for learning rate
EVAL_STEPS = 100  # Evaluation steps
SAVE_STEPS = 1000  # Checkpoint saving steps

# File paths
TRAIN_CSV = '/workspace/rtp/CF-PepMLM/data/datasets/train.csv'
VAL_CSV = '/workspace/rtp/CF-PepMLM/data/datasets/val.csv'
TEST_CSV = '/workspace/rtp/CF-PepMLM/data/datasets/test.csv'

# WandB parameters
WANDB_PROJECT = 'CF3-PepMLM'
WANDB_ENTITY = 'RTP'

# Checkpoint directory
CHECKPOINT_DIR = '/workspace/rtp/CF-PepMLM/checkpoints'
