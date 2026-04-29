import torch

class TrainingConfig:
    # Hyperparameters from the paper
    LEARNING_RATE = 2e-5
    BATCH_SIZE = 16
    EPOCHS = 4
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    MAX_SEQ_LEN = 128
    DROPOUT = 0.1
    GRADIENT_CLIPPING = 1.0
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Paths
    OUTPUT_DIR = 'results'
    MODEL_SAVE_DIR = 'saved_models'
    
    @classmethod
    def get_config(cls):
        return {k: v for k, v in cls.__dict__.items() if not k.startswith('__')}
