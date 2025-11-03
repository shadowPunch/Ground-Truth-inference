import torch

class Config:
    # --- File Paths ---
    BASE_DATA_PATH = "./data/"
    TRAIN_FILE = BASE_DATA_PATH + "biased.word.train.tsv"
    DEV_FILE = BASE_DATA_PATH + "biased.word.dev.tsv"
    TEST_FILE = BASE_DATA_PATH + "biased.word.test.tsv"
    NEUTRAL_FILE = BASE_DATA_PATH + "neutral.txt"  # For pre-training (if implemented)
    
    BERT_VOCAB_FILE = BASE_DATA_PATH + "bert.vocab"
    LEXICON_DIR = "./lexicons"
    MODEL_SAVE_PATH = "./concurrent_model.pt"
    
    # --- Model Hyperparameters ---
    BERT_MODEL_NAME = 'bert-base-uncased'
    LSTM_HIDDEN_DIM = 768
    LSTM_LAYERS = 2
    EMBED_DIM = 768
    DROPOUT = 0.2
    
    # --- Training Hyperparameters ---
    EPOCHS = 10
    BATCH_SIZE = 16
    LEARNING_RATE = 5e-5
    MAX_SEQ_LEN = 128
    MAX_STEPS_PER_EPOCH = -1  # -1 means no limit (full epoch)
    LOSS_ALPHA = 1.3
    
    # --- Evaluation ---
    RUN_EVALUATION = True
    SBERT_MODEL = 'all-MiniLM-L6-v2'  # For Semantic Similarity

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
