import torch

# --- Model Hyperparameters ---
# Select BERT model
BERT_MODEL_NAME = 'bert-base-uncased'
# Hidden dimension of BERT and LSTM
# Must match BERT's hidden dim (768 for base, 1024 for large)
HIDDEN_DIM = 768
# Number of layers for the LSTM decoder
LSTM_LAYERS = 2
# Dropout probability
DROPOUT = 0.2

# --- Training Hyperparameters ---
# Set device (GPU if available, else CPU)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Number of epochs for pre-training (denoising autoencoder)
PRETRAIN_EPOCHS = 4
# Number of epochs for fine-tuning (neutralization task)
FINETUNE_EPOCHS = 7
# Batch size (reduce if you get CUDA out-of-memory errors)
BATCH_SIZE = 16
# Learning rate
LEARNING_RATE = 5e-5
# Gradient clipping (as mentioned in paper)
GRAD_CLIP = 3.0
# Alpha for token-weighted loss (from paper)
TOKEN_WEIGHT_ALPHA = 1.3
# Teacher forcing ratio (0.5 means 50% of the time)
TEACHER_FORCING_RATIO = 0.5

# --- Data & File Paths ---
# NOTE: Update these paths to point to your downloaded data
# Path to the neutral text file (one sentence per line)
WNC_NEUTRAL_PATH = './data/wnc_neutral_sentences.txt'
# Path to the biased-word parallel corpus (e.g., a CSV/TSV)
# Assumes a CSV with 'biased_text' and 'neutral_text' columns
WNC_BIASED_WORD_PATH = './data/wnc_biased_word_corpus.csv'

# Path to save the pre-trained model
PRETRAIN_SAVE_PATH = './models/pretrained_decoder.pt'
# Path to save the final fine-tuned model
FINETUNE_SAVE_PATH = './models/neutralizer_model.pt'

# --- Tokenizer & Sequence Settings ---
# Max sequence length for tokenizer
MAX_LEN = 128
# Probability of dropping a word in the denoising autoencoder
WORD_DROP_PROB = 0.25
# Max shuffle distance for denoising autoencoder
WORD_SHUFFLE_K = 3

