import torch
import pandas as pd
import random
import csv
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import config

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
PAD_TOKEN_ID = tokenizer.pad_token_id

def corrupt_text(text_tokens):
    """
    Applies the denoising autoencoder corruption from the paper:
    1. Slightly shuffles words (within k=3)
    2. Drops words (with p=0.25)
    """
    n = len(text_tokens)
    k = config.WORD_SHUFFLE_K
    p = config.WORD_DROP_PROB
    
    # 1. Shuffle
    shuffled_indices = list(range(n))
    for i in range(n):
        # Calculate valid permutation range [max(0, i-k), min(n-1, i+k)]
        swap_range_start = max(0, i - k)
        swap_range_end = min(n - 1, i + k)
        
        # Select a random index within the range
        j = random.randint(swap_range_start, swap_range_end)
        
        # Swap indices
        shuffled_indices[i], shuffled_indices[j] = shuffled_indices[j], shuffled_indices[i]
        
    shuffled_tokens = [text_tokens[i] for i in shuffled_indices]

    # 2. Drop
    corrupted_tokens = [token for token in shuffled_tokens if random.random() > p]
    
    # Handle empty case
    if not corrupted_tokens:
        corrupted_tokens = [tokenizer.cls_token, tokenizer.sep_token]
        
    return " ".join(corrupted_tokens)

class NeutralDataset(Dataset):
    """Dataset for pre-training (Denoising Autoencoder)"""
    def __init__(self, filepath, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Load data (assumes one neutral sentence per line)
        with open(filepath, 'r', encoding='utf-8') as f:
            self.sentences = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        clean_text = self.sentences[idx]
        
        # Corrupt the text for the encoder input
        # We tokenize with basic split to manipulate words, not sub-words
        corrupted_text = corrupt_text(clean_text.split())
        
        # Tokenize clean (target) and corrupted (source) text
        source = self.tokenizer(
            corrupted_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target = self.tokenizer(
            clean_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'source_ids': source['input_ids'].squeeze(0),
            'source_mask': source['attention_mask'].squeeze(0),
            'target_ids': target['input_ids'].squeeze(0),
        }

class NeutralizationDataset(Dataset):
    """Dataset for fine-tuning (Biased -> Neutral)"""
    def __init__(self, filepath, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Load data (assumes TSV with 'src_raw' and 'tgt_raw' columns)
        # Use csv.QUOTE_NONE to handle potential quote issues in the raw text
        self.data = pd.read_csv(
            filepath, 
            sep='\t', 
            quoting=csv.QUOTE_NONE, 
            on_bad_lines='skip'
        )
        # Ensure columns are string type and handle NaNs
        self.data['src_raw'] = self.data['src_raw'].astype(str).fillna('')
        self.data['tgt_raw'] = self.data['tgt_raw'].astype(str).fillna('')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        biased_text = row['src_raw']
        neutral_text = row['tgt_raw']
        
        source = self.tokenizer(
            biased_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target = self.tokenizer(
            neutral_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'source_ids': source['input_ids'].squeeze(0),
            'source_mask': source['attention_mask'].squeeze(0),
            'target_ids': target['input_ids'].squeeze(0),
        }

def create_data_loader(dataset, batch_size, shuffle=True):
    """Utility function to create a DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2  # You can adjust this
    )

if __name__ == '__main__':
    # --- Test Data Loaders ---
    print("Testing NeutralDataset (Pre-training)...")
    try:
        pretrain_dataset = NeutralDataset(config.WNC_NEUTRAL_PATH, tokenizer, config.MAX_LEN)
        pretrain_loader = create_data_loader(pretrain_dataset, config.BATCH_SIZE)
        batch = next(iter(pretrain_loader))
        print("Batch shape (source_ids):", batch['source_ids'].shape)
        print("Batch shape (target_ids):", batch['target_ids'].shape)
        print("Example source:", tokenizer.decode(batch['source_ids'][0], skip_special_tokens=True))
        print("Example target:", tokenizer.decode(batch['target_ids'][0], skip_special_tokens=True))
    except Exception as e:
        print(f"Error loading pre-training data: {e}")
        print("Please check config.WNC_NEUTRAL_PATH")

    print("\nTesting NeutralizationDataset (Fine-tuning)...")
    try:
        finetune_dataset = NeutralizationDataset(config.WNC_BIASED_WORD_PATH, tokenizer, config.MAX_LEN)
        finetune_loader = create_data_loader(finetune_dataset, config.BATCH_SIZE)
        batch = next(iter(finetune_loader))
        print("Batch shape (source_ids):", batch['source_ids'].shape)
        print("Batch shape (target_ids):", batch['target_ids'].shape)
        print("Example source (src_raw):", tokenizer.decode(batch['source_ids'][0], skip_special_tokens=True))
        print("Example target (tgt_raw):", tokenizer.decode(batch['target_ids'][0], skip_special_tokens=True))
    except Exception as e:
        print(f"Error loading fine-tuning data: {e}")


