import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import config
from data_loader import (
    tokenizer,
    NeutralDataset,
    NeutralizationDataset,
    create_data_loader,
    PAD_TOKEN_ID
)
from model import Encoder, Decoder, ConcurrentModel
from utils import TokenWeightedLoss, save_checkpoint, load_checkpoint

def pretrain_epoch(model, dataloader, optimizer, criterion, device):
    """Runs one epoch of pre-training (denoising autoencoder)"""
    model.train()
    epoch_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Pre-training", leave=False)
    
    for batch in progress_bar:
        source_ids = batch['source_ids'].to(device)
        source_mask = batch['source_mask'].to(device)
        target_ids = batch['target_ids'].to(device)
        
        optimizer.zero_grad()
        
        # Use 100% teacher forcing for pre-training
        outputs = model(source_ids, source_mask, target_ids, teacher_forcing_ratio=1.0)
        
        # outputs: [batch, target_len, vocab_size]
        # target_ids: [batch, target_len]
        
        # Flatten for loss calculation
        # Skip <CLS> token (index 0) in loss
        output_dim = outputs.shape[-1]
        outputs_flat = outputs[:, 1:].reshape(-1, output_dim)
        targets_flat = target_ids[:, 1:].reshape(-1)
        
        loss = criterion(outputs_flat, targets_flat)
        loss.backward()
        
        clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
    return epoch_loss / len(dataloader)

def finetune_epoch(model, dataloader, optimizer, criterion, device):
    """Runs one epoch of fine-tuning (neutralization)"""
    model.train()
    epoch_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Fine-tuning", leave=False)
    
    for batch in progress_bar:
        source_ids = batch['source_ids'].to(device)
        source_mask = batch['source_mask'].to(device)
        target_ids = batch['target_ids'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            source_ids, 
            source_mask, 
            target_ids, 
            teacher_forcing_ratio=config.TEACHER_FORCING_RATIO
        )
        
        # Custom weighted loss calculation
        # Skip <CLS> token (index 0) in loss
        loss = criterion(
            predictions=outputs[:, 1:, :],
            targets=target_ids[:, 1:],
            source_ids=source_ids
        )
        
        loss.backward()
        
        clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
    return epoch_loss / len(dataloader)

def main():
    device = config.DEVICE
    print(f"Using device: {device}")

    # --- Initialize Model ---
    encoder = Encoder().to(device)
    decoder = Decoder(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=config.HIDDEN_DIM,
        lstm_layers=config.LSTM_LAYERS,
        dropout=config.DROPOUT
    ).to(device)
    model = ConcurrentModel(encoder, decoder, tokenizer, device).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # --- 1. Pre-training Phase ---
    print("--- Starting Pre-training Phase ---")
    pretrain_dataset = NeutralDataset(
        config.WNC_NEUTRAL_PATH, tokenizer, config.MAX_LEN
    )
    pretrain_loader = create_data_loader(
        pretrain_dataset, config.BATCH_SIZE, shuffle=True
    )
    
    # Standard Cross-Entropy Loss for pre-training
    pretrain_criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
    
    for epoch in range(config.PRETRAIN_EPOCHS):
        train_loss = pretrain_epoch(
            model, pretrain_loader, optimizer, pretrain_criterion, device
        )
        print(f"Pre-train Epoch {epoch+1}/{config.PRETRAIN_EPOCHS} | Loss: {train_loss:.4f}")
        
    # Save the pre-trained model
    save_checkpoint(model, optimizer, config.PRETRAIN_SAVE_PATH)
    print("Pre-training complete. Model saved.")

    # --- 2. Fine-tuning Phase ---
    print("\n--- Starting Fine-tuning Phase ---")
    
    # Reload the best pre-trained model (or just continue)
    # model, optimizer = load_checkpoint(model, optimizer, config.PRETRAIN_SAVE_PATH)

    finetune_dataset = NeutralizationDataset(
        config.WNC_BIASED_WORD_PATH, tokenizer, config.MAX_LEN
    )
    finetune_loader = create_data_loader(
        finetune_dataset, config.BATCH_SIZE, shuffle=True
    )
    
    # Custom Token-Weighted Loss for fine-tuning
    finetune_criterion = TokenWeightedLoss(
        alpha=config.TOKEN_WEIGHT_ALPHA,
        pad_token_id=PAD_TOKEN_ID,
        ignore_index=PAD_TOKEN_ID
    )
    
    best_loss = float('inf')
    for epoch in range(config.FINETUNE_EPOCHS):
        train_loss = finetune_epoch(
            model, finetune_loader, optimizer, finetune_criterion, device
        )
        print(f"Fine-tune Epoch {epoch+1}/{config.FINETUNE_EPOCHS} | Loss: {train_loss:.4f}")
        
        if train_loss < best_loss:
            best_loss = train_loss
            save_checkpoint(model, optimizer, config.FINETUNE_SAVE_PATH)
            
    print(f"Fine-tuning complete. Best model saved to {config.FINETUNE_SAVE_PATH}")

if __name__ == "__main__":
    main()

