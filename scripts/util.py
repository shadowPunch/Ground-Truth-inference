import torch
import torch.nn as nn
from nltk.translate.bleu_score import corpus_bleu
from data_loader import PAD_TOKEN_ID
import config

class TokenWeightedLoss(nn.Module):
    """
    Implements the token-weighted loss from the paper (Section 3.1).
    It applies a higher weight (alpha) to tokens in the target
    that are *not* present in the source.
    """
    def __init__(self, alpha, pad_token_id, ignore_index):
        super(TokenWeightedLoss, self).__init__()
        self.alpha = alpha
        self.pad_token_id = pad_token_id
        # Use reduction='none' to get per-token losses
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, predictions, targets, source_ids):
        # predictions: [batch_size, target_len, vocab_size]
        # targets: [batch_size, target_len]
        # source_ids: [batch_size, source_len]
        
        batch_size, target_len, vocab_size = predictions.shape
        
        # Flatten predictions and targets
        # predictions_flat: [batch_size * target_len, vocab_size]
        # targets_flat: [batch_size * target_len]
        predictions_flat = predictions.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        
        # Calculate standard cross-entropy loss per token
        # loss: [batch_size * target_len]
        loss = self.criterion(predictions_flat, targets_flat)
        
        # --- Create weights matrix ---
        
        # Reshape loss and targets to 2D
        # loss_2d: [batch_size, target_len]
        # targets_2d: [batch_size, target_len]
        loss_2d = loss.view(batch_size, target_len)
        targets_2d = targets_flat.view(batch_size, target_len)

        # Initialize weights to 1.0
        weights = torch.ones_like(loss_2d)
        
        # Create a set of source tokens for each batch item for fast lookup
        source_sets = [set(row.tolist()) for row in source_ids]
        
        # Apply alpha weight
        for i in range(batch_size): # Iterate over batch
            for j in range(target_len): # Iterate over sequence
                token = targets_2d[i, j].item()
                # Apply alpha if token is NOT in source and is NOT a pad token
                if token not in source_sets[i] and token != self.pad_token_id:
                    weights[i, j] = self.alpha
                    
        # Flatten weights to match loss
        weights_flat = weights.view(-1)
        
        # Calculate final weighted loss
        weighted_loss = loss * weights_flat
        
        # Return the mean loss (ignoring padding, which CrossEntropyLoss already did)
        return weighted_loss.mean()


def save_checkpoint(model, optimizer, filename):
    """Saves model and optimizer state."""
    print(f"=> Saving checkpoint to {filename}")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename):
    """Loads model and optimizer state."""
    print(f"=> Loading checkpoint from {filename}")
    checkpoint = torch.load(filename, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer

def calculate_metrics(predictions, targets, tokenizer):
    """
    Calculates BLEU and Accuracy.
    - predictions: List of predicted sentences (strings)
    - targets: List of target sentences (strings)
    """
    
    # 1. Accuracy
    correct = 0
    for pred, target in zip(predictions, targets):
        if pred.strip() == target.strip():
            correct += 1
    accuracy = correct / len(targets)
    
    # 2. BLEU Score
    # For corpus_bleu, targets need to be list of lists of tokens
    # e.g., [['this', 'is', 'a', 'test'], ['another', 'one']]
    # predictions are list of tokens
    # e.g., ['this', 'is', 'test']
    
    target_tokens = [[tokenizer.tokenize(t)] for t in targets]
    pred_tokens = [tokenizer.tokenize(p) for p in predictions]
    
    bleu_score = corpus_bleu(target_tokens, pred_tokens)
    
    return accuracy, bleu_score

