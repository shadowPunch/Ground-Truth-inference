import os
import sys
import time
import torch
from torch.optim import AdamW
from transformers import BertModel, BertConfig, BertTokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from scripts.config import Config, DEVICE
    from scripts.model import BiasNeutralizationModel, LSTMDecoder
    import scripts.util as util
except ImportError:
    print("Error: Could not import from 'scripts' directory. Make sure config.py, model.py, util.py exist there.", file=sys.stderr)
    sys.exit(1)


def main():
    config = Config()
    print(f"Using device: {DEVICE}")

    # Load tokenizer
    print(f"Loading BERT tokenizer from local file: {config.BERT_VOCAB_FILE}...")
    if not os.path.exists(config.BERT_VOCAB_FILE):
        print(f"ERROR: BERT vocab file not found at {config.BERT_VOCAB_FILE}", file=sys.stderr)
        return
    tokenizer = BertTokenizer(
        vocab_file=config.BERT_VOCAB_FILE,
        do_lower_case=True
    )
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id

    # Load training and dev data
    print(f"Loading training data from: {config.TRAIN_FILE}...")
    train_loader = util.create_dataloader(
        config.TRAIN_FILE,
        tokenizer,
        config.MAX_SEQ_LEN,
        config.BATCH_SIZE,
        shuffle=True
    )
    print(f"Loading validation data from: {config.DEV_FILE}...")
    dev_loader = util.create_dataloader(
        config.DEV_FILE,
        tokenizer,
        config.MAX_SEQ_LEN,
        config.BATCH_SIZE,
        shuffle=False
    )

    # Build Model
    print("Building model architecture...")
    bert_config = BertConfig.from_pretrained(config.BERT_MODEL_NAME)
    bert_encoder = BertModel.from_pretrained(config.BERT_MODEL_NAME, config=bert_config).to(DEVICE)

    decoder = LSTMDecoder(
        vocab_size=vocab_size,
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.LSTM_HIDDEN_DIM,
        num_layers=config.LSTM_LAYERS,
        dropout=config.DROPOUT
    ).to(DEVICE)

    model = BiasNeutralizationModel(encoder=bert_encoder, decoder=decoder, vocab_size=vocab_size).to(DEVICE)

    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    max_steps_per_epoch = len(train_loader) if config.MAX_STEPS_PER_EPOCH <= 0 else min(len(train_loader), config.MAX_STEPS_PER_EPOCH)
    total_training_steps = max_steps_per_epoch * config.EPOCHS
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_training_steps)

    loss_fn = util.build_token_weighted_loss(config.LOSS_ALPHA, pad_token_id)

    best_val_loss = float('inf')
    print(f"\n--- Starting Training for {config.EPOCHS} epochs ---")
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        epoch_loss = 0
        for step, batch in enumerate(train_loader, 1):
            if config.MAX_STEPS_PER_EPOCH > 0 and step > config.MAX_STEPS_PER_EPOCH:
                print(f"Reached max steps {config.MAX_STEPS_PER_EPOCH} for epoch {epoch}")
                break

            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            decoder_input_ids = batch['decoder_input_ids'].to(DEVICE)
            target_labels = batch['target_labels'].to(DEVICE)

            logits = model(input_ids, attention_mask, decoder_input_ids, target_labels)
            loss = loss_fn(logits, target_labels, batch['src_text'], batch['tgt_text'], tokenizer)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if step % 100 == 0 or step == max_steps_per_epoch:
                print(f"Epoch {epoch}/{config.EPOCHS}, Step {step}/{max_steps_per_epoch}, Loss: {loss.item():.4f}")

        avg_train_loss = epoch_loss / max_steps_per_epoch

        # Validation
        model.eval()
        with torch.no_grad():
            val_scores = util.calculate_metrics(model, dev_loader, loss_fn, tokenizer, None, None)

        print(f"\n--- Epoch {epoch} Summary ---")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss:   {val_scores['loss']:.4f} | Val Acc: {val_scores['accuracy']*100:.2f}% | Val BLEU: {val_scores['bleu']*100:.2f}")
        print("-" * 30)

        # Save best model
        if val_scores['loss'] < best_val_loss:
            best_val_loss = val_scores['loss']
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"Validation loss improved, model saved to: {config.MODEL_SAVE_PATH}")

    print("\n--- Training Complete ---")


if __name__ == "__main__":
    main()
