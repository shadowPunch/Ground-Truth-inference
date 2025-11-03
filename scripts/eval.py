import os
import sys
import time
import torch
from transformers import BertTokenizer

# Ensure scripts directory is in path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from scripts.config import Config, DEVICE
    from scripts.model import BiasNeutralizationModel, LSTMDecoder, BahdanauAttention
    import scripts.util as util
except ImportError:
    print("Error: Could not import scripts modules. Make sure scripts/config.py, model.py and util.py exist.", file=sys.stderr)
    sys.exit(1)


def main():
    config = Config()
    print(f"Using device: {DEVICE}")

    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"ERROR: Model file not found at {config.MODEL_SAVE_PATH}. Train model first.", file=sys.stderr)
        return

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

    print("Loading evaluation dependencies (SBERT, Lexicons)...")
    sbert_model = util.load_sbert_model(config.SBERT_MODEL)
    lexicons = util.load_lexicons(config.LEXICON_DIR)
    if sbert_model is None or lexicons is None:
        print("Could not load evaluation dependencies.", file=sys.stderr)
        return

    print(f"Loading test data from {config.TEST_FILE}...")
    test_loader = util.create_dataloader(
        config.TEST_FILE,
        tokenizer,
        config.MAX_SEQ_LEN,
        config.BATCH_SIZE,
        shuffle=False
    )

    print("Building model architecture...")
    decoder = LSTMDecoder(
        vocab_size=vocab_size,
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.LSTM_HIDDEN_DIM,
        num_layers=config.LSTM_LAYERS,
        dropout=config.DROPOUT
    ).to(DEVICE)

    try:
        encoder = torch.hub.load('huggingface/pytorch-transformers', 'model', config.BERT_MODEL_NAME)
    except Exception:
        # fallback to BertModel.from_pretrained in actual usage, or import BertModel directly
        from transformers import BertModel
        encoder = BertModel.from_pretrained(config.BERT_MODEL_NAME)

    encoder.to(DEVICE)

    model = BiasNeutralizationModel(
        encoder=encoder,
        decoder=decoder,
        vocab_size=vocab_size
    ).to(DEVICE)

    print(f"Loading trained model weights from {config.MODEL_SAVE_PATH}...")
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded.")

    loss_fn = util.build_token_weighted_loss(config.LOSS_ALPHA, pad_token_id)

    print("\n--- Running Final Evaluation on Test Set ---")
    start_time = time.time()

    scores = util.calculate_metrics(model, test_loader, loss_fn, tokenizer, sbert_model, lexicons)

    end_time = time.time()

    print(f"Evaluation complete in {end_time - start_time:.2f} seconds.")
    print("\n--- Test Set Results ---")
    print(f"Test Loss:        {scores['loss']:.4f}")
    print(f"Test Accuracy:    {scores['accuracy']*100:.2f}%")
    print(f"Test BLEU:        {scores['bleu']*100:.2f}")
    print(f"Semantic Similarity: {scores['semantic_sim']:.4f}")
    print(f"Bias Score (Source): {scores['bias_score_src']:.4f}")
    print(f"Bias Score (Target): {scores['bias_score_ref']:.4f}")
    print(f"Bias Score (Pred):   {scores['bias_score_pred']:.4f}")

    print("\n--- Sample Neutralizations ---")
    with torch.no_grad():
        try:
            batch = next(iter(test_loader))
        except StopIteration:
            print("No test data available for examples.")
            return
        input_ids = batch['input_ids'][:5].to(DEVICE)
        attention_mask = batch['attention_mask'][:5].to(DEVICE)
        encoder_outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hypotheses = model.generate(encoder_outputs, tokenizer, config.MAX_SEQ_LEN)
        references = batch['tgt_text'][:5]
        sources = batch['src_text'][:5]
        for i in range(len(hypotheses)):
            print(f"\nSource:     {sources[i]}")
            print(f"Target:     {references[i]}")
            print(f"Prediction: {hypotheses[i]}")


if __name__ == "__main__":
    main()
