import os
import sys
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
import numpy as np


try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)

STOP_WORDS = set(stopwords.words('english'))


def load_sbert_model():
    
    print("Loading Sentence-BERT model (all-MiniLM-L6-v2)...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        print(f"Error loading SBERT model. Do you have internet enabled in Kaggle? \n{e}", file=sys.stderr)
        return None
    print("SBERT model loaded.")
    return model

def load_lexicons(lexicon_dir="lexicons"):
    """Loads all subjectivity lexicons from the specified directory."""
    print(f"Loading subjectivity lexicons from ./{lexicon_dir}/...")
    LEXICON_FILES = [
        "assertives_hooper1975.txt", "entailed_arg_berant2012.txt",
        "entailed_berant2012.txt", "entailing_arg_berant2012.txt",
        "entailing_berant2012.txt", "factives_hooper1975.txt",
        "hedges_hyland2005.txt", "implicatives_karttunen1971.txt",
        "negative_liu2005.txt", "npov_lexicon.txt",
        "positive_liu2005.txt", "report_verbs.txt",
        "strong_subjectives_riloff2003.txt", "weak_subjectives_riloff2003.txt"
    ]
    
    lexicons = {}
    if not os.path.exists(lexicon_dir):
        print(f"Warning: Lexicon directory not found at {lexicon_dir}.", file=sys.stderr)
        print("Please run `setup_lexicons.py` first.", file=sys.stderr)
        return None

    for fname in LEXICON_FILES:
        key = fname.split('.')[0]
        filepath = os.path.join(lexicon_dir, fname)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                # Use a set for fast lookups
                lexicons[key] = set(line.strip().lower() for line in f if line.strip())
        else:
            print(f"Warning: Lexicon file not found: {filepath}", file=sys.stderr)
            
    print(f"Loaded {len(lexicons)} lexicons.")
    return lexicons

# --- METRIC 1: Exact Match Accuracy ---
def get_exact_match(prediction, reference):
    """Calculates 1.0 if strings match exactly (after strip), 0.0 otherwise."""
    return 1.0 if prediction.strip().lower() == reference.strip().lower() else 0.0

# --- METRIC 2: BLEU Score ---
def get_bleu_score(prediction, reference):
    """Calculates sentence-level BLEU score."""
    pred_tokens = prediction.strip().lower().split()
    ref_tokens = [reference.strip().lower().split()] # Must be list of lists
    
    # Use smoothing to avoid 0.0 scores for short sentences
    smoothie = SmoothingFunction().method1
    return sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie)

# --- METRIC 3: Semantic Similarity Score ---

def _calculate_jaccard(s1_tokens, s2_tokens):
    """Calculates Jaccard similarity """
    set1 = {w for w in s1_tokens if w not in STOP_WORDS and w.isalnum()}
    set2 = {w for w in s2_tokens if w not in STOP_WORDS and w.isalnum()}
    
    if not set1 and not set2:
        return 1.0  # Both are empty
    if not set1 or not set2:
        return 0.0  # One is empty
        
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def _calculate_sbert_sim(s1, s2, sbert_model):
    """Calculates cosine similarity using SBERT."""
    if sbert_model is None:
        return 0.0 # Return 0 if model failed to load
    embeddings = sbert_model.encode([s1, s2], convert_to_tensor=True)
    cos_sim = util.cos_sim(embeddings[0], embeddings[1])
    return cos_sim.item()

def get_semantic_similarity(original_text, neutralized_text, sbert_model):
    """
    Calculates the composite semantic similarity score
    """
    s1_tokens = original_text.lower().split()
    s2_tokens = neutralized_text.lower().split()
    
    sbert_score = _calculate_sbert_sim(original_text, neutralized_text, sbert_model)
    jaccard_score = _calculate_jaccard(s1_tokens, s2_tokens)
    
    composite_score = (0.8 * sbert_score) + (0.2 * jaccard_score)
    return composite_score

# --- METRIC 4: Aggregate Bias Score ---

def _calculate_lexicon_score(text_tokens, lexicons):
    """Finds the ratio of biased words in the text based on lexicons."""
    if not lexicons or not text_tokens:
        return 0.0
        
    biased_word_count = 0
    all_lexicon_words = set().union(*lexicons.values())
    
    for token in text_tokens:
        if token in all_lexicon_words:
            biased_word_count += 1
            
    return biased_word_count / len(text_tokens)

def get_aggregate_bias_score(text, detector_prob, ideology_prob, lexicons):
    
    #Calculates the composite bias score from 0.0 (neutral) to 1.0 (biased).
    #Score = (Detector_Prob + Lexicon_Score + Ideology_Prob) / 3

    text_tokens = text.lower().split()
    
    # 1. Lexicon Score
    lexicon_score = _calculate_lexicon_score(text_tokens, lexicons)
    
    # 2. Detector Probability
    # Since this function is called from a model that might not *be* the
    # detector, we'll use a placeholder.
    # In a real pipeline, you'd run the Detector model first and pass its score here.
    if detector_prob is None:
        # Use lexicon_score as a fallback if no detector prob is provided
        detector_prob = lexicon_score 
        
    # 3. Ideology Probability

    if ideology_prob is None:
        ideology_prob = 0.0
    
    # Calculate the average
    score = (detector_prob + lexicon_score + (1.0 - ideology_prob)) / 3.0
    return min(1.0, score) 


def evaluate_batch(batch, model, tokenizer, loss_fn, sbert_model, lexicons):
    """
    Runs all evaluation metrics for a single batch from the dataloader.
    
    Returns a dictionary of *average* scores for the batch.
    """
    model.eval()
    
    input_ids = batch['input_ids'].to(DEVICE)
    attention_mask = batch['attention_mask'].to(DEVICE)
    decoder_input_ids = batch['decoder_input_ids'].to(DEVICE)
    target_labels = batch['target_labels'].to(DEVICE)
    
    batch_size = input_ids.size(0)
    
    with torch.no_grad():
        # --- 1. Get Model Outputs ---
        # Get logits for loss calculation
        logits = model(input_ids, attention_mask, decoder_input_ids, target_labels)
        
        # Get generated sentences for BLEU, Accuracy, etc.
        encoder_outputs = model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state
        
        hypotheses = model.generate(encoder_outputs, tokenizer, Config.MAX_SEQ_LEN)
    
    # --- 2. Get References ---
    references = batch['tgt_text']
    sources = batch['src_text']

    # --- 3. Calculate Metrics ---
    total_loss = loss_fn(
        logits, target_labels, 
        batch['src_text'], batch['tgt_text'], 
        tokenizer
    ).item()
    
    total_bleu = 0
    total_accuracy = 0
    total_semantic_sim = 0
    total_bias_score_pred = 0
    total_bias_score_ref = 0

    for i in range(batch_size):
        pred_sent = hypotheses[i]
        ref_sent = references[i]
        src_sent = sources[i]
        
        total_accuracy += get_exact_match(pred_sent, ref_sent)
        total_bleu += get_bleu_score(pred_sent, ref_sent)
        total_semantic_sim += get_semantic_similarity(src_sent, pred_sent, sbert_model)
        
        # Bias Score: Compare bias of prediction vs. reference
        total_bias_score_pred += get_aggregate_bias_score(pred_sent, None, None, lexicons)
        total_bias_score_ref += get_aggregate_bias_score(ref_sent, None, None, lexicons)

    # Return average scores for the batch
    return {
        "loss": total_loss, # This is batch loss, not avg
        "accuracy": total_accuracy / batch_size,
        "bleu": total_bleu / batch_size,
        "semantic_sim": total_semantic_sim / batch_size,
        "bias_score_pred": total_bias_score_pred / batch_size,
        "bias_score_ref": total_bias_score_ref / batch_size,
    }

