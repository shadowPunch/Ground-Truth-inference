import torch
import torch.nn.functional as F
import pandas as pd
import csv
import os
import sys
import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util as sbert_util
from collections import defaultdict


class NeutralizationDataset(torch.utils.data.Dataset):
    def __init__(self, tsv_file, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        column_names = [
            'id', 'src_tok', 'tgt_tok', 'src_raw', 'tgt_raw', 
            'src_POS_tags', 'tgt_parse_tags'
        ]
        try:
            self.data = pd.read_csv(
                tsv_file,
                sep='\t',
                quoting=csv.QUOTE_NONE,
                on_bad_lines='skip',
                header=None,
                names=column_names
            )
        except FileNotFoundError:
            print(f"--- FATAL ERROR: File not found at {tsv_file} ---", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"--- FATAL ERROR: Error reading {tsv_file}: {e} ---", file=sys.stderr)
            sys.exit(1)
        self.data = self.data.dropna(subset=['src_raw', 'tgt_raw'])
        self.data['src_raw'] = self.data['src_raw'].astype(str)
        self.data['tgt_raw'] = self.data['tgt_raw'].astype(str)
        if len(self.data) == 0:
            print(f"Error: No data loaded from {tsv_file}. Check file content.", file=sys.stderr)
            sys.exit(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        src_text = row['src_raw']
        tgt_text = row['tgt_raw']

        src_encoding = self.tokenizer(
            src_text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        tgt_encoding = self.tokenizer(
            tgt_text,
            truncation=True,
            max_length=self.max_len - 1,
            padding='max_length',
            return_tensors='pt'
        )
        decoder_input_ids = torch.cat([
            torch.tensor([self.tokenizer.cls_token_id]),
            tgt_encoding['input_ids'][0, 1:]
        ])
        target_labels = torch.cat([
            tgt_encoding['input_ids'][0, 1:],
            torch.tensor([self.tokenizer.sep_token_id])
        ])
        decoder_input_ids = F.pad(decoder_input_ids, (0, self.max_len - len(decoder_input_ids)), value=self.tokenizer.pad_token_id)
        target_labels = F.pad(target_labels, (0, self.max_len - len(target_labels)), value=self.tokenizer.pad_token_id)

        return {
            'input_ids': src_encoding['input_ids'].squeeze(0),
            'attention_mask': src_encoding['attention_mask'].squeeze(0),
            'decoder_input_ids': decoder_input_ids,
            'target_labels': target_labels,
            'src_text': src_text,
            'tgt_text': tgt_text
        }


def create_dataloader(file_path, tokenizer, max_len, batch_size, shuffle=True):
    dataset = NeutralizationDataset(tsv_file=file_path, tokenizer=tokenizer, max_len=max_len)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)


def build_token_weighted_loss(alpha, pad_token_id):
    def compute_loss(logits, targets, source_text_batch, target_text_batch, tokenizer):
        logits_flat = logits.view(-1, logits.shape[-1])
        targets_flat = targets.view(-1)
        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=pad_token_id, reduction='none')
        weights = torch.ones_like(targets_flat, dtype=torch.float, device=logits.device)
        batch_size, seq_len = targets.shape
        for i in range(batch_size):
            src_words = set(source_text_batch[i].lower().split())
            tgt_token_ids = targets[i]
            for j in range(seq_len):
                token_id = tgt_token_ids[j].item()
                if token_id == pad_token_id:
                    continue 
                token = tokenizer.decode([token_id]).strip()
                if token and token not in src_words and token not in (tokenizer.cls_token, tokenizer.sep_token):
                    weights[i * seq_len + j] = alpha
        weighted_loss = (loss * weights).mean()
        return weighted_loss
    return compute_loss


@torch.no_grad()
def load_sbert_model(model_name='all-MiniLM-L6-v2'):
    print(f"Loading Sentence-BERT model: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
        print("SBERT model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading SBERT model. Make sure internet is on. Error: {e}", file=sys.stderr)
        return None


def load_lexicons(lexicon_dir):
    print(f"Loading bias lexicons from {lexicon_dir}...")
    if not os.path.exists(lexicon_dir):
        print(f"--- ERROR: Lexicon directory not found at: {lexicon_dir} ---", file=sys.stderr)
        print("Please run 'python scripts/setup_lexicons.py' first.", file=sys.stderr)
        return None
        
    lexicons = {}
    lexicon_files = [
        "assertives_hooper1975.txt", "entailed_arg_berant2012.txt",
        "entailed_berant2012.txt", "entailing_arg_berant2012.txt",
        "entailing_berant2012.txt",
        "factives_hooper1975.txt", "hedges_hyland2005.txt",
        "implicatives_karttunen1971.txt", "negative_liu2005.txt", 
        "npov_lexicon.txt", "positive_liu2005.txt", "report_verbs.txt",
        "strong_subjectives_riloff2003.txt", "weak_subjectives_riloff2003.txt"
    ]
    
    for fname in set(lexicon_files):
        key = fname.split('.')[0]
        try:
            with open(os.path.join(lexicon_dir, fname), 'r', encoding='utf-8') as f:
                lexicons[key] = set(line.strip().lower() for line in f if line.strip())
        except FileNotFoundError:
            lexicons[key] = set()
            
    print(f"Loaded {len(lexicons)} lexicons.")
    return lexicons


def get_exact_match(pred_sent, ref_sent):
    return 1.0 if pred_sent.strip().lower() == ref_sent.strip().lower() else 0.0


def get_bleu_score(pred_sent, ref_sent):
    reference = [nltk.word_tokenize(ref_sent.lower())]
    hypothesis = nltk.word_tokenize(pred_sent.lower())
    chencherry = SmoothingFunction().method1 
    return sentence_bleu(reference, hypothesis, smoothing_function=chencherry)


@torch.no_grad()
def get_semantic_similarity(sent1, sent2, sbert_model):
    try:
        embeddings = sbert_model.encode([sent1, sent2], convert_to_tensor=True)
        sbert_sim = sbert_util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
        sbert_sim = (sbert_sim + 1) / 2 
        tokens1 = set(nltk.word_tokenize(sent1.lower()))
        tokens2 = set(nltk.word_tokenize(sent2.lower()))
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        jaccard_sim = len(intersection) / len(union) if len(union) > 0 else 0.0
        weighted_sim = (0.8 * sbert_sim) + (0.2 * jaccard_sim)
        return weighted_sim
    except Exception:
        return 0.0


def get_aggregate_bias_score(sentence, detector_prob, ideology_prob, lexicons):
    if not sentence.strip():
        return 0.0
    tokens = nltk.word_tokenize(sentence.lower())
    if not tokens:
        return 0.0
    p_detector = detector_prob if detector_prob is not None else 0.5
    token_set = set(tokens)
    bias_word_count = 0
    for lex_name, word_set in lexicons.items():
        if "npov" in lex_name or "subjective" in lex_name or "negative" in lex_name or "positive" in lex_name:
            bias_word_count += len(token_set.intersection(word_set))
    s_lexicon = min(1.0, bias_word_count / len(tokens)) 
    p_neutral = (1.0 - ideology_prob) if ideology_prob is not None else 0.5
    return s_lexicon


@torch.no_grad()
def calculate_metrics(model, dataloader, loss_fn, tokenizer, sbert_model, lexicons):
    model.eval()
    total_scores = defaultdict(float)
    total_samples = 0
    all_refs = []
    all_hyps = []
    
    from scripts.config import Config
    max_len = Config.MAX_SEQ_LEN

    for batch in dataloader:
        batch_size = batch['input_ids'].size(0)
        input_ids = batch['input_ids'].to(torch.device(tokenizer.device if hasattr(tokenizer, 'device') else 'cpu'))
        attention_mask = batch['attention_mask'].to(torch.device(tokenizer.device if hasattr(tokenizer, 'device') else 'cpu'))
        decoder_input_ids = batch['decoder_input_ids'].to(torch.device(tokenizer.device if hasattr(tokenizer, 'device') else 'cpu'))
        target_labels = batch['target_labels'].to(torch.device(tokenizer.device if hasattr(tokenizer, 'device') else 'cpu'))
        encoder_outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hypotheses = model.generate(encoder_outputs, tokenizer, max_len=max_len)
        logits = model(input_ids, attention_mask, decoder_input_ids, target_labels)
        loss = loss_fn(logits, target_labels, batch['src_text'], batch['tgt_text'], tokenizer)
        total_scores['loss'] += loss.item()
        references = batch['tgt_text']
        sources = batch['src_text']
        batch_refs_for_bleu = []
        batch_hyps_for_bleu = []
        for i in range(batch_size):
            pred_sent = hypotheses[i]
            ref_sent = references[i]
            src_sent = sources[i]
            total_scores['accuracy'] += get_exact_match(pred_sent, ref_sent)
            total_scores['semantic_sim'] += get_semantic_similarity(src_sent, pred_sent, sbert_model)
            total_scores['bias_score_pred'] += get_aggregate_bias_score(pred_sent, None, None, lexicons)
            total_scores['bias_score_ref'] += get_aggregate_bias_score(ref_sent, None, None, lexicons)
            total_scores['bias_score_src'] += get_aggregate_bias_score(src_sent, None, None, lexicons)
            batch_refs_for_bleu.append([nltk.word_tokenize(ref_sent.lower())])
            batch_hyps_for_bleu.append(nltk.word_tokenize(pred_sent.lower()))
        all_refs.extend(batch_refs_for_bleu)
        all_hyps.extend(batch_hyps_for_bleu)
        total_samples += batch_size
    avg_scores = {key: val / total_samples for key, val in total_scores.items()}
    avg_scores['loss'] = total_scores['loss'] / len(dataloader)
    chencherry = SmoothingFunction().method1
    avg_scores['bleu'] = corpus_bleu(all_refs, all_hyps, smoothing_function=chencherry)
    return avg_scores
