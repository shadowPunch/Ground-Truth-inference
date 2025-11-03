import os
import sys
import time
import csv
import pandas as pd
import google.generativeai as genai
import torch
import torch.nn.functional as F

try:
    from config import Config
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from config import Config
    except ImportError:
        print("Fatal Error: Cannot find 'config.py'. Make sure it is in the 'scripts' directory.", file=sys.stderr)
        sys.exit(1)

try:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set.", file=sys.stderr)
        print("Please set it in your Kaggle Secrets or environment.", file=sys.stderr)
        sys.exit(1)
    genai.configure(api_key=api_key)
except Exception as e:
    print(f"Error configuring Gemini API: {e}", file=sys.stderr)
    sys.exit(1)

GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": 32,
    "max_output_tokens": 1024,
}

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

llm = genai.GenerativeModel(
    model_name="gemini-2.5-flash-preview-09-2025",
    generation_config=GENERATION_CONFIG,
    safety_settings=SAFETY_SETTINGS
)

NEUTRALIZE_PROMPT_TEMPLATE = """
**Task:** Act as a journalist adhering to a strict Neutral Point of View (NPOV).
Rewrite the following biased sentence to be neutral, balanced, and objective.
Your goal is to *neutralize* the subjective language while *preserving* the core facts.
Do not add any commentary. Only output the neutralized sentence.

**Biased Sentence:**
"{sentence}"

**Neutralized Sentence:**
"""

BIAS_PROMPT_TEMPLATE = """
**Task:** Act as a political commentator.
Rewrite the following neutral sentence to include a clear subjective bias (e.g., Left or Right leaning).
Use framing, loaded language, and subjective verbs to introduce opinion.
Do not add any commentary. Only output the biased sentence.

**Neutral Sentence:**
"{sentence}"

**Biased Sentence:**
"""

def generate_with_backoff(prompt_text):
    delay = 1.0
    for _ in range(5):
        try:
            response = llm.generate_content(prompt_text)
            return response.text.strip().replace('\n', ' ')
        except Exception as e:
            if "resource has been exhausted" in str(e).lower() or "rate limit" in str(e).lower():
                print(f"Rate limit hit, sleeping for {delay:.1f}s...")
                time.sleep(delay)
                delay *= 2.0
            else:
                print(f"An unexpected error occurred: {e}")
                return None
    print("Failed to generate content after multiple retries.")
    return None

def augment_data(input_files, output_file, num_to_generate):
    column_names = ['id', 'src_tok', 'tgt_tok', 'src_raw', 'tgt_raw', 'src_POS_tags', 'tgt_parse_tags']
    all_dfs = []
    for file in input_files:
        try:
            df = pd.read_csv(
                file,
                sep='\t',
                quoting=csv.QUOTE_NONE,
                on_bad_lines='skip',
                header=None,
                names=column_names
            ).dropna(subset=['src_raw', 'tgt_raw'])
            all_dfs.append(df)
        except FileNotFoundError:
            print(f"--- ERROR: File not found at {file} ---", file=sys.stderr)
            return
    combined_df = pd.concat(all_dfs, ignore_index=True)
    if len(combined_df) < num_to_generate:
        print(f"Warning: Requested {num_to_generate} samples, but only {len(combined_df)} available. Using all.")
        num_to_generate = len(combined_df)
    df_sample = combined_df.sample(n=num_to_generate, random_state=42)
    new_data = []
    for i, (idx, row) in enumerate(df_sample.iterrows()):
        biased_sentence = row['src_raw']
        prompt = NEUTRALIZE_PROMPT_TEMPLATE.format(sentence=biased_sentence)
        synthetic_neutral = generate_with_backoff(prompt)
        if synthetic_neutral:
            new_data.append({
                'id': f"syn_neut_{row['id']}",
                'src_raw': biased_sentence,
                'tgt_raw': synthetic_neutral,
                'src_tok': 'synthetic', 'tgt_tok': 'synthetic',
                'src_POS_tags': '', 'tgt_parse_tags': ''
            })
        neutral_sentence = row['tgt_raw']
        prompt = BIAS_PROMPT_TEMPLATE.format(sentence=neutral_sentence)
        synthetic_biased = generate_with_backoff(prompt)
        if synthetic_biased:
            new_data.append({
                'id': f"syn_bias_{row['id']}",
                'src_raw': synthetic_biased,
                'tgt_raw': neutral_sentence,
                'src_tok': 'synthetic', 'tgt_tok': 'synthetic',
                'src_POS_tags': '', 'tgt_parse_tags': ''
            })
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{num_to_generate} samples...")
    new_df = pd.DataFrame(new_data)
    print(f"Generated {len(new_df)} new sentences. Saving to {output_file}...")
    new_df.to_csv(
        output_file,
        sep='\t',
        index=False,
        header=False,
        columns=column_names,
        quoting=csv.QUOTE_NONE
    )
    print("Data augmentation complete.")

def load_data(files, tokenizer, max_len):
    column_names = ['id', 'src_tok', 'tgt_tok', 'src_raw', 'tgt_raw', 'src_POS_tags', 'tgt_parse_tags']
    all_dfs = []
    for tsv_file in files:
        try:
            df = pd.read_csv(
                tsv_file,
                sep='\t',
                quoting=csv.QUOTE_NONE,
                on_bad_lines='skip',
                header=None,
                names=column_names
            )
            all_dfs.append(df)
        except Exception as e:
            print(f"Error reading {tsv_file}: {e}", file=sys.stderr)
            sys.exit(1)
    data = pd.concat(all_dfs, ignore_index=True)
    data = data.dropna(subset=['src_raw', 'tgt_raw'])
    data['src_raw'] = data['src_raw'].astype(str)
    data['tgt_raw'] = data['tgt_raw'].astype(str)
    if len(data) == 0:
        print(f"Warning: No data loaded from {files}. Check file content.", file=sys.stderr)
    else:
        print(f"Successfully loaded {len(data)} total examples.")
    # Return full dataframe for Dataset to use tokenization as in training code
    return data

def setup_lexicons():
    try:
        from config import Config
        LEXICON_DIR = Config.LEXICON_DIR
    except ImportError:
        print("config.py not found. Using default './lexicons' directory.")
        LEXICON_DIR = "./lexicons"
    LEXICON_FILES = [
        "assertives_hooper1975.txt",
        "entailed_arg_berant2012.txt",
        "entailed_berant2012.txt",
        "entailing_arg_berant2012.txt",
        "factives_hooper1975.txt",
        "hedges_hyland2005.txt",
        "implicatives_karttunen1971.txt",
        "negative_liu2005.txt",
        "npov_lexicon.txt",
        "positive_liu2005.txt",
        "report_verbs.txt",
        "strong_subjectives_riloff2003.txt",
        "weak_subjectives_riloff2003.txt"
    ]
    if not os.path.exists(LEXICON_DIR):
        try:
            os.makedirs(LEXICON_DIR)
            print(f"Created directory: {LEXICON_DIR}/")
        except OSError as e:
            print(f"Error creating directory {LEXICON_DIR}: {e}", file=sys.stderr)
            return
    else:
        print(f"Directory {LEXICON_DIR}/ already exists.")
    count = 0
    for filename in set(LEXICON_FILES):
        filepath = os.path.join(LEXICON_DIR, filename)
        if not os.path.exists(filepath):
            try:
                with open(filepath, 'w'):
                    pass
                count += 1
            except IOError as e:
                print(f"Error creating file {filepath}: {e}", file=sys.stderr)
    if count > 0:
        print(f"Created {count} new empty lexicon files in {LEXICON_DIR}/.")
    else:
        print("All lexicon files already exist.")
    print("Lexicon setup complete. You can now (optionally) add words to these .txt files.")

if __name__ == "__main__":
    INPUT_FILES = Config.TRAIN_FILES
    OUTPUT_FILE = os.path.join(os.path.dirname(INPUT_FILES[0]), "synthetic_data.tsv") if INPUT_FILES else "./synthetic_data.tsv"
    NUM_SAMPLES = 50 
    augment_data(INPUT_FILES, OUTPUT_FILE, NUM_SAMPLES)
    setup_lexicons()
