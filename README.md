# Ideological Bias Neutralization in media texts

This project adapts the **CONCURRENT** model from the paper  **"Automatically Neutralizing Subjective Bias in Text" (Pryzant et al., 2020)**.  
My goal is to create an efficient system for identifying and neutralizing ideological bias in news text that can be trained and deployed in **resource-constrained environments** (e.g., free Kaggle GPUs) instead of relying on high-end, inaccessible hardware.

### Key Modifications

#### 1. **Hybrid Dataset Augmentation**
Supplement the original **WNC dataset** with **synthetic biased/neutral pairs** generated via the Gemini API.

#### 2. **Robust Evaluation Metrics**
We move beyond BLEU and Accuracy to introduce:
- **Semantic Similarity Score**
- **Aggregate Bias Score**

These better measure how well the model preserves meaning while removing bias.

---

## Live Demo

A **Streamlit** web app (`app.py`) is included for live demonstrations. Also, the Google Drive link for the project demo presentation is also attached.


This is the google drive link to the video presentation on this project  :  [Video presentation](https://drive.google.com/drive/folders/1pi5l832wiVKQa8GRIOSAmEoMdDNGdayc)

---

## File Structure

```
.
├── app.py                
├── data/
│   ├── biased.full
│   ├── biased.word.dev
│   ├── biased.word.test
│   ├── biased.word.train
│   └── neutral
│
└── scripts/
    ├── config.py
    ├── datafile.py
    ├── eval.py
    ├── model.py
    ├── requirements.txt
    ├── setup_lexicons.py
    ├── train.py
    └── util.py
```

---

## Setup & Installation

### Clone the Repository
```
git clone https://github.com/shadowPunch/Ground-Truth-inference
cd Ground-Truth-inference
```

### Install Dependencies
```
pip install -r scripts/requirements.txt
```


### Get the Data
Download the WNC dataset from the [original paper’s repository](https://github.com/rpryzant/neutralizing-bias)  
and place the files in the `data/` directory as shown above.

### Setup Lexicon Files
```
python scripts/setup_lexicons.py
```

---

## How to Run

### 1. Generate Synthetic Data
Augment the dataset using the **Gemini API**.

```
export GEMINI_API_KEY="your-api-key-here"
python scripts/datafile.py
```

### 2. Train the Model
```
python scripts/train.py
```
The script:
- Loads data from `data/`
- Builds the CONCURRENT model (BERT Encoder + LSTM Decoder)
- Trains for configured epochs
- Saves the best checkpoint: `./concurrent_model.pt`

### 3. Evaluate the Model
```
python scripts/eval.py
```
Runs evaluation metrics:
- BLEU  
- Accuracy  
- Semantic Similarity  
- Aggregate Bias Score

### 4. Run the Streamlit Demo
```
streamlit run app.py
```

---

## Model Architecture

Implemented the **CONCURRENT model (Section 3.2)** from Pryzant et al. (2020):

- **Encoder:** Pre-trained `bert-base-uncased`, producing hidden states from biased input.  
- **Decoder:** Attentional LSTM that uses the encoder’s [CLS] token for initialization and attends to hidden states for token-by-token neutralization.  
- **Loss Function:** Token-Weighted Loss with \( \alpha = 1.3 \), emphasizing accurate edits of biased words.

---

## Evaluation Metrics

### Semantic Similarity (Sim)
A weighted combination ensuring factual preservation:
$$
\text{Sim}(S_{\text{orig}}, S_{\text{neut}}) = (0.8 \times \text{Sim}_{\text{SBERT}}) + (0.2 \times \text{Sim}_{\text{Jaccard}})
$$

### Aggregate Bias Score (Bias)
Quantifies residual bias from 0 (neutral) to 1 (biased):
$$
\text{Bias}(S) = \frac{P_{\text{detector}} + S_{\text{lexicon}} + (1 - P_{\text{neutral}})}{3}
$$


---

## Performance Comparison

| **Metric** | **Pryzant et al. (Modular)** | **Our Adapted Model** |
|-------------|------------------------------|------------------------|
| **Hardware** | NVIDIA TITAN X | Kaggle T4 × 2 |
| **Training Time** | ~10 hours | ~6 hours |
| **Generation Accuracy** | 45.8% | **37.2%** |
| **Semantic Similarity** | (Not Measured) | **0.84** |
| **Bias Score (Post-Neutral)** | (Human-Judged) | **0.31 (Low)** |

Our model’s 37.2% accuracy and 0.84 semantic similarity show effective bias reduction while maintaining factual integrity.

---

## Acknowledgments

This project adapts and extends the foundational work of:
**Pryzant, R., Martinez, R. D., Dass, N., Kurohashi, S., Jurafsky, D., & Yang, D. (2020).**  
*Automatically Neutralizing Subjective Bias in Text.*  
In *Proceedings of the AAAI Conference on Artificial Intelligence.*  
[https://arxiv.org/abs/2004.09986](https://arxiv.org/abs/2004.09986)

---

## License

This project is licensed under the **MIT License**.
```
