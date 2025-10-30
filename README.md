# Ideological Bias Neutralization in media texts

This project adapts the **CONCURRENT** model from the paper  **"Automatically Neutralizing Subjective Bias in Text" (Pryzant et al., 2020)**.  
My goal is to create an efficient system for identifying and neutralizing ideological bias in news text that can be trained and deployed in **resource-constrained environments** (e.g., free Kaggle GPUs) instead of relying on high-end, inaccessible hardware.

---

## Key Modifications

### 1. **Hybrid Dataset Augmentation**
We supplement the original **WNC dataset** with **synthetic biased/neutral pairs** generated via the Gemini API.

### 2. ****

### 3. **Robust Evaluation Metrics**
We move beyond BLEU and Accuracy to introduce:
- **Semantic Similarity Score**
- **Aggregate Bias Score**

These better measure how well the model preserves meaning while removing bias.

---

## Live Demo

A **Streamlit** web app (`app.py`) is included for live demonstrations. Also, the Google Drive link for the project demo presentation is also attached.

- **Input Options:** Direct text input or file uploads (`.txt`, `.pdf`, `.docx`)
- **Output:** Complete analysis with bias classification, neutralized output, and evaluation metrics.

---

## Project Contributions

This project is an **adaptation** focused on accessibility and improved evaluation.

### Accessibility over Compute
| Feature | Pryzant et al. | Our Adaptation |
|----------|----------------|----------------|
| **Hardware** | NVIDIA TITAN X | Kaggle T4 GPU |
| **Training Time** | ~10 hours | < 6 hours |
| **Accuracy (Exact Match)** | 45.8% | **37.2%** |

### Addressing the "Evaluation Gap"
We implement two **composite metrics** to capture model quality more effectively:

#### **Semantic Similarity**
Weighted average of **SBERT cosine similarity** and **Jaccard similarity**, ensuring factual consistency.

$$
\text{Sim}(S_{\text{orig}}, S_{\text{neut}}) = (0.8 \times \text{Sim}_{\text{SBERT}}) + (0.2 \times \text{Sim}_{\text{Jaccard}})
$$

#### **Aggregate Bias Score**
A custom bias measure combining:
- Model confidence from bias detector  
- Lexicon-based subjectivity  
- Neutrality probability from ideology classifier  

$$
\text{Bias}(S) = \frac{P_{\text{detector}} + S_{\text{lexicon}} + (1 - P_{\text{neutral}})}{3}
$$

---

## File Structure

