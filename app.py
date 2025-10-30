import streamlit as st
import backend # This is our mock backend
import os
import base64 # For the download link
import PyPDF2 # For PDF reading
import docx # For Word doc reading

# --- Helper Functions for File Reading ---

def read_txt(file):
    """Reads a .txt file."""
    return file.read().decode("utf-8")

def read_pdf(file):
    """Reads a .pdf file."""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def read_docx(file):
    """Reads a .docx file."""
    try:
        doc = docx.Document(file)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return None

def get_download_link(text_data, filename="results.txt"):
    """Generates a link to download a text file."""
    b64 = base64.b64encode(text_data.encode()).decode()
    href = f'<a href="data:file/text;base64,{b64}" download="{filename}">Download Results as .txt File</a>'
    return href

# --- Main Application UI ---

st.set_page_config(page_title="Bias Neutralization Tool", layout="wide")
st.title("Automatic Bias Analysis and Neutralization")
st.markdown("Project for PHC-351 by Nithish Ravikkumar. This tool can analyze text for ideological bias, suggest neutral alternatives, and provide a bias score.")

# --- Sidebar for Options ---
st.sidebar.title("Analysis Options")
st.sidebar.markdown("Select which analyses to run on the input text.")

# The user's "dialog box" of options, implemented as checkboxes
run_classification = st.sidebar.checkbox("Bias Classification (Left/Right/Center)", value=True)
run_neutralization = st.sidebar.checkbox("Bias Neutralization", value=True)
run_score = st.sidebar.checkbox("Bias Score & Similarity", value=True)

# Store options in a dictionary to pass to the backend
options = {
    "classification": run_classification,
    "neutralization": run_neutralization,
    "score": run_score
}

# --- Main Panel for Input ---
input_text = ""

tab1, tab2 = st.tabs(["✍️ Text Input", "📁 File Upload"])

with tab1:
    text_area_input = st.text_area("Enter your text here:", height=300, placeholder="e.g., 'The politician exposed his own incompetency...'")
    if text_area_input:
        input_text = text_area_input

with tab2:
    uploaded_file = st.file_uploader("Upload a .txt, .pdf, or .docx file", type=["txt", "pdf", "docx"])
    if uploaded_file:
        with st.spinner(f"Reading {uploaded_file.name}..."):
            if uploaded_file.type == "text/plain":
                input_text = read_txt(uploaded_file)
            elif uploaded_file.type == "application/pdf":
                input_text = read_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                input_text = read_docx(uploaded_file)
        
        if input_text:
            st.success("File read successfully!")
            # Show a preview
            st.text_area("File Content Preview", input_text[:1000] + "...", height=150)

# --- Processing and Output ---
if st.button("Analyze Text", type="primary", use_container_width=True):
    if not input_text:
        st.warning("Please enter some text or upload a file first.")
    elif not any(options.values()):
        st.warning("Please select at least one analysis option in the sidebar.")
    else:
        with st.spinner("Analyzing text... This may take a moment."):
            # Call the backend with the input text and selected options
            results = backend.process_text(input_text, options)

        st.subheader("📊 Analysis Results")
        
        # Prepare a string for text download
        download_content = f"--- Original Text ---\n{input_text}\n\n--- Analysis Results ---\n"
        
        # Display outputs based on selected options
        col1, col2 = st.columns(2)

        with col1:
            if "neutralization" in results:
                st.markdown("### 🕊️ Bias Neutralization")
                neutral_text = results["neutralization"]["text"]
                st.text_area("Neutralized Text", neutral_text, height=300)
                download_content += f"\n[Neutralized Text]\n{neutral_text}\n"
        
        with col2:
            if "classification" in results:
                st.markdown("### 🧭 Bias Classification")
                label = results["classification"]["label"]
                score = results["classification"]["score"]
                
                # Use a different color for the "meter" based on the label
                if label == "Left":
                    color = "#3498DB" # Blue
                elif label == "Right":
                    color = "#E74C3C" # Red
                else:
                    color = "#F1C40F" # Yellow
                
                # Manually create a simple colored bar
                st.markdown(f"**Classification:** `{label}`")
                st.markdown(f"**Confidence:** `{score*100:.0f}%`")
                st.progress(score, text=f"{label} Lean")
                
                download_content += f"\n[Bias Classification]\nLabel: {label}\nConfidence: {score*100:.0f}%\n"

            if "score" in results:
                st.markdown("### 📈 Bias & Similarity Score")
                bias_score = results["score"]["bias_score"]
                similarity = results["score"]["similarity"]
                
                st.metric(label="Overall Bias Score", value=f"{bias_score*100:.1f} / 100")
                st.metric(label="Semantic Similarity (to original)", value=f"{similarity*100:.1f}%")
                
                download_content += f"\n[Scores]\nBias Score: {bias_score:.4f}\nSemantic Similarity: {similarity:.4f}\n"

        # --- Download Button ---
        st.markdown(get_download_link(download_content), unsafe_allow_html=True)

