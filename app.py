import streamlit as st
import requests
from bs4 import BeautifulSoup
from io import BytesIO
from pypdf import PdfReader
from transformers import pipeline
import time

# --- Page Config ---
st.set_page_config(
    page_title="Text Summarizer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions for Extraction ---
def extract_text_from_url(url):
    try:
        # User-Agent header to mimic a browser (avoid 403 errors)
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Improve extraction: grab paragraphs
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        
        if len(text) < 50:
            return None, "No substantial text found. Website might be blocking scrapers."
            
        return text, None
    except Exception as e:
        return None, str(e)

def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text, None
    except Exception as e:
        return None, str(e)

# --- Custom CSS for Modern Look ---
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #0e1117;
    }
    
    /* Header styling */
    h1 {
        font-family: 'Helvetica Neue', sans-serif;
        color: #FFFFFF;
        font-weight: 700;
        margin-bottom: 20px;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        background-color: #262730;
        color: #FAFAFA;
        border-radius: 10px;
        border: 1px solid #4B5563;
    }
    .stTextArea textarea:focus {
        border-color: #3B82F6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background-image: linear-gradient(to right, #4F46E5, #9333EA);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-image: linear-gradient(to right, #4338CA, #7E22CE);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transform: translateY(-2px);
    }
    
    /* Success info box */
    .stSuccess {
        background-color: #1c1c2e;
        border-left: 5px solid #10B981;
        padding: 20px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("✨ Text Summarizer System")
st.markdown("Transform long articles, PDFs, or Webpages into concise summaries instantly using **T5 Transformer** technology.")
st.markdown("---")

# --- Model Loading (Cached) ---
model_path = "./model"

@st.cache_resource
def load_model():
    try:
        # Check if local model exists
        pipe = pipeline("summarization", model=model_path, tokenizer=model_path)
        return pipe, "Custom Finetuned Model"
    except Exception:
        # Fallback
        pipe = pipeline("summarization", model="t5-small")
        return pipe, "Base T5-Small Model (Fallback)"

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model Status
    with st.spinner("Loading AI Model..."):
        summarizer, model_type = load_model()
        
    if "Custom" in model_type:
        st.success(f"✅ Active: {model_type}")
    else:
        st.warning(f"⚠️ Active: {model_type}")
        
    st.divider()
    
    # Parameters
    st.subheader("Summary Control")
    min_len = st.slider("Min Length", 10, 100, 30, help="Minimum number of words in the summary")
    max_len = st.slider("Max Length", 50, 500, 150, help="Maximum number of words in the summary")
    
    st.info("💡 **Tip:** Adjust lengths to get more detailed or concise results.")

# --- Main Interface ---

st.subheader("📥 Choose Input Source")
tab1, tab2, tab3 = st.tabs(["📝 Paste Text", "🌐 Web URL", "📄 Upload PDF"])

final_input_text = ""

# TAB 1: Direct Text Input
with tab1:
    text_input_raw = st.text_area(
        "Paste your article here:",
        height=300,
        placeholder="Enter or paste your text here to generate a summary..."
    )
    if text_input_raw:
        final_input_text = text_input_raw

# TAB 2: URL Input
with tab2:
    url_input = st.text_input("Enter Article URL:", placeholder="https://example.com/news-article")
    if url_input:
        if st.button("Fetch Content"):
            with st.spinner("🕷️ Crawling website..."):
                extracted_text, error = extract_text_from_url(url_input)
                if error:
                    st.error(f"❌ Error fetching URL: {error}")
                else:
                    st.success("✅ Content fetched successfully!")
                    with st.expander("👁️ Preview Fetched Text"):
                        st.text(extracted_text[:1000] + "...")
                    
                    # Auto-populate for summarization if user wants
                    st.session_state['url_text'] = extracted_text

    # Check if we have stored text from URL in session state
    if 'url_text' in st.session_state and url_input:
        final_input_text = st.session_state['url_text']


# TAB 3: PDF Input
with tab3:
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    if uploaded_file:
        with st.spinner("📄 Reading PDF..."):
            extracted_text, error = extract_text_from_pdf(uploaded_file)
            if error:
                st.error(f"❌ Error reading PDF: {error}")
            else:
                st.success("✅ PDF loaded successfully!")
                with st.expander("👁️ Preview Extracted Text"):
                    st.text(extracted_text[:1000] + "...")
                final_input_text = extracted_text

st.markdown("---")

# --- Generation Section ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### Ready to Summarize?")
    if final_input_text:
        st.info(f"Arguments: Min={min_len}, Max={max_len}")
        if st.button("🚀 Generate Summary", type="primary"):
             with col2:
                with st.spinner("🧠 Analyzing and summarizing..."):
                    try:
                        # Adding a small artificial delay for UX smoothness
                        time.sleep(0.5) 
                        
                        # Ensure logic consistency
                        if min_len >= max_len:
                            max_len = min_len + 10
                        
                        start_time = time.time()
                        # Using Beam Search for better quality
                        summary_output = summarizer(
                            final_input_text, 
                            max_length=max_len, 
                            min_length=min_len, 
                            num_beams=4, 
                            length_penalty=2.0,
                            early_stopping=True
                        )
                        end_time = time.time()
                        
                        summary_text = summary_output[0]['summary_text']
                        
                        st.subheader("📤 Generated Summary")
                        st.success(summary_text)
                        
                        # Metrics
                        st.caption(f"⚡ Completed in {round(end_time - start_time, 2)} seconds.")
                        
                    except Exception as e:
                        st.error(f"❌ Model Error: {e}")
    else:
        st.warning("👈 Please provide input text in one of the tabs above.")

with col2:
    if not final_input_text:
        st.write("") # Spacer

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>Built with ❤️ using Streamlit & Hugging Face Transformers</div>", 
    unsafe_allow_html=True
)
