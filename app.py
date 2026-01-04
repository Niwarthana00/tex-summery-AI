import streamlit as st
from transformers import pipeline
import time

# --- Page Config ---
st.set_page_config(
    page_title="Text Summarizer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
st.markdown("Transform long articles into concise summaries instantly using **T5 Transformer** technology.")
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
        return pipe, "Base T5-Small Model"

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model Status
    with st.spinner("Loading AI Model..."):
        summarizer, model_type = load_model()
        
    if model_type == "Custom Finetuned Model":
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
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📥 Input Text")
    text_input = st.text_area(
        "Paste your article here:",
        height=400,
        placeholder="Enter or paste your text here to generate a summary..."
    )
    
    if st.button("Generate Summary"):
        if text_input:
            with col2:
                with st.spinner("🧠 Analyzing text and generating summary..."):
                    try:
                        # Adding a small artificial delay for UX smoothness on fast CPUs
                        time.sleep(0.5) 
                        
                        # Ensure logic consistency
                        if min_len >= max_len:
                            max_len = min_len + 10
                        
                        start_time = time.time()
                        # Using Beam Search for better quality and adherence to length constraints
                        summary_output = summarizer(
                            text_input, 
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
                        
                        # Metrics (Optional UX touch)
                        st.caption(f"⚡ Generated in {round(end_time - start_time, 2)} seconds.")
                        
                    except Exception as e:
                        st.error(f"❌ An error occurred: {e}")
        else:
            st.toast("⚠️ Please enter some text first!")
            
with col2:
    # Placeholder or specific instruction if empty
    if not text_input:
        st.info("👈 Waiting for input... The summary will appear here.")
        st.markdown(
            """
            ### Features
            - **Fast Processing**: Uses optimized T5 architecture.
            - **Custom Trained**: Fine-tuned on news data for accuracy.
            - **Adjustable**: Control the length of your output.
            """
        )

# --- Footer ---
st.markdown("---")
st.center = True
st.markdown(
    "<div style='text-align: center; color: #666;'>Built with ❤️ using Streamlit & Hugging Face Transformers</div>", 
    unsafe_allow_html=True
)
