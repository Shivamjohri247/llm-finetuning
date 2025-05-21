"""
Streamlit UI for TinyLlama fine-tuned model inference.
Run: streamlit run serve/inference_streamlit.py
"""
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os
import gc

def clear_cuda_memory():
    """Clear CUDA memory and garbage collect"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../train/checkpoints')
MODEL_PATH = os.environ.get("FINETUNED_MODEL_PATH", DEFAULT_MODEL_PATH)
st.title("TinyLlama Inference (Streamlit)")

# Initialize session state for model and tokenizer
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'pipe' not in st.session_state:
    st.session_state.pipe = None

# Add debug info
st.write("Debug Information:")
st.write(f"Model path: {MODEL_PATH}")
st.write(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    st.write(f"CUDA device: {torch.cuda.get_device_name(0)}")
    st.write(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    st.write(f"CUDA memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

# Load model only if not already loaded
if st.session_state.model is None:
    try:
        clear_cuda_memory()  # Clear memory before loading
        
        st.write("Loading tokenizer...")
        st.session_state.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        
        st.write("Loading model...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        st.session_state.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True,
            offload_folder="offload"  # Enable CPU offloading if needed
        )
        
        st.write(f"Creating pipeline on device {device}...")
        st.session_state.pipe = pipeline(
            "text-generation",
            model=st.session_state.model,
            tokenizer=st.session_state.tokenizer,
            device_map="auto"
        )
        st.write("Model loaded successfully!")
        
        if torch.cuda.is_available():
            st.write(f"CUDA memory after loading: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        clear_cuda_memory()
        st.stop()

prompt = st.text_area("Prompt", "What is the capital of France?")
max_new_tokens = st.slider("Max new tokens", 16, 512, 128)
temperature = st.slider("Temperature", 0.1, 1.5, 0.7)

if st.button("Generate") and prompt.strip():
    with st.spinner("Generating..."):
        try:
            if torch.cuda.is_available():
                st.write(f"CUDA memory before generation: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            
            st.write("Starting generation...")
            output = st.session_state.pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=st.session_state.tokenizer.pad_token_id,
                eos_token_id=st.session_state.tokenizer.eos_token_id
            )
            st.write("Generation complete!")
            st.write("Generated text:")
            st.write(output[0]["generated_text"])
            
            if torch.cuda.is_available():
                st.write(f"CUDA memory after generation: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
                clear_cuda_memory()
                st.write(f"CUDA memory after cleanup: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        except Exception as e:
            st.error(f"Error during generation: {str(e)}")
            st.write("Full error details:")
            st.exception(e)
            clear_cuda_memory()  # Clean up on error
