# TinyLLM Fine-Tuning Pipeline

A streamlined pipeline for fine-tuning small language models (primarily TinyLlama-1.1B-Chat-v1.0) on domain-specific text using HuggingFace Transformers. Includes preprocessing, training, and deployment via FastAPI/Streamlit.

## Features
- **Efficient Fine-Tuning**: Memory-efficient training using HuggingFace Transformers
- **Domain Adaptation**: Easy preprocessing of custom text data
- **Low Resource Requirements**: Optimized for 8GB GPU machines
- **Dual Inference Options**: 
  - FastAPI server for production deployments
  - Streamlit UI for interactive demos
- **Reproducible Pipeline**: End-to-end scripts from data prep to deployment

## Project Structure
```
.
├── .github/              # GitHub Actions workflows (e.g., Pylint)
├── data/
│   ├── preprocess.py     # Data preprocessing and tokenization script
│   └── processed/        # Default output for processed data (content ignored by .gitignore)
├── train/
│   ├── train.py          # Model fine-tuning script
│   └── checkpoints/      # Default output for model checkpoints (content ignored by .gitignore)
├── optimize/
│   └── export_onnx.py    # Script to export model to ONNX
├── serve/
│   ├── inference_fastapi.py # FastAPI inference server
│   └── inference_streamlit.py # Streamlit UI for inference
├── utils/
│   ├── benchmark.py      # Performance benchmarking (example)
│   └── metrics.py        # Evaluation metrics (example)
├── tests/                # Unit tests (recommended, structure may vary)
├── .gitignore            # Specifies intentionally untracked files by Git
├── README.md             # This file
├── requirements.txt      # Project dependencies
└── setup.sh              # Environment setup script (creates venv, installs deps)
```

## Requirements
- Python 3.10+
- PyTorch (CUDA 11.8+ recommended for GPU)
- 8GB+ GPU (or CPU-only mode)
- Key dependencies:
  ```
  transformers
  fastapi
  uvicorn
  streamlit
  torch
  accelerate
  optimum[onnxruntime] # For ONNX export
  ```
- **HF_TOKEN**: Ensure you have the `HF_TOKEN` environment variable set if you are using models that require authentication from the Hugging Face Hub (e.g., private models or some Llama variants).

## Installation
```bash
git clone [repository-url]
cd llm-finetuning
pip install -r requirements.txt
```

## Usage Guide

### 1. Data Preprocessing
```bash
python data/preprocess.py \
  --input data/raw/ \
  --output data/processed/ \
  --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0
```
- Processes raw text into tokenized training data
- Supports local files or HuggingFace datasets
- Outputs: `data/processed/tokenized_data.jsonl`

### 2. Model Fine-Tuning
```bash
python train/train.py \
  --data data/processed/tokenized_data.jsonl \
  --output train/checkpoints/ \
  --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --epochs 3 \
  --batch_size 1 \
  --lr 2e-5 \
  --max_length 128 \
  # --fp16  # Uncomment to enable mixed-precision training if desired
```
Key Parameters:
- `model_name`: Identifier for the Hugging Face model to use (default: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`).
- `batch_size`: Default 1 for 8GB GPUs.
- `max_length`: 128 tokens recommended for memory efficiency.
- `fp16`: Optional flag to enable fp16 mixed-precision training (e.g., add `--fp16`). If not provided, training is in full precision (fp32).

### 3. Model Serving

**Note on Model Path:** The serving scripts (`inference_fastapi.py` and `inference_streamlit.py`) load the fine-tuned model from the path specified by the `FINETUNED_MODEL_PATH` environment variable. If this variable is not set, they default to looking for the model in `train/checkpoints/` (relative to the project root, assuming scripts are run from there or the path resolves correctly).

#### Option A: FastAPI Server
```bash
uvicorn serve.inference_fastapi:app --host 0.0.0.0 --port 8000
```
- RESTful API endpoint at `/generate`
- Example request:
  ```bash
  curl -X POST "http://localhost:8000/generate" \
       -H "Content-Type: application/json" \
       -d '{"prompt": "What is the capital of France?", "max_new_tokens": 128, "temperature": 0.7}'
  ```

#### Option B: Streamlit UI
```bash
streamlit run serve/inference_streamlit.py
```
- Interactive web interface
- Real-time text generation
- Memory-efficient with automatic CUDA management

## Memory Management
The implementation includes several optimizations for running on consumer GPUs:
- Automatic CUDA memory management
- Session state persistence in Streamlit
- CPU offloading when needed
- Garbage collection after inference

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch_size to 1
   - Decrease max_length
   - Enable CPU offloading

2. **Streamlit Device Errors**
   ```bash
   # Run with specific watcher type
   export STREAMLIT_WATCHER_TYPE=watchman
   # Or disable file watching
   streamlit run serve/inference_streamlit.py --server.runOnSave false
   ```

3. **Model Loading Issues**
   - Check CUDA availability
   - Verify checkpoint files
   - Clear CUDA cache if needed

## Best Practices
1. **Training**
   - Start with small batch sizes
   - Monitor GPU memory usage
   - Save checkpoints frequently

2. **Inference**
   - Use FastAPI for production
   - Use Streamlit for demos
   - Clear CUDA cache between runs

3. **Data Preparation**
   - Verify tokenizer matches model
   - Check for proper padding/truncation
   - Validate processed data format

## Contributing
Contributions welcome! Please check the issues page and submit PRs for improvements.

## License
MIT
