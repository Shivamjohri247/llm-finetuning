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
├── data/
│   ├── preprocess.py      # Data preprocessing and tokenization
│   └── processed/         # Processed and tokenized data
├── train/
│   ├── train.py          # Model fine-tuning script
│   └── checkpoints/      # Saved model checkpoints
├── serve/
│   ├── inference_fastapi.py  # FastAPI inference server
│   └── inference_streamlit.py # Streamlit UI
└── utils/
    ├── benchmark.py      # Performance benchmarking
    └── metrics.py        # Evaluation metrics
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
  ```

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
  --epochs 3 \
  --batch_size 1 \
  --lr 2e-5 \
  --max_length 128
```
Key Parameters:
- `batch_size`: Default 1 for 8GB GPUs
- `max_length`: 128 tokens recommended for memory efficiency
- `fp16`: Disabled by default for stability

### 3. Model Serving

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
