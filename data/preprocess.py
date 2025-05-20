"""
Preprocess and tokenize domain-specific text data for LLM fine-tuning.
Usage: python data/preprocess.py --input data/raw/ --output data/processed/
"""
import os
import argparse
import pandas as pd
from transformers import AutoTokenizer
import sys
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

def preprocess_text(text):
    # Basic cleaning, can be extended for domain-specific needs
    return text.strip().replace("\n", " ")

def main(args):
    os.makedirs(args.output, exist_ok=True)
    # Always use the correct model/tokenizer for preprocessing
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name if hasattr(args, 'model_name') else 'meta-llama/Llama-3.2-1B-Instruct',
        token=os.environ.get("HF_TOKEN")
    )
    # Ensure pad_token is set (match training script)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    all_data = []
    if hasattr(args, 'hf_dataset') and args.hf_dataset:
        if load_dataset is None:
            print('Please install the datasets library: pip install datasets')
            sys.exit(1)
        print(f"Loading HuggingFace dataset: {args.hf_dataset}")
        ds = load_dataset(args.hf_dataset, split='train')
        for i, item in enumerate(ds):
            if hasattr(args, 'max_records') and args.max_records and i >= args.max_records:
                break
            text = preprocess_text(item.get('text', str(item)))
            tokens = tokenizer(text, truncation=True, padding='max_length', max_length=args.max_length)
            all_data.append({"input_text": text, "input_ids": tokens["input_ids"]})
    else:
        for fname in os.listdir(args.input):
            if fname.endswith('.txt'):
                with open(os.path.join(args.input, fname), 'r', encoding='utf-8') as f:
                    text = preprocess_text(f.read())
                    tokens = tokenizer(text, truncation=True, padding='max_length', max_length=args.max_length)
                    all_data.append({"input_text": text, "input_ids": tokens["input_ids"]})
    df = pd.DataFrame(all_data)
    df.to_json(os.path.join(args.output, "tokenized_data.jsonl"), orient="records", lines=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=False, help='Input raw data folder')
    parser.add_argument('--output', type=str, required=True, help='Output processed data folder')
    parser.add_argument('--model_name', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0', help='Model name for tokenizer')
    parser.add_argument('--max_length', type=int, default=2048, help='Max token length')
    parser.add_argument('--hf_dataset', type=str, required=False, help='HuggingFace dataset name (optional)')
    parser.add_argument('--max_records', type=int, required=False, help='Max records to process from dataset')
    args = parser.parse_args()
    main(args)
