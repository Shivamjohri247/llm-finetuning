"""
Fine-tune Llama-3-1B model directly using HuggingFace Transformers Trainer.
Usage: python train/train.py --data data/processed/tokenized_data.jsonl --output train/checkpoints/
"""
import os
import argparse
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch

# MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # Made configurable via args

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [eval(line) for line in f]

def main(args):
    os.makedirs(args.output, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=os.environ.get("HF_TOKEN"))
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModelForCausalLM.from_pretrained(args.model_name, token=os.environ.get("HF_TOKEN"))
    data = load_jsonl(args.data)
    # Use pre-tokenized input_ids from JSONL
    input_ids = [item['input_ids'] for item in data]
    # Optionally, create attention_mask (1 for non-pad, 0 for pad)
    pad_token_id = tokenizer.pad_token_id
    attention_masks = [[1 if token != pad_token_id else 0 for token in ids] for ids in input_ids]
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, input_ids, attention_masks):
            self.input_ids = input_ids
            self.attention_masks = attention_masks
        def __len__(self):
            return len(self.input_ids)
        def __getitem__(self, idx):
            return {
                'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
                'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.long),
                'labels': torch.tensor(self.input_ids[idx], dtype=torch.long)
            }
    dataset = SimpleDataset(input_ids, attention_masks)
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        save_total_limit=1,
        save_steps=100,
        logging_steps=10,
        report_to=[],
        fp16=args.fp16,
    )
    data_collator = None  # Not needed, dataset already tokenized
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

def test_model(args):
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.output)
    tokenizer = AutoTokenizer.from_pretrained(args.output)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    prompt = args.input_text
    print(f"Prompt: {prompt}")
    outputs = pipe(prompt, max_new_tokens=128, do_sample=True, temperature=0.7)
    print("Generated:")
    print(outputs[0]['generated_text'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='Tokenized training data')
    parser.add_argument('--output', type=str, required=True, help='Output checkpoint directory')
    parser.add_argument('--model_name', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0', help='Model name for tokenizer and model')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1)  # Set default batch size to 1
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=128)  # Set default max_length to 128
    parser.add_argument('--fp16', action='store_true', help='Enable fp16 training')
    parser.add_argument('--test', action='store_true', help='Test the fine-tuned model')
    parser.add_argument('--input_text', type=str, default=None, help='Input text for testing')
    args = parser.parse_args()
    if args.test and args.input_text:
        test_model(args)
    else:
        if not args.data:
            parser.error('--data is required for training')
        main(args)
