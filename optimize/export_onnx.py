"""
Export a fine-tuned LLM to ONNX or TensorRT for optimized inference.
Usage: python optimize/export_onnx.py --checkpoint train/checkpoints/ --output optimize/onnx/
"""
import os
import argparse
from transformers import AutoTokenizer # Removed AutoModelForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM

def main(args):
    os.makedirs(args.output, exist_ok=True)
    # model = AutoModelForCausalLM.from_pretrained(args.checkpoint) # Removed this line
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    # Changed to use from_pretrained with export=True
    onnx_model = ORTModelForCausalLM.from_pretrained(args.checkpoint, export=True)
    onnx_model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"ONNX model exported to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to fine-tuned model checkpoint')
    parser.add_argument('--output', type=str, required=True, help='Output directory for ONNX model')
    args = parser.parse_args()
    main(args)
