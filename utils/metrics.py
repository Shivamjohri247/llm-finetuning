"""
Evaluation metrics for summarization/Q&A (BLEU, ROUGE, latency).
"""
from rouge_score import rouge_scorer
import sacrebleu
import time

def compute_rouge(preds, refs):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    return [scorer.score(r, p) for r, p in zip(refs, preds)]

def compute_bleu(preds, refs):
    return sacrebleu.corpus_bleu(preds, [refs]) # Return the BLEUScore object

def measure_latency(fn, *args, **kwargs):
    start = time.time()
    result = fn(*args, **kwargs)
    latency = time.time() - start
    return result, latency
