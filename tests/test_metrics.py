import unittest
import sys
import os

# Add the project root to the Python path to allow importing from utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.metrics import compute_bleu, compute_rouge # Assuming these are in utils.metrics

class TestMetrics(unittest.TestCase):

    def test_compute_bleu_perfect_match(self):
        preds = ["this is a test"]
        refs = ["this is a test"]
        # sacrebleu.corpus_bleu returns a BLEUScore object. We need .score
        bleu_score_obj = compute_bleu(preds, refs) 
        self.assertAlmostEqual(bleu_score_obj.score, 100.0, places=1)

    def test_compute_bleu_no_match(self):
        preds = ["completely different"]
        refs = ["this is a test"]
        bleu_score_obj = compute_bleu(preds, refs)
        # Score might not be exactly 0, but should be very low.
        # Depending on sacrebleu version and smoothing, 0 is expected for no n-gram overlap.
        self.assertLess(bleu_score_obj.score, 5.0) # Check if it's very low

    def test_compute_bleu_partial_match(self):
        preds = ["this is a"]
        refs = ["this is a test"]
        bleu_score_obj = compute_bleu(preds, refs)
        self.assertTrue(0 < bleu_score_obj.score < 100)

    # It would also be good to add tests for compute_rouge if time permits
    # For example:
    def test_compute_rouge_perfect_match(self):
        preds = ["hello world"]
        refs = ["hello world"]
        rouge_scores = compute_rouge(preds, refs)
        # rouge_scores is a list of dicts. [{ 'rouge1': Score(...), 'rougeL': Score(...) }]
        self.assertAlmostEqual(rouge_scores[0]['rougeL'].fmeasure, 1.0, places=2)

    def test_compute_rouge_no_match(self):
        preds = ["completely different"]
        refs = ["hello world"]
        rouge_scores = compute_rouge(preds, refs)
        self.assertLess(rouge_scores[0]['rougeL'].fmeasure, 0.1) # Check if very low

# To make the test runnable with 'python tests/test_metrics.py'
if __name__ == '__main__':
    unittest.main()
