#!/usr/bin/env python3
"""Cumulative N-gram BLEU score"""
import numpy as np


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence.
    Weights are distributed evenly across all n-grams up to n.
    """
    def get_ngram_precision(references, sentence, k):
        """Helper to calculate modified precision for a specific k-gram"""
        sentence_ngrams = [" ".join(sentence[i:i + k])
                           for i in range(len(sentence) - k + 1)]

        if not sentence_ngrams:
            return 0

        counts = {}
        for gram in sentence_ngrams:
            counts[gram] = counts.get(gram, 0) + 1

        max_counts = {}
        for gram in counts:
            max_ref_count = 0
            for ref in references:
                ref_ngrams = [" ".join(ref[i:i + k])
                              for i in range(len(ref) - k + 1)]
                count_in_ref = ref_ngrams.count(gram)
                if count_in_ref > max_ref_count:
                    max_ref_count = count_in_ref
            max_counts[gram] = max_ref_count

        keep = sum(min(counts[gram], max_counts.get(gram, 0))
                   for gram in counts)
        return keep / len(sentence_ngrams)

    precisions = []
    for k in range(1, n + 1):
        p_k = get_ngram_precision(references, sentence, k)

        if p_k == 0:
            return 0.0
        precisions.append(p_k)

    weights = 1 / n
    geometric_mean = np.exp(sum(weights * np.log(p) for p in precisions))

    c_len = len(sentence)
    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(ref_lens, key=lambda x: abs(x - c_len))

    if c_len > closest_ref_len:
        bp = 1
    else:
        bp = np.exp(1 - closest_ref_len / c_len) if c_len > 0 else 0

    return bp * geometric_mean
