#!/usr/bin/env python3
"""N-gram BLEU score"""
import numpy as np


def ngram_bleu(references, sentence, n):
    """Calculates the n-gram BLEU score for a sentence"""
    def get_ngrams(tokens, n):
        """Helper to extract n-grams from a list of tokens"""
        return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    candidate_ngrams = get_ngrams(sentence, n)
    c_len = len(sentence)

    if not candidate_ngrams:
        return 0.0

    counts = {}
    for gram in candidate_ngrams:
        counts[gram] = counts.get(gram, 0) + 1

    max_counts = {}
    for gram in counts:
        max_ref_count = 0
        for ref in references:
            ref_ngrams = get_ngrams(ref, n)
            count_in_ref = ref_ngrams.count(gram)
            if count_in_ref > max_ref_count:
                max_ref_count = count_in_ref
        max_counts[gram] = max_ref_count

    keep = sum(min(counts[gram], max_counts.get(gram, 0)) for gram in counts)
    precision = keep / len(candidate_ngrams)

    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(ref_lens, key=lambda x: abs(x - c_len))

    if c_len > closest_ref_len:
        bp = 1
    else:
        bp = np.exp(1 - closest_ref_len / c_len) if c_len > 0 else 0

    return bp * precision
