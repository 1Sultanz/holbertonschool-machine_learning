#!/usr/bin/env python
"""Unigram BLEU score"""
import numpy as np

def uni_bleu(references, sentence):
    """Calculates the unigram BLEU score for a sentence"""
    c_len = len(sentence)

    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(ref_lens, key=lambda x: abs(x - c_len))

    counts = {}
    for word in sentence:
        counts[word] = counts.get(word, 0) + 1

    max_counts = {}
    for word in counts:
        max_ref_count = 0
        for ref in references:
            count_in_ref = ref.count(word)
            if count_in_ref > max_ref_count:
                max_ref_count = count_in_ref
        max_counts[word] = max_ref_count

    keep = sum(min(counts[word], max_counts.get(word, 0)) for word in counts)
    
    precision = keep / c_len if c_len > 0 else 0

    if c_len > closest_ref_len:
        bp = 1
    else:
        bp = np.exp(1 - closest_ref_len / c_len) if c_len > 0 else 0

    return bp * precision
