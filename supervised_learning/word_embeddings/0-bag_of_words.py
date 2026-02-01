#!/usr/bin/env python3
"""Bag Of Words"""
import numpy as np

def bag_of_words(sentences, vocab=None):
    """Creates a bag of words embedding matrix"""
    if vocab is None:
        words = set()
        for sentence in sentences:
            for word in sentence.split():
                words.add(word)
        features = sorted(list(words))
    else:
        features = vocab
    
    word2idx = {word: idx for idx, word in enumerate(features)}
    embeddings = np.zeros((len(sentences), len(features)), dtype=int)
    
    for i, sentence in enumerate(sentences):
        for word in sentence.split():
            if word in word2idx:
                embeddings[i, word2idx[word]] += 1
    
    return embeddings, features
