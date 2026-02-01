#!/usr/bin/env python3
"""Train Word2Vec"""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """Creates , builds and trains a gensim word2vec model"""
    if cbow:
        sg = 0
    else:
        sg = 1

    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        seed=seed,
        epochs=epochs,
        workers=workers
    )
                     
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count,
                epochs=model.epochs)
                     
    return model
