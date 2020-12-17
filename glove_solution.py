#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
import random
from consts_glove import *

def main():
    print("loading cooccurrence matrix")
    with open(COOC_PATH, 'rb') as f:
        cooc = pickle.load(f)
    print("{} nonzero entries".format(cooc.nnz))

    nmax = 100
    print("using nmax =", nmax, ", cooc.max() =", cooc.max())

    print("initializing embeddings")
    
    eta = 0.01
    alpha = 3 / 4
    
    xs = np.random.normal(size=(cooc.shape[0], EMBEDDING_DIM))
    ys = np.random.normal(size=(cooc.shape[1], EMBEDDING_DIM))
    
    # initialize vectors for Adagrad
    g_t_main = np.ones(EMBEDDING_DIM)
    g_t_context = np.ones(EMBEDDING_DIM)

    for epoch in range(EPOCHS):
        print("epoch {}".format(epoch))
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):

            f = ((n/nmax)**alpha if n < nmax else 1)
            main = xs[ix, :]
            context = ys[jy, :]
            
            # compute gradients
            c = 2 * f * (np.dot(main, context) - np.log(n))
            g_main = c * context
            g_context = c * main
            
            # update rule
            xs[ix, :] -= (eta / np.sqrt(g_t_main)) * g_main
            ys[jy, :] -= (eta / np.sqrt(g_t_context)) * g_context
            
            # accumulate past gradients for Adagrad
            g_t_main += np.square(g_main)
            g_t_context += np.square(g_context)

    with open(VOCAB_CUT_PATH, 'r') as f:
        voc = f.read().splitlines()
        
    out = open(EMBEDDINGS_PATH,'w') 
    for i, token in enumerate(voc):
        out.write(voc[i] + ' ' + ' '.join(map(str, xs[i])) + '\n')
    out.close()


if __name__ == '__main__':
    main()
