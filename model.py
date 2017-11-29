#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:10:56 2017

@author: diana
"""

import torch
import torch.nn as nn

class DependencyParseModel(nn.Module):
    def __init__(self, word_embeddings_dim, tag_embeddings_dim, vocabulary_size, tag_uniqueCount, pretrainedWordEmbeddings=None, pretrainedTagEmbeddings=None):
        super(DependencyParseModel, self).__init__()
        
        self.word_embeddings = nn.Embedding(vocabulary_size, word_embeddings_dim)
        if pretrainedWordEmbeddings.any():
            assert pretrainedWordEmbeddings.shape == (vocabulary_size, word_embeddings_dim)
            self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrainedWordEmbeddings))
        
        self.tag_embeddings = nn.Embedding(tag_uniqueCount, tag_embeddings_dim)
        if pretrainedTagEmbeddings.any():
            assert pretrainedTagEmbeddings.shape == (tag_uniqueCount, tag_embeddings_dim)
            self.tag_embeddings.weight.data.copy_(torch.from_numpy(pretrainedTagEmbeddings))
        
        inputSize = word_embeddings_dim + tag_embeddings_dim # The number of expected features in the input x
        hiddenSize = inputSize * 2 # 512?
        nLayers = 2
        
        self.biLstm = nn.LSTM(inputSize, hiddenSize, nLayers, bidirectional=True)
        
    def forward(self, sentence):
        return