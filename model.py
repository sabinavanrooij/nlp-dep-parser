#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:10:56 2017

@author: diana
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

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
        
        self.inputSize = word_embeddings_dim + tag_embeddings_dim # The number of expected features in the input x
        hiddenSize = self.inputSize #* 2 # 512? is this the same as outputSize?
        nLayers = 2
        
        self.biLstm = nn.LSTM(self.inputSize, hiddenSize, nLayers, bidirectional=True)
        
        nDirections = 2
        self.batch = 1 # this is per recommendation
        
        # Initial states
        self.hiddenState = Variable(torch.randn(nLayers * nDirections, self.batch, hiddenSize))
        self.cellState = Variable(torch.randn(nLayers * nDirections, self.batch, hiddenSize))
        
    def forward(self, wordsTensor, tagsTensor):
        wordEmbeds = self.word_embeddings(wordsTensor)
        tagEmbeds = self.tag_embeddings(tagsTensor)        
        
#        print(wordEmbeds.size())
#        print(tagEmbeds.size())
        
        assert len(wordsTensor) == len(tagsTensor)
        seq_len = len(wordsTensor)
        
#        inputs = Variable(torch.randn(seq_len, batch, inputSize))
        
        inputTensor = torch.cat((wordEmbeds, tagEmbeds), 1)
#        print(inputTensor.size())
        
        output, (self.hiddenState, self.cellState) = self.biLstm(inputTensor.view(seq_len, self.batch, self.inputSize), (self.hiddenState, self.cellState))
        
        return output, (self.hiddenState, self.cellState)
        