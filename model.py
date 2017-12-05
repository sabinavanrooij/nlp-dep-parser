#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:10:56 2017

@author: diana
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from MLP import MLP
import itertools

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
        self.hiddenSize = self.inputSize #* 2 # 512? is this the same as outputSize?
        self.nLayers = 2
        
        self.biLstm = nn.LSTM(self.inputSize, self.hiddenSize, self.nLayers, bidirectional=True)
        
        self.nDirections = 2
        self.batch = 1 # this is per recommendation
        
        # Initial states
        self.hiddenState, self.cellState = self.initHiddenCellState()
        
    def initHiddenCellState(self):
        hiddenState = Variable(torch.randn(self.nLayers * self.nDirections, self.batch, self.hiddenSize))
        cellState = Variable(torch.randn(self.nLayers * self.nDirections, self.batch, self.hiddenSize))
        
        return hiddenState, cellState
    
    
    def forward(self, wordsTensor, tagsTensor):
        wordEmbeds = self.word_embeddings(wordsTensor)
        tagEmbeds = self.tag_embeddings(tagsTensor)        
        
#        print(wordEmbeds.size()[0])
#        print(tagEmbeds.size())
        
        assert len(wordsTensor) == len(tagsTensor)
        seq_len = len(wordsTensor)
        
#        inputs = Variable(torch.randn(seq_len, batch, inputSize))
        
        inputTensor = torch.cat((wordEmbeds, tagEmbeds), 1)
#        print(inputTensor.size())
        
        hVector, (self.hiddenState, self.cellState) = self.biLstm(inputTensor.view(seq_len, self.batch, self.inputSize), (self.hiddenState, self.cellState))
        
        # MLP
        nWordsInSentence = wordEmbeds.size()[0]

        # Creation of dependency matrix. size: (length of sentence) x (length of sentence)
        scoreTensor = torch.FloatTensor(nWordsInSentence, nWordsInSentence).zero_()
    
        mlp = MLP(hVector[0, :, :].size()[1] * 2, 400)
    
        # All possible combinations between head and dependent for the given sentence
        permutations = list(itertools.permutations([x for x in range(nWordsInSentence)], 2))
    
        # Concatenate the vector corresponding to the words for all permutations
        for permutation in permutations:
            hvectorConcat = torch.cat((hVector[permutation[0], :, :], hVector[permutation[1], :, :]), 1)
            score = mlp(hvectorConcat)
    
            # Fill dependency matrix
            scoreTensor[permutation[0], permutation[1]] = float(score.data[0].numpy()[0])
        
        
        return scoreTensor
        