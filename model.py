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

def disableTrainingForEmbeddings(model, *embeddingLayers):
    for e in embeddingLayers:
        e.weight.requires_grad = False

class DependencyParseModel(nn.Module):
    def __init__(self, word_embeddings_dim, tag_embeddings_dim, vocabulary_size, tag_uniqueCount, label_uniqueCount, pretrainedWordEmbeddings=None, pretrainedTagEmbeddings=None):
        super().__init__()
        
        self.word_embeddings = nn.Embedding(vocabulary_size, word_embeddings_dim)
        if pretrainedWordEmbeddings.any():
            assert pretrainedWordEmbeddings.shape == (vocabulary_size, word_embeddings_dim)
            self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrainedWordEmbeddings))
        
        self.tag_embeddings = nn.Embedding(tag_uniqueCount, tag_embeddings_dim)
        if pretrainedTagEmbeddings.any():
            assert pretrainedTagEmbeddings.shape == (tag_uniqueCount, tag_embeddings_dim)
            self.tag_embeddings.weight.data.copy_(torch.from_numpy(pretrainedTagEmbeddings))
        
        # Save computation time by not training already trained word vectors
        disableTrainingForEmbeddings(self, self.word_embeddings, self.tag_embeddings)
        
        self.inputSize = word_embeddings_dim + tag_embeddings_dim # The number of expected features in the input x
        self.hiddenSize = self.inputSize #* 2 # 512? is this the same as outputSize?
        self.nLayers = 2
        
        self.biLstm = nn.LSTM(self.inputSize, self.hiddenSize, self.nLayers, bidirectional=True)
        
        self.nDirections = 2
        self.batch = 1 # this is per recommendation
        
        # Initial states
        self.hiddenState, self.cellState = self.initHiddenCellState()        
        
        # Input size of the MLP for arcs scores is the size of the output from previous step concatenated with another of the same size
        biLstmOutputSize = self.hiddenSize * self.nDirections
        mlpForScoresInputSize = biLstmOutputSize * 2
        self.mlpArcsScores = MLP(mlpForScoresInputSize, mlpForScoresInputSize, 1)

        # MLP for labels
        self.label_uniqueCount = label_uniqueCount
        self.mlpLabels = MLP(mlpForScoresInputSize, mlpForScoresInputSize, self.label_uniqueCount)
        
        
    def initHiddenCellState(self):
        hiddenState = Variable(torch.randn(self.nLayers * self.nDirections, self.batch, self.hiddenSize))
        cellState = Variable(torch.randn(self.nLayers * self.nDirections, self.batch, self.hiddenSize))
        
        return hiddenState, cellState    
    
    def forward(self, words_tensor, tags_tensor, headsIndices):
        scoreTensor = self.predictArcs(words_tensor, tags_tensor)
        labelTensor = self.predictLabels(headsIndices)        

        # Make scoreTensor and labelTensor variables so we can update weights
        scoreTensor = nn.Parameter(scoreTensor, requires_grad=True)
        labelTensor = nn.Parameter(labelTensor, requires_grad=True)
        
        return scoreTensor, labelTensor

    def predictArcs(self, words_tensor, tags_tensor):
        # BiLSTM
        wordsTensor = Variable(words_tensor)
        tagsTensor = Variable(tags_tensor)
        
        wordEmbeds = self.word_embeddings(wordsTensor)
        tagEmbeds = self.tag_embeddings(tagsTensor)
        
        assert len(wordsTensor) == len(tagsTensor)
        seq_len = len(wordsTensor)
        
        inputTensor = torch.cat((wordEmbeds, tagEmbeds), 1)        
        self.hVector, (self.hiddenState, self.cellState) = self.biLstm(inputTensor.view(seq_len, self.batch, self.inputSize), (self.hiddenState, self.cellState))
        
        # MLP for arcs scores
        nWordsInSentence = wordEmbeds.size()[0]

        # Creation of dependency matrix. size: (length of sentence + 1)  x (length of sentence + 1)
        scoreTensor = torch.FloatTensor(nWordsInSentence + 1, nWordsInSentence + 1).zero_()

        # All possible combinations between head and dependent for the given sentence
        permutations = list(itertools.permutations([x for x in range(nWordsInSentence)], 2))
    
        # Concatenate the vector corresponding to the words for all permutations
        for permutation in permutations:
            hvectorConcatForArcs = torch.cat((self.hVector[permutation[0], :, :], self.hVector[permutation[1], :, :]), 1)
            score = self.mlpArcsScores(hvectorConcatForArcs)
    
            # Fill dependency matrix
            scoreTensor[permutation[0] + 1, permutation[1] + 1] = float(score.data[0].numpy()[0])
        
        return scoreTensor
    
    def predictLabels(self, headsIndices):
        # headsIndices is nWordsInSentence + 1 because it's accounting for the root first position
        
        # MLP for labels
        # Creation of matrix with label-probabilities. size: (length of sentence) x (unique tag count)
        labelTensor = torch.FloatTensor(len(headsIndices) - 1, self.label_uniqueCount).zero_()
        
        # WE DONT REALLY FILL IN THE WHOLE THING> CHECK THIS!!
        
        assert len(headsIndices) - 1 == self.hVector.size()[0]

        for i, head in enumerate(headsIndices[1:]):
            # skip all elements that have root (0) as head since we don't have hVectors for root and thus nothing to concatenate
            if head == 0:
                continue
            
            hvectorConcatForLabels = torch.cat((self.hVector[i, :, :], self.hVector[head - 1, :, :]), 1)
            score = self.mlpLabels(hvectorConcatForLabels)
            labelTensor[i, :] = score.data[0]
        
        return labelTensor