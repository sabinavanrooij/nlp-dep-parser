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
#import math

def disableTrainingForEmbeddings(*embeddingLayers):
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
        disableTrainingForEmbeddings(self.word_embeddings, self.tag_embeddings)
        
        self.inputSize = word_embeddings_dim + tag_embeddings_dim # The number of expected features in the input x
        self.hiddenSize = self.inputSize #* 2 # 512? is this the same as outputSize?
        self.nLayers = 2
        
        self.biLstm = nn.LSTM(self.inputSize, self.hiddenSize, self.nLayers, bidirectional=True)
        
        self.nDirections = 2
        self.batch = 1 # this is per recommendation
 
        # Input size of the MLP for arcs scores is the size of the output from previous step concatenated with another of the same size
        biLstmOutputSize = self.hiddenSize * self.nDirections
        mlpForScoresInputSize = biLstmOutputSize * 2
        self.mlpArcsScores = MLP(mlpForScoresInputSize, hidden_size=mlpForScoresInputSize, output_size=1)
        
        # MLP for labels
        self.label_uniqueCount = label_uniqueCount
        self.mlpLabels = MLP(mlpForScoresInputSize, hidden_size=mlpForScoresInputSize, output_size=self.label_uniqueCount)
        
    def initHiddenCellState(self):
        hiddenState = Variable(torch.randn(self.nLayers * self.nDirections, self.batch, self.hiddenSize), requires_grad=False)
        cellState = Variable(torch.randn(self.nLayers * self.nDirections, self.batch, self.hiddenSize), requires_grad=False)
        
        return hiddenState, cellState    
    
    def forward(self, words_tensor, tags_tensor, arcs_refdata_tensor):

        scoreTensor = self.predictArcs(words_tensor, tags_tensor)
        labelTensor = self.predictLabels(arcs_refdata_tensor)        

#        hvectorConcatForArcs = torch.cat((self.hVector[0, :, :], self.hVector[1, :, :]), 1)
#        
#        for i in range(nWordsInSentence + 1):
#            for j in range(nWordsInSentence + 1):             
#                scoreTensor[i,j] = self.mlpArcsScores(hvectorConcatForArcs).data[0,0]

        return scoreTensor,labelTensor
#        return scoreTensor, labelTensor

    def predictArcs(self, words_tensor, tags_tensor):        
                
        # BiLSTM        
        wordEmbeds = self.word_embeddings(words_tensor)
        tagEmbeds = self.tag_embeddings(tags_tensor)
        
        assert len(tags_tensor) == len(tags_tensor)
        seq_len = len(words_tensor)
        
        inputTensor = torch.cat((wordEmbeds, tagEmbeds), 1)
        
        self.hVector, (self.hiddenState, self.cellState) = self.biLstm(inputTensor.view(seq_len, self.batch, self.inputSize), (self.hiddenState, self.cellState))
        
        # MLP for arcs scores
        nWordsInSentence = words_tensor.size()[0]

        # Creation of dependency matrix. size: (length of sentence + 1)  x (length of sentence + 1)
        scoreTensor = Variable(torch.zeros(nWordsInSentence + 1, nWordsInSentence + 1))
        
        # We know root goes to root, so adding a 1 there
        scoreTensor[0,0] = 1.0

        # All possible combinations between head and dependent for the given sentence
        for head in range(nWordsInSentence):
            for dep in range(nWordsInSentence):
                if head == dep:
                    continue
                
                hvectorConcatForArcs = torch.cat((self.hVector[head, :, :], self.hVector[dep, :, :]), 1)
                scoreVar = self.mlpArcsScores(hvectorConcatForArcs)
                scoreTensor[head + 1, dep + 1] = scoreVar
                
                hvectorConcatForArcs = torch.cat((self.hVector[dep, :, :], self.hVector[head, :, :]), 1)
                scoreVar = self.mlpArcsScores(hvectorConcatForArcs)
                scoreTensor[dep + 1, head + 1] = scoreVar
                
        return scoreTensor
    
    def predictLabels(self, arcs_refdata_tensor):
        # headsIndices is nWordsInSentence + 1 because it's accounting for the root first position
        
        # MLP for labels
        # Creation of matrix with label-probabilities. size: (length of sentence) x (unique tag count)
        labelTensor = Variable(torch.zeros(len(arcs_refdata_tensor) - 1, self.label_uniqueCount))
        
        # WE DONT REALLY FILL IN THE WHOLE THING> CHECK THIS!!
        
        assert len(arcs_refdata_tensor) - 1 == self.hVector.size()[0]

        for i, head in enumerate(arcs_refdata_tensor[1:]):
            # skip all elements that have root (0) as head since we don't have hVectors for root and thus nothing to concatenate
            if head == 0:
                continue
            
            hvectorConcatForLabels = torch.cat((self.hVector[i, :, :], self.hVector[head - 1, :, :]), 1)
            score = self.mlpLabels(hvectorConcatForLabels)
            labelTensor[i, :] = score
        
        return labelTensor