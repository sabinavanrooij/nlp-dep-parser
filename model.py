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
        # Is this necessary?
#        parameters = filter(lambda p: p.requires_grad, model.parameters())
#        model.optimizer = torch.optim.SGD(parameters, lr=0.01)

class DependencyParseModel(nn.Module):
    def __init__(self, word_embeddings_dim, tag_embeddings_dim, vocabulary_size, tag_uniqueCount, pretrainedWordEmbeddings=None, pretrainedTagEmbeddings=None):
        super().__init__()
        
        self.word_embeddings = nn.Embedding(vocabulary_size, word_embeddings_dim)
        if pretrainedWordEmbeddings.any():
            assert pretrainedWordEmbeddings.shape == (vocabulary_size, word_embeddings_dim)
            self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrainedWordEmbeddings))
        
        self.tag_embeddings = nn.Embedding(tag_uniqueCount, tag_embeddings_dim)
        if pretrainedTagEmbeddings.any():
            assert pretrainedTagEmbeddings.shape == (tag_uniqueCount, tag_embeddings_dim)
            self.tag_embeddings.weight.data.copy_(torch.from_numpy(pretrainedTagEmbeddings))
        
        # Save computation type by not training already trained word vectors
        disableTrainingForEmbeddings(self, self.word_embeddings, self.tag_embeddings)
        
        self.inputSize = word_embeddings_dim + tag_embeddings_dim # The number of expected features in the input x
        self.hiddenSize = self.inputSize #* 2 # 512? is this the same as outputSize?
        self.nLayers = 2
        
        self.biLstm = nn.LSTM(self.inputSize, self.hiddenSize, self.nLayers, bidirectional=True)
        
        self.nDirections = 2
        self.batch = 1 # this is per recommendation
        
        # Initial states
        self.hiddenState, self.cellState = self.initHiddenCellState()        
        
        # Input size of the MLP is the size of the output from previous step concatenated with another of the same size
        biLstmOutputSize = self.hiddenSize * self.nDirections
        mlpInputSize = biLstmOutputSize * 2
        self.mlp = MLP(mlpInputSize, hidden_size=mlpInputSize)
        
    def initHiddenCellState(self):
        hiddenState = Variable(torch.randn(self.nLayers * self.nDirections, self.batch, self.hiddenSize))
        cellState = Variable(torch.randn(self.nLayers * self.nDirections, self.batch, self.hiddenSize))
        
        return hiddenState, cellState    
    
    def forward(self, sentenceDependencies, w2i, t2i):
        
        sentenceInWords, sentenceInTags = sentenceDependencies.getSentenceInWordsAndInTags()
    
        wordsToIndices = [w2i[w] for w in sentenceInWords]
        words_tensor = torch.LongTensor(wordsToIndices)
        
        tagsToIndices = [t2i[t] for t in sentenceInTags]
        tags_tensor = torch.LongTensor(tagsToIndices)
        
        wordsTensor = Variable(words_tensor)
        tagsTensor = Variable(tags_tensor)
        
        wordEmbeds = self.word_embeddings(wordsTensor)
        tagEmbeds = self.tag_embeddings(tagsTensor)
        
#        print(wordEmbeds.size()[0])
#        print(tagEmbeds.size())
        
        assert len(wordsTensor) == len(tagsTensor)
        seq_len = len(wordsTensor)
        
        inputTensor = torch.cat((wordEmbeds, tagEmbeds), 1)
#        print(inputTensor.size())
        
        hVector, (self.hiddenState, self.cellState) = self.biLstm(inputTensor.view(seq_len, self.batch, self.inputSize), (self.hiddenState, self.cellState))
#        print(hVector)
        
        # MLP
        nWordsInSentence = wordEmbeds.size()[0]

        # Creation of dependency matrix. size: (length of sentence) x (length of sentence)
        scoreTensor = torch.FloatTensor(nWordsInSentence, nWordsInSentence).zero_()        
    
        # All possible combinations between head and dependent for the given sentence
        permutations = list(itertools.permutations([x for x in range(nWordsInSentence)], 2))
    
        # Concatenate the vector corresponding to the words for all permutations
        for permutation in permutations:
            hvectorConcat = torch.cat((hVector[permutation[0], :, :], hVector[permutation[1], :, :]), 1)
            score = self.mlp(hvectorConcat)
    
            # Fill dependency matrix
            scoreTensor[permutation[0], permutation[1]] = float(score.data[0].numpy()[0])

        # Normalize the columns
        for i in range(nWordsInSentence):
            scoreTensor[:, i] = scoreTensor[:, i] / sum(scoreTensor[:, i])
        
        # Use Softmax to get a positive value between 0 and 1
        m = nn.Softmax()
        scoreTensor = m(scoreTensor)
        
#        print(scoreTensor)
            
        # Do something with adjacency matrix
        # sentenceDependencies.getAdjacencyMatrx()
        
            
        return scoreTensor
