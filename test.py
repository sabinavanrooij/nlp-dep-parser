# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sentenceDependencies import ConlluFileReader, ConlluFileWriter
from wordEmbeddingsReader import GloVeFileReader
from gensim.models import Word2Vec
import numpy as np
import torch
from torch.autograd import Variable


# Read sentence dependencies from file
trainingSetReader = ConlluFileReader(r"UD_English/en-ud-train.conllu")
trainingSet = trainingSetReader.readTrainingSet()

# This is for word2vec training of the POS tags
posTagsTrainingSet = []
for example in trainingSet:
    sentence = []
    for tokenKey in example.tokens:
        sentence.append(example.tokens[tokenKey].POSTag)
    posTagsTrainingSet.append(sentence)

posTags_embeddings_dim = 100

POSTagsModel = Word2Vec(posTagsTrainingSet, size=posTags_embeddings_dim, window=5, min_count=0, workers=4)
#print(POSTagsModel.wv.vocab.keys())
#print(type(POSTagsModel['DET']))

word_embeddings_dim = 50

# Read the word embeddings
wordEmbeddingsReader = GloVeFileReader(r"GloVe/glove.6B.50d.txt")
wordsEmbeddings = wordEmbeddingsReader.readWordEmbeddings()

# Or train the embeddings too
wordsTrainingSet = []
for example in trainingSet:
    sentence = []
    for tokenKey in example.tokens:
        sentence.append(example.tokens[tokenKey].word)
    wordsTrainingSet.append(sentence)

wordsModel = Word2Vec(wordsTrainingSet, size=word_embeddings_dim, window=5, min_count=0, workers=4)

sentenceVectors = []
for sentence in trainingSet:
    vectors = []
    for k, v in sentence.tokens.items():        
        if v.word in wordsEmbeddings:
            wordVector = wordsEmbeddings[v.word]
        else:
            wordVector = wordsModel[v.word]
        vectors.append(np.concatenate((wordVector, POSTagsModel[v.POSTag])))
    sentenceVectors.append(vectors)

# I'm guessing this is per sentence

for vector in sentenceVectors:    

    # LSTM training
    inputSize = word_embeddings_dim + posTags_embeddings_dim # The number of expected features in the input x
    hiddenSize = inputSize * 2 # 512?
    nLayers = 2
    nDirections = 2
    batch = 3
    seq_len = len(vector) #5 # time
    
    biLstm = torch.nn.LSTM(inputSize, hiddenSize, nLayers, bidirectional=True)
    
#    inputVar = Variable(torch.randn(seq_len, batch, inputSize))
    
#    inputVar = Variable(torch.from_numpy(np.array(vector)))
    
    print(np.array(vector).shape)
    break
    
#    hiddenState0Var = Variable(torch.randn(nLayers * nDirections, batch, hiddenSize))
#    cellState0Var = Variable(torch.randn(nLayers * nDirections, batch, hiddenSize))
#    output, hidden = biLstm(inputVar, (hiddenState0Var, cellState0Var))
    
    #print(output)
    #print(hidden)



#writer = ConlluFileWriter('cosa.conllu')
#writer.write(trainingSet)