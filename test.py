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

trainingSetReader = ConlluFileReader(r"UD_English/en-ud-train.conllu")
trainingSet = trainingSetReader.readTrainingSet()

# This is for word2vec training of the POS tags
posTagsTrainingSet = []
for example in trainingSet:
    sentence = []
    for tokenKey in example.tokens:
        sentence.append(example.tokens[tokenKey].POSTag)
    posTagsTrainingSet.append(sentence)

#print(posTagsTrainingSet[0])

POSTagsModel = Word2Vec(posTagsTrainingSet, size=100, window=5, min_count=0, workers=4)
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
    
#print(wordsTrainingSet[0])

wordsModel = Word2Vec(wordsTrainingSet, size=word_embeddings_dim, window=5, min_count=0, workers=4)

for sentence in trainingSet:
    vectors = []
    for k, v in sentence.tokens.items():        
        if v.word in wordsEmbeddings:
            wordVector = wordsEmbeddings[v.word]
        else:
            wordVector = wordsModel[v.word]
        vectors.append(np.concatenate((wordVector, POSTagsModel[v.POSTag])))

#inputSize = 10
#hiddenSize = 20
#nLayers = 2
#
#bilstm = torch.nn.LSTM(inputSize, hiddenSize, nLayers)




#writer = ConlluFileWriter('cosa.conllu')
#writer.write(trainingSet)