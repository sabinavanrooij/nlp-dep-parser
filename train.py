# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
from sentenceDependencies import ConlluFileReader #, ConlluFileWriter
from dataProcessor import DataProcessor
from wordEmbeddingsReader import GloVeFileReader
from gensim.models import Word2Vec
import numpy as np
from model import DependencyParseModel
import torch.nn as nn
from torch.autograd import Variable
from random import shuffle
import time

unknownMarker = '<unk>'

sentencesReader = ConlluFileReader(r"UD_English/en-ud-train.conllu")
sentencesDependencies = sentencesReader.readSentencesDependencies(unknownMarker)

dataProcessor = DataProcessor(sentencesDependencies)
w2i, t2i, l2i, i2w, i2t, i2l = dataProcessor.buildDictionaries()

sentencesInWords, sentencesInTags = dataProcessor.getTrainingSetsForWord2Vec()

word_embeddings_dim = 50
posTags_embeddings_dim = 50
minCountWord2Vec_words = 5
minCountWord2Vec_tags = 0

# Train the POS tags
POSTagsModel = Word2Vec(sentencesInTags, size=posTags_embeddings_dim, window=5, min_count=minCountWord2Vec_tags, workers=4)

# Read the word embeddings
wordEmbeddingsReader = GloVeFileReader(r"GloVe/glove.6B.50d.txt")
wordsEmbeddings = wordEmbeddingsReader.readWordEmbeddings()

# Or train the embeddings too
wordsModel = Word2Vec(sentencesInWords, size=word_embeddings_dim, window=5, min_count=minCountWord2Vec_words, workers=4)

# LSTM training prep
vocabularySize = len(w2i)
tagsUniqueCount = len(t2i)

pretrainedWordEmbeddings = np.empty((vocabularySize, word_embeddings_dim))
for k,v in i2w.items():
    if v in wordsEmbeddings:
        pretrainedWordEmbeddings[k] = wordsEmbeddings[v]
    elif v in wordsModel.wv.vocab:
        pretrainedWordEmbeddings[k] = wordsModel[v]
    else:
        pretrainedWordEmbeddings[k] = wordsModel[unknownMarker]

pretrainedTagEmbeddings = np.empty((tagsUniqueCount, posTags_embeddings_dim))
for k,v in i2t.items():
    assert v in POSTagsModel.wv.vocab
    pretrainedTagEmbeddings[k] = POSTagsModel[v]


model = DependencyParseModel(word_embeddings_dim, posTags_embeddings_dim, vocabularySize, tagsUniqueCount, pretrainedWordEmbeddings, pretrainedTagEmbeddings)
parameters = filter(lambda p: p.requires_grad, model.parameters())
parameters = nn.ParameterList(list(parameters))
optimizer = torch.optim.SGD(parameters, lr=0.01)

epochs = 1
lossgraph = []
counter = 0
outputarray = []

for epoch in range(epochs):
    shuffle(sentencesDependencies)
    total_output = 0
    for s in sentencesDependencies:
        # Clear hidden and cell previous state
        model.hiddenState, model.cellState = model.initHiddenCellState()

        # Forward pass
        result = model(s, w2i, t2i)
        # print(result) # result so far is scores matrix

        # Get reference data (gold)
        refdata = s.getAdjacencyMatrix()
        refdata = torch.from_numpy(refdata)
        refdata = refdata.float()

        #get sentence length
        sentence_length = len(s.tokens)

        # Calculate loss
        # here output is the sum of the losses over the columns
        output = 0
        for column in range(0, sentence_length+1):
            loss = nn.BCELoss()
            modelinput = result[:, column]
            target = Variable(refdata[:, column])
            output += loss(modelinput, target)
        #print("this is the output ",output)
        output.backward()
        optimizer.step()
        counter += 1 
        #running_loss += output.data[0]
        outputarray.append(output.data[0])
        total_output += output.data[0]
        # just for testing purposes. Remove when doing the actual training
        if counter == 10:
            break
        
    lossgraph.append(total_output)

print(outputarray)
print(lossgraph)
#date = str(time.strftime("%d_%m"))
#savename = "DependencyParserModel_" + date + ".pkl"
savename = "DependencyParserModel.pkl"
torch.save(model.state_dict(), savename)



