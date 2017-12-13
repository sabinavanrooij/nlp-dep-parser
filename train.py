# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
from conlluFilesOperations import ConlluFileReader
from dataProcessor import DataProcessor
from wordEmbeddingsReader import GloVeFileReader
from gensim.models import Word2Vec
import numpy as np
from model import DependencyParseModel
import torch.nn as nn
from torch.autograd import Variable
from random import shuffle
import time
import matplotlib.pyplot as plt
import datetime


#### This is important. Remove when done double checking all the code

useDummyTrainData = True

####

unknownMarker = '<unk>'

sentencesReader = ConlluFileReader(r"UD_English/en-ud-train.conllu")
sentencesDependencies = sentencesReader.readSentencesDependencies(unknownMarker)


# Select 5 short sentences from the train file (of sentence length at most 10, say)
if useDummyTrainData:
    nSentencesToUse = 5
    nSentencesSoFar = 0
    newSentencesDependencies = []
    i = 0
    while(nSentencesSoFar < nSentencesToUse):
        if(len(sentencesDependencies[i].tokens) <= 10):
            newSentencesDependencies.append(sentencesDependencies[i])
            nSentencesSoFar += 1
        i += 1
        
    sentencesDependencies = newSentencesDependencies

dataProcessor = DataProcessor(sentencesDependencies, unknownMarker)
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
labelsUniqueCount = len(l2i)

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


model = DependencyParseModel(word_embeddings_dim, posTags_embeddings_dim, vocabularySize, tagsUniqueCount, labelsUniqueCount, pretrainedWordEmbeddings, pretrainedTagEmbeddings)
parameters = filter(lambda p: p.requires_grad, model.parameters())
parameters = nn.ParameterList(list(parameters))
optimizer = torch.optim.Adam(parameters, lr=0.01, weight_decay=1E-6)

epochs = 1000 if useDummyTrainData else 1 # we want to do until convergence for dummy test set
lossgraph = []
outputarray = []
outputarrayarcs = []
outputarraylabels = []

start = datetime.datetime.now()

for epoch in range(epochs):        
    shuffle(sentencesDependencies)
    total_output = 0
    for s in sentencesDependencies:
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Clear hidden and cell previous state
        model.hiddenState, model.cellState = model.initHiddenCellState()

        sentenceInWords, sentenceInTags = s.getSentenceInWordsAndInTags()
    
        wordsToIndices = [w2i[w] for w in sentenceInWords]
        words_tensor = torch.LongTensor(wordsToIndices)
        
        tagsToIndices = [t2i[t] for t in sentenceInTags]
        tags_tensor = torch.LongTensor(tagsToIndices)
        
        arcs_refdata = s.getHeadsForWords()
        
        # Forward pass
        scoreTensor, labelTensor = model(words_tensor, tags_tensor, arcs_refdata)

        # Get reference data (gold) for arcs
        arcs_refdata = torch.from_numpy(arcs_refdata)
        arcs_refdata = arcs_refdata.long()

        # also need to use the gold data for labels here:
        labels_refdata = s.getLabelsForWords(l2i)
        labels_refdata = torch.from_numpy(labels_refdata)
        labels_refdata = labels_refdata.long()
        
        #get sentence length
        sentence_length = len(s.tokens)
        
        # Calculate loss
        loss = nn.CrossEntropyLoss()
        
        # For the arcs classification
        modelinput_arcs = scoreTensor.transpose(0,1)
        target_arcs = Variable(arcs_refdata)
        loss_arcs = loss(modelinput_arcs, target_arcs)
        
        # For the label classification
        modelinput_labels = labelTensor
        target_labels = Variable(labels_refdata)
        loss_labels = loss(modelinput_labels, target_labels)
        
        output = loss_arcs + loss_labels
        
        if useDummyTrainData: # stop if converging            
            if -1e-3 < output < 1e-3: # almost zero
                print('Number of epochs till convergence: {}'.format(epoch))
                break
            if epoch % 100 == 0:
                print('Epochs so far without convergence: {}'.format(epoch))
                
        output.backward()
        optimizer.step()
        
        outputarray.append(output.data[0])
        outputarrayarcs.append(loss_arcs.data[0])
        outputarraylabels.append(loss_labels.data[0])
        total_output += output.data[0]

        
    lossgraph.append(total_output)

print(outputarray)
print(lossgraph)

print('Training time: {}'.format(datetime.datetime.now() - start))

date = str(time.strftime("%d_%m"))
savename = "DependencyParserModel_" + date + ".pkl"
imagename = "DependencyParserModel_" + date + ".jpg"

torch.save(model, savename)

fig, axes = plt.subplots(2,2)
axes[0, 0].plot(lossgraph)
axes[0, 1].plot(outputarray)
axes[1, 0].plot(outputarrayarcs)
axes[1, 1].plot(outputarraylabels)
axes[0, 0].set_title('Loss per epoch')
axes[0, 1].set_title('Loss per sentence')
axes[1, 0].set_title('Loss arcs MLP')
axes[1, 1].set_title('Loss label MLP')
fig.subplots_adjust(hspace=0.5)
fig.subplots_adjust(wspace=0.5)
plt.savefig(imagename)
