# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
from conlluFilesOperations import ConlluFileReader
from dataProcessor import buildDictionaries,getTrainingSetsForWord2Vec
from wordEmbeddingsReader import GloVeFileReader
from gensim.models import Word2Vec
import numpy as np
from model import DependencyParseModel
import torch.nn as nn
from torch.autograd import Variable
from random import shuffle
import time
import matplotlib.pyplot as plt

#### This is important. Remove when done double checking all the code

useDummyTrainData = True

####


unknownMarker = '<unk>'
rootMarker = '<root>'

sentencesReader = ConlluFileReader(r"UD_English/en-ud-train.conllu")
word2VecInputs = sentencesReader.readSentencesDependencies(rootMarker)
trainingSet = sentencesReader.getSentenceDependenciesUnknownMarker(unknownMarker)

# Select 5 short sentences from the train file (of sentence length at most 10, say)
if useDummyTrainData:
    nSentencesToUse = 5
    nSentencesSoFar = 0
    newSentencesDependencies = []
    i = 0
    while(nSentencesSoFar < nSentencesToUse):
        if(7<= len(word2VecInputs[i].tokens) <= 9):
            newSentencesDependencies.append(word2VecInputs[i])
            nSentencesSoFar += 1
        i += 1
    
    word2VecInputs = newSentencesDependencies
    trainingSet = newSentencesDependencies


w2i, t2i, l2i, i2w, i2t, i2l = buildDictionaries(trainingSet, unknownMarker)
sentencesInWords, sentencesInTags = getTrainingSetsForWord2Vec(word2VecInputs)

word_embeddings_dim = 50
posTags_embeddings_dim = 50
minCountWord2Vec_words = 1 if useDummyTrainData else 5
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

# Words that do not have a corresponding embedding (aka. unknown and root) will start with a random vector
pretrainedWordEmbeddings = np.random.rand(vocabularySize, word_embeddings_dim)
for k,v in i2w.items():
    if v in wordsEmbeddings:
        pretrainedWordEmbeddings[k] = wordsEmbeddings[v]
    elif v in wordsModel.wv.vocab:
        pretrainedWordEmbeddings[k] = wordsModel[v]

# Tags that do not have a corresponding embedding (aka root) will start with a random vector
pretrainedTagEmbeddings = np.random.rand(tagsUniqueCount, posTags_embeddings_dim)
for k,v in i2t.items():
    if v in POSTagsModel:
        pretrainedTagEmbeddings[k] = POSTagsModel[v]
        
model = DependencyParseModel(word_embeddings_dim, posTags_embeddings_dim, vocabularySize, tagsUniqueCount, labelsUniqueCount, pretrainedWordEmbeddings, pretrainedTagEmbeddings)
parameters = filter(lambda p: p.requires_grad, model.parameters())

optimizer = torch.optim.Adam(parameters, lr=0.01, weight_decay=1E-6)

epochs = 70 if useDummyTrainData else 1 # we want to do until convergence for dummy test set

for epoch in range(epochs):        
#    shuffle(trainingSet)
    total_output = 0
    for sentenceIndex, s in enumerate(trainingSet):
        
        # Plot gold info
        if epoch == 0:
            G = np.zeros((len(s.tokens), len(s.tokens)))
            for i,h in enumerate(s.getHeadsForWords()):            
                G[int(h), i] = 1
                        
            plt.clf() # clear the plotting canvas just to be sure
            plt.imshow(G) # draw the heatmap
            plt.savefig("gold-sent-{}.png".format(sentenceIndex)) # save and give it a name: gold-sent-1.pdf for example
            
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Clear hidden and cell previous state
        model.hiddenState, model.cellState = model.initHiddenCellState()

        sentenceInWords, sentenceInTags = s.getSentenceInWordsAndInTags()
    
        wordsToIndices = [w2i[w] for w in sentenceInWords]
        words_tensor = Variable(torch.LongTensor(wordsToIndices),requires_grad=False)
        
        tagsToIndices = [t2i[t] for t in sentenceInTags]
        tags_tensor = Variable(torch.LongTensor(tagsToIndices),requires_grad=False)
        
        # Get reference data (gold) for arcs
        arcs_refdata_tensor = torch.LongTensor(s.getHeadsForWords())
        
        scoreTensor, labelTensor = model(words_tensor, tags_tensor, arcs_refdata_tensor)        

        # also need to use the gold data for labels here:
        labels_refdata = s.getLabelsForWords(l2i)
        labels_refdata = torch.from_numpy(labels_refdata).long()
        
        #get sentence length
        sentence_length = len(s.tokens)
                
        # Calculate loss
        loss = nn.CrossEntropyLoss()
        
        # For the arcs classification
        modelinput_arcs = (scoreTensor)
        target_arcs = Variable(arcs_refdata_tensor, requires_grad=False)
        loss_arcs = loss(modelinput_arcs, target_arcs)
        
        # For the label classification
        modelinput_labels = labelTensor
        target_labels = Variable(labels_refdata)
        loss_labels = loss(modelinput_labels, target_labels)
        
        output = loss_arcs + loss_labels
        print(output.data[0]) 
         
        output.backward()
        optimizer.step()
                
        # Plot prediction        
        m = nn.Softmax(dim=1)
        A = m(scoreTensor)
        A = torch.t(A)
    
        plt.clf()
        numpy_A = A.data.numpy() # get the data in Variable, and then the torch Tensor as numpy array
        plt.imshow(numpy_A)
        plt.savefig("pred-sent-{}-epoch-{}".format(sentenceIndex, epoch))
            

# Save the model on disk        
date = str(time.strftime("%d_%m_%H_%M"))
savename = "DependencyParserModel_" + date + ".pkl"

torch.save(model, savename)

# Print to check it the label prediction is working
for i in range(labelTensor.size()[0]):
    predictedlabel = labelTensor[i,:]