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
import datetime

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

#print(len(list(parameters))) #do not ever do this

optimizer = torch.optim.Adam(parameters, lr=0.01, weight_decay=1E-6)

#for p in model.parameters():
#    if p.requires_grad:
#        print(p.size())

#for name,param in model.named_parameters():
#    if param.requires_grad:
#        print(name,param.size())
        
#print(l2i)
epochs = 60 if useDummyTrainData else 1 # we want to do until convergence for dummy test set

#start = datetime.datetime.now()

for epoch in range(epochs):        
#    shuffle(trainingSet)
    total_output = 0
    for sentenceIndex, s in enumerate(trainingSet):
        
        # First plot gold info
        #************************************************************************
        # G the gold 0/1-matrix as numpy array (pyplot cannot plot torch Tensors)
        # S is the score torch Tensor: the output of your model
        # A is the softmax version of S, also a torch Tensor! (actually more acurately it's a Variable(Tensor(..))
        #************************************************************************
        
#        G = np.zeros((len(s.tokens), len(s.tokens)))
#        for i,h in enumerate(s.getHeadsForWords()):            
#            G[int(h), i] = 1
#        
##        # for each sentence i you do:
#        plt.clf() # clear the plotting canvas just to be sure
#        plt.imshow(G) # draw the heatmap
#        plt.savefig("gold-sent-{}.png".format(sentenceIndex)) # save and give it a name: gold-sent-1.pdf for example
#        
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
#        print(arcs_refdata_tensor)
        
        scoreTensor, labelTensor = model(words_tensor, tags_tensor, arcs_refdata_tensor)        

#        # also need to use the gold data for labels here:
        labels_refdata = s.getLabelsForWords(l2i)
        labels_refdata = torch.from_numpy(labels_refdata).long()

        
        #get sentence length
        sentence_length = len(s.tokens)
        
#        firstword = sentenceInWords[1]
        
        # Calculate loss
        loss = nn.CrossEntropyLoss()
#        loss = nn.NLLLoss()
        
        # For the arcs classification
        modelinput_arcs = (scoreTensor)
#        modelinput_arcs = Variable(torch.Tensor([[1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,1,0,0,0,0]]),requires_grad=True)
#        print(s)
        target_arcs = Variable(arcs_refdata_tensor, requires_grad=False)
#        print(target_arcs)
        loss_arcs = loss(modelinput_arcs, target_arcs)
#        print(loss_arcs.data[0])
        
        # For the label classification
        
        modelinput_labels = labelTensor
        target_labels = Variable(labels_refdata)
        loss_labels = loss(modelinput_labels, target_labels)
#        print(s, target_labels)
        
        output = loss_arcs #+ loss_labels
        print(output.data[0]) 
         
        output.backward()
        optimizer.step()
#        print(loss_arcs.data[0])
                
        # Then during training, for each epoch step and for each i you do:
        m = nn.Softmax()
#        A = torch.t(scoreTensor)
        A = m(scoreTensor)
#        A = scoreTensor
        A = torch.t(A)

        
        plt.clf()
        numpy_A = A.data.numpy() # get the data in Variable, and then the torch Tensor as numpy array
#        print(numpy_A)
        plt.imshow(numpy_A)
        plt.savefig("pred-sent-{}-epoch-{}".format(sentenceIndex, epoch))
        
#        print(scoreTensor)
#        print(loss_arcs.data[0])
#        break
        
print("last loss", output)

#for name, param in model.named_parameters():
#    if param.requires_grad:
#        print(param)
#        break


#print('Training time: {}'.format(datetime.datetime.now() - start))
#
#date = str(time.strftime("%d_%m"))
#savename = "DependencyParserModel_" + date + ".pkl"
#imagename = "DependencyParserModel_" + date + ".jpg"
#
#torch.save(model, savename)



#fig, axes = plt.subplots(2,2)
#axes[0, 0].plot(lossgraph)
#axes[0, 1].plot(outputarray)
#axes[1, 0].plot(outputarrayarcs)
#axes[1, 1].plot(outputarraylabels)
#axes[0, 0].set_title('Loss per epoch')
#axes[0, 1].set_title('Loss per sentence')
#axes[1, 0].set_title('Loss arcs MLP')
#axes[1, 1].set_title('Loss label MLP')
#fig.subplots_adjust(hspace=0.5)
#fig.subplots_adjust(wspace=0.5)
#plt.savefig(imagename)
