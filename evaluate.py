#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:30:31 2017

@author: diana
"""

from conlluFilesOperations import ConlluFileReader, ConlluFileWriter
from sentenceDependencies import createSentenceDependencies
from dataProcessor import buildDictionaries
import torch
from mst import mst
import numpy as np
from torch.autograd import Variable

useEnglish = False # change to false for Dutch

modelId = "{}17_12".format("" if useEnglish else "Dutch_")  # change this each run
filename = "results/DependencyParserModel_{}.pkl".format(modelId)
model = torch.load(filename)

testFileName = r"UD_English/en-ud-test.conllu" if useEnglish else r"UD_Dutch/nl-ud-test.conllu"
testSentencesReader = ConlluFileReader(testFileName)
testSentences = testSentencesReader.readSentencesDependencies('<root>')

trainFileName = r"UD_English/en-ud-train.conllu" if useEnglish else r"UD_Dutch/nl-ud-train.conllu"

 # These are needed for sentence prep
trainSentencesReader = ConlluFileReader(trainFileName)
trainingSet = trainSentencesReader.getSentenceDependenciesUnknownMarker('<unk>')


######################### Remove this when done testing

#trainSentencesReader = ConlluFileReader(r"UD_English/en-ud-train.conllu")
#trainingSet = trainSentencesReader.readSentencesDependencies('<root>')
#
### Just for dummy data testing 
#nSentencesToUse = 5
#nSentencesSoFar = 0
#newSentencesDependencies = []
#i = 0
#while(nSentencesSoFar < nSentencesToUse):
#    if(7<= len(trainingSet[i].tokens) <= 9):
#        newSentencesDependencies.append(trainingSet[i])
#        nSentencesSoFar += 1
#    i += 1
#
#trainingSet = newSentencesDependencies

########################


w2i, t2i, _, _, _, i2l = buildDictionaries(trainingSet, '<unk>')

sentencesDepsPredictions = []

for s in testSentences:
    # Input prep
    sentenceInWords, sentenceInTags = s.getSentenceInWordsAndInTags() # Getting tokens and tags

    wordsToIndices = [w2i[w] for w in sentenceInWords]
    words_tensor = torch.LongTensor(wordsToIndices)
    
    tagsToIndices = [t2i[t] for t in sentenceInTags]
    tags_tensor = torch.LongTensor(tagsToIndices)
    
    scoreMatrix = model.predictArcs(Variable(words_tensor), Variable(tags_tensor))    
    headsForWords = mst(scoreMatrix.data.numpy())    
    
    labelsMatrix = model.predictLabels(torch.LongTensor(headsForWords))
    labelsForWords = np.argmax(labelsMatrix.data.numpy(), axis=1)
    
    sentencesDepsPredictions.append(createSentenceDependencies(sentenceInWords, sentenceInTags, headsForWords, [i2l[l] for l in labelsForWords]))

writer = ConlluFileWriter('output/predictions_{}.conllu'.format(modelId))
writer.write(sentencesDepsPredictions)