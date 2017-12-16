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

filename = "DependencyParserModel_15_12.pkl" # change this each run
model = torch.load(filename)

testSentencesReader = ConlluFileReader(r"UD_English/en-ud-test.conllu")
testSentences = testSentencesReader.readSentencesDependencies()

# These are needed for sentence prep
trainSentencesReader = ConlluFileReader(r"UD_English/en-ud-train.conllu")
trainingSet = trainSentencesReader.getSentenceDependenciesUnknownMarker('<unk>')
w2i, t2i, _, _, _, i2l = buildDictionaries(trainingSet, '<unk>')

sentencesDepsPredictions = []

for i, s in enumerate(testSentences[91:]):
    print(i)
    print(s)
    # Input prep
    sentenceInWords, sentenceInTags = s.getSentenceInWordsAndInTags() # Getting tokens and tags
    
    wordsToIndices = [w2i[w] for w in sentenceInWords]
    words_tensor = torch.LongTensor(wordsToIndices)
    
    tagsToIndices = [t2i[t] for t in sentenceInTags]
    tags_tensor = torch.LongTensor(tagsToIndices)
   
    scoreMatrix = model.predictArcs(Variable(words_tensor), Variable(tags_tensor))
    
    if scoreMatrix.size() == (1,1) and scoreMatrix.data[0, 0] == 0:
        headsForWords = 0
        print('huh')
    else:
        headsForWords = mst(scoreMatrix.data.numpy().T)    
    
    labelsMatrix = model.predictLabels(torch.LongTensor(headsForWords))
    labelsForWords = np.argmax(labelsMatrix.data.numpy(), axis=1)
    
    sentencesDepsPredictions.append(createSentenceDependencies(sentenceInWords, sentenceInTags, headsForWords, [i2l[l] for l in labelsForWords]))
    break

writer = ConlluFileWriter('output/predictions.conllu')
writer.write(sentencesDepsPredictions)