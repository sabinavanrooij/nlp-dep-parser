#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:30:31 2017

@author: diana
"""

from conlluFilesOperations import ConlluFileReader, ConlluFileWriter
from sentenceDependencies import createSentenceDependencies
from dataProcessor import DataProcessor
import torch
from mst import mst
import numpy as np

filename = "DependencyParserModel_13_12.pkl" # change this each run
model = torch.load(filename)

testSentencesReader = ConlluFileReader(r"UD_English/en-ud-test.conllu")
testSentences = testSentencesReader.readSentencesDependencies()

# These are needed for sentence prep
trainSentencesReader = ConlluFileReader(r"UD_English/en-ud-train.conllu")
dataProcessor = DataProcessor(trainSentencesReader.readSentencesDependencies('<unk>'), '<unk>')
w2i, t2i, _, _, _, i2l = dataProcessor.buildDictionaries()

sentencesDepsPredictions = []

for s in testSentences:
    
    # Input prep
    sentenceInWords, sentenceInTags = s.getSentenceInWordsAndInTags() # Getting tokens and tags
    
    wordsToIndices = [w2i[w] for w in sentenceInWords]
    words_tensor = torch.LongTensor(wordsToIndices)
    
    tagsToIndices = [t2i[t] for t in sentenceInTags]
    tags_tensor = torch.LongTensor(tagsToIndices)
   
    scoreMatrix = model.predictArcs(words_tensor, tags_tensor)
    headsForWords = mst(scoreMatrix.numpy())
  
    labelsMatrix = model.predictLabels(headsForWords) 
    labelsForWords = np.argmax(labelsMatrix.numpy(), axis=1)
    
    sentencesDepsPredictions.append(createSentenceDependencies(sentenceInWords, sentenceInTags, headsForWords[1:], [i2l[l] for l in labelsForWords]))


writer = ConlluFileWriter('output/predictions.conllu')
writer.write(sentencesDepsPredictions)