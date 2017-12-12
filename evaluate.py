#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:30:31 2017

@author: diana
"""

from model import DependencyParseModel
from conlluFilesOperations import ConlluFileReader, ConlluFileWriter
from sentenceDependencies import createSentenceDependencies
from dataProcessor import DataProcessor
import torch
from mst import mst
import numpy as np

filename = "DependencyParserModel_12_12.pkl" # change this each run

word_embeddings_dim = 50
posTags_embeddings_dim = 50

# To use the network:
#finalmodel = DependencyParseModel(word_embeddings_dim, posTags_embeddings_dim, -1, -1)
finalmodel = torch.load(filename)
#finalmodel.load_state_dict(torch.load(filename))

testSentencesReader = ConlluFileReader(r"UD_English/en-ud-test.conllu")
testSentences = testSentencesReader.readSentencesDependencies()

trainSentencesReader = ConlluFileReader(r"UD_English/en-ud-train.conllu")
dataProcessor = DataProcessor(trainSentencesReader.readSentencesDependencies())
_, _, _, _, _, i2l = dataProcessor.buildDictionaries()

sentencesDeps = []

for s in testSentences:
    prediction = finalmodel(s)
    scoreMatrix, labelsMatrix = finalmodel() # what the hell goes here?
    
    headsForWords = mst(scoreMatrix)
    labelsForWords = np.argmax(labelsMatrix, axis=1)
    
    sentenceInWords, _ = s.getSentenceInWordsAndInTags()
    sentencesDeps.append(createSentenceDependencies(sentenceInWords, headsForWords[1:], labelsForWords, i2l))

writer = ConlluFileWriter('output/predictions.conllu')
writer.write(sentencesDeps)