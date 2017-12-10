#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:28:08 2017

@author: diana
"""

from sentenceDependencies import ConlluFileReader
from collections import defaultdict

class DataProcessor:
    
    def __init__(self, fileName):
        # Read sentences dependencies from file
        sentencesReader = ConlluFileReader(fileName)
        self.sentencesDeps = sentencesReader.readSentencesDependencies()
        self.unknownMarker = '<unk>'

        # Replace words with count = 1 with <unk>
        wordCounts = sentencesReader.wordCounts
        for s in self.sentencesDeps:
            for k,v in s.tokens.items():
                if wordCounts[v.word] == 1:
                    v.word = self.unknownMarker
                    
        print(self.sentencesDeps[0])

    def buildDictionaries(self):        
        w2i = defaultdict(lambda: len(w2i))
        t2i = defaultdict(lambda: len(t2i))
        l2i = defaultdict(lambda: len(l2i))
        i2w = dict()
        i2t = dict()
        i2l = dict()
        
        for s in self.sentencesDeps:
            for k,v in s.tokens.items():
                i2w[w2i[v.word]] = v.word
                i2t[t2i[v.POSTag]] = v.POSTag
                i2l[l2i[v.label]] = v.label
                
        w2i = defaultdict(lambda: w2i[self.unknownMarker], w2i)
        
        return w2i, t2i, l2i, i2w, i2t, i2l
    
    def getTrainingSetsForWord2Vec(self):        
        wordsTrainingSet = []
        posTagsTrainingSet = []
        
        for s in self.sentencesDeps:
            sentenceInWords = []
            sentenceInTags = []
            for k,v in s.tokens.items():
                sentenceInWords.append(v.word)
                sentenceInTags.append(v.POSTag)
            wordsTrainingSet.append(sentenceInWords)
            posTagsTrainingSet.append(sentenceInTags)
            
        return wordsTrainingSet, posTagsTrainingSet
