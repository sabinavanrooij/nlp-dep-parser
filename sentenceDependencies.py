#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:49:31 2017

@author: diana
"""
import numpy as np

# creates an object from headsArr from MST and labelsArr from model (?)
def createSentenceDependencies(sentenceInWords, headsArr, labelsArr, i2l):
    assert len(sentenceInWords) == headsArr.size == labelsArr.size, "Length mismatch: {0} words, {1} heads, {2} labels".format(len(sentenceInWords), headsArr.size, labelsArr.size)
    
    sentenceDep = SentenceDependencies()
    for i, w in enumerate(sentenceInWords):
        sentenceDep.addToken(Token(index=i+1, word=w, POSTag='_', head=headsArr[i], label=i2l[labelsArr[i]]))
        
    return sentenceDep
        
class Token:
    def __init__(self, index, word, POSTag, head, label):
        self.index = index
        self.word = word
        self.POSTag = POSTag
        self.head = head
        self.label = label
    
    def __str__(self):        
        return "Index: {0}, Word: \"{1}\", POSTag: {2}, Head: {3}, Label: {4}".format(self.index, self.word, self.POSTag, self.head, self.label)

class SentenceDependencies:
    def __init__(self):
        self.tokens = {}
        self.sentenceInWords = []
        self.sentenceInTags = []        
    
    def addToken(self, token):
        self.tokens[token.index] = token
    
    def __str__(self):
        strList = []
        for k in self.tokens:
            strList.append(str(self.tokens[k]))
        return '\n'.join(strList)
    
    def getSentenceInWordsAndInTags(self):
        if len(self.sentenceInWords) > 0:
            assert len(self.sentenceInTags) > 0
            return self.sentenceInWords, self.sentenceInTags
        
        assert len(self.tokens) > 0
        
        for k,v in self.tokens.items():
            self.sentenceInWords.append(v.word)
            self.sentenceInTags.append(v.POSTag)
        return self.sentenceInWords, self.sentenceInTags
            
#    def getAdjacencyMatrix(self):
#        # Rows are heads, columns are dependents
#        mSize = len(self.tokens) + 1 # account for root
#        m = np.zeros((mSize, mSize))
#        
#        m[0][0] = 1 # Root goes to root
#        
#        for k,v in self.tokens.items():
#            m[v.head][v.index] = 1
#        
#        return m
    
    def getHeadsForWords(self):
        # numpy arr where value i is the head for word i
        sentenceLength = len(self.tokens)
        arr = np.zeros(sentenceLength + 1) # account for the root, first element is 0
        for k, v in self.tokens.items():
            arr[k] = v.head
        
        return arr
    
    def getLabelsForWords(self, l2i):
        # numpy arr where value i is the label index for word i
        sentenceLength = len(self.tokens)
        arr = np.zeros(sentenceLength)
        for k, v in self.tokens.items():
            arr[k-1] = l2i[v.label]
        
        return arr
