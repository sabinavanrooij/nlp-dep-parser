#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:49:31 2017

@author: diana
"""
import numpy as np

# creates an object from headsArr from MST and labels list from model
def createSentenceDependencies(sentenceInWords, sentenceInTags, headsArr, labels):
    assert len(sentenceInWords) == headsArr.size == len(labels), "Length mismatch: {0} words, {1} heads, {2} labels".format(len(sentenceInWords), headsArr.size, len(labels))
    
    sentenceDep = SentenceDependencies()
    for i, w in enumerate(sentenceInWords):
        # Do not print root
        if i == 0:
            continue
        
        sentenceDep.addToken(Token(index=i, word=w, POSTag=sentenceInTags[i], head=headsArr[i], label=labels[i]))
        
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
    
    def getHeadsForWords(self):
        # numpy arr where value i is the head for word i
        sentenceLength = len(self.tokens)
        arr = np.zeros(sentenceLength, dtype=int) # account for the root, first element is 0
        for k, v in self.tokens.items():
            arr[k] = v.head
        
        return arr
    
    def getLabelsForWords(self, l2i):
        # numpy arr where value i is the label index for word i
        sentenceLength = len(self.tokens)
        arr = np.zeros(sentenceLength)
        for k, v in self.tokens.items():
            arr[k] = l2i[v.label]
        
        return arr
