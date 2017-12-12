# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sentenceDependencies import ConlluFileReader,SentenceDependencies #, ConlluFileWriter
from dataProcessor import DataProcessor
from wordEmbeddingsReader import GloVeFileReader
from gensim.models import Word2Vec
import numpy as np
from model import DependencyParseModel

unknownMarker = '<unk>'

sentencesReader = ConlluFileReader(r"UD_English/en-ud-train.conllu")
sentencesDependencies = sentencesReader.readSentencesDependencies(unknownMarker)

dataProcessor = DataProcessor(sentencesDependencies)
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


model = DependencyParseModel(word_embeddings_dim, posTags_embeddings_dim, vocabularySize, tagsUniqueCount, pretrainedWordEmbeddings, pretrainedTagEmbeddings)

for s in sentencesDependencies:
    # Clear hidden and cell previous state
    model.hiddenState, model.cellState = model.initHiddenCellState()

    # Forward pass
    result,refdata = model(s, w2i, t2i)
#    print(result) # result so far is scores matrix
    
    #get sentence length
    sentenceInWords, sentenceInTags = s.getSentenceInWordsAndInTags()
    sentence_length = len(sentenceInWords)
       
    # Calculate loss
    output = 0
    for column in range(0,sentence_length):
        loss = nn.BCELoss()
        input = result[:,column] 
        target =  Variable(refdata[:,column])
        output += loss(input,target)
    
    output.backward()
    
    break # just for testing purposes. Remove when doing the actual training


#writer = ConlluFileWriter('testFile.conllu')
#writer.write(trainingSet)