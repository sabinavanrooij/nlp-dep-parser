# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#from sentenceDependencies import ConlluFileWriter
from dataProcessor import DataProcessor
from wordEmbeddingsReader import GloVeFileReader
from gensim.models import Word2Vec
import numpy as np
#import torch
#from torch.autograd import Variable
from model import DependencyParseModel


dataProcessor = DataProcessor(r"UD_English/en-ud-train.conllu")
w2i, t2i, l2i, i2w, i2t, i2l = dataProcessor.buildDictionaries()

sentencesInWords, sentencesInTags = dataProcessor.getTrainingSetsForWord2Vec()

word_embeddings_dim = 50
posTags_embeddings_dim = 100
minCountWord2Vec_words = 5
minCountWord2Vec_tags = 0

# Train the POS tags
POSTagsModel = Word2Vec(sentencesInTags, size=posTags_embeddings_dim, window=5, min_count=minCountWord2Vec_tags, workers=4)

# Read the word embeddings
wordEmbeddingsReader = GloVeFileReader(r"GloVe/glove.6B.50d.txt")
wordsEmbeddings = wordEmbeddingsReader.readWordEmbeddings()

# Or train the embeddings too
wordsModel = Word2Vec(sentencesInWords, size=word_embeddings_dim, window=5, min_count=minCountWord2Vec_words, workers=4)


#        vectors.append(np.concatenate((wordVector, POSTagsModel[v.POSTag])))


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
        pretrainedWordEmbeddings[k] = wordsModel[dataProcessor.unknownMarker]

pretrainedTagEmbeddings = np.empty((tagsUniqueCount, posTags_embeddings_dim))
for k,v in i2t.items():
    assert v in POSTagsModel.wv.vocab
    pretrainedTagEmbeddings[k] = POSTagsModel[v]

model = DependencyParseModel(word_embeddings_dim, posTags_embeddings_dim, vocabularySize, tagsUniqueCount, pretrainedWordEmbeddings, pretrainedTagEmbeddings)
    

#nDirections = 2
#batch = 1 # this is per recommendation
#seq_len = len(sentence) # number of words


# apparently this is the actual input
#inputs = Variable(torch.randn(seq_len, batch, inputSize))
#print(inputs.size())

#tensor = torch.from_numpy(np.array(sentenceVectors))
#print(tensor.size())
#print(sentenceVectors[0])

#initialHiddenState = Variable(torch.randn(nLayers * nDirections, batch, hiddenSize))
#initialCellState = Variable(torch.randn(nLayers * nDirections, batch, hiddenSize))
#output, hidden = biLstm(inputs, (initialHiddenState, initialCellState))

#print(output)
#print(hidden)


#writer = ConlluFileWriter('testFile.conllu')
#writer.write(trainingSet)