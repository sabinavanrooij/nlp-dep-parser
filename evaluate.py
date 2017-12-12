#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:30:31 2017

@author: diana
"""

from model import DependencyParseModel
import torch

savename = "DependencyParserModel.pkl"

word_embeddings_dim = 50
posTags_embeddings_dim = 50

# To use the network:
finalmodel = DependencyParseModel(word_embeddings_dim, posTags_embeddings_dim, -1, -1)
finalmodel.load_state_dict(torch.load(savename))


# the sentence you want to evaluate, NOTE: sentenceToEvaluate not defined yet!
evaluation = finalmodel(sentenceToEvaluate)

#writer = ConlluFileWriter('testFile.conllu')
#writer.write(trainingSet)