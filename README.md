# nlp-dep-parser
NLP project on dependency parsers

<b>FINAL REPORT Template</b>: https://www.overleaf.com/12707057qsjgmmyjnswh

Requirements for the report: https://github.com/tdeoskar/NLP1-2017/blob/master/project-reqs.md


Milestone:

<b>Dependency-data</b>

Reading in and writing out text from the CONLL-U file type. <i>Done</i>

Replace all the words in your training file that occur just once with the word <unk>. <i>Done</i>

w2i, t2i, and l2i dicts. And inverse: i2w, i2t, i2l <i>Done</i>

Remove lines from .conllu file that have non integers as indices <i>Done</i>

<b>MST</b>

<i>In progress</i>

<b>LSTM</b>

Embedding layer for words <i>Done</i>

Support for optional pretrained word embeddings <i>Done</i>

Embedding layer for POS tags <i>Done</i>

Support for optional pretrained tag embeddings <i>Done</i>

Concatenate these word embeddings <i>Done</i>

LSTM layer <i>Done</i>


<b> TO DO: </b>

- Implementation of MLP2 in pytorch (label classification)

- Change cross entropy functions 

- Translation from MST to CoNLL file

- Gold thingy for the labels



- Handle multiple cycles in the MST algo
