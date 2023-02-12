import spacy
import numpy as np
from spacy import Language

#@Language.component("normalization")
def normalization(doc):
  word_list = []
  for token in doc:
    if not (token.is_punct or token.is_space or (not token.has_vector)):
      word_list.append(token.lemma_.lower())

  spaces = np.ones(len(word_list))
  if len(word_list) > 1:
    spaces[-1] = 0 

  return spacy.tokens.Doc(doc.vocab, word_list, spaces)

def clean_pipe(docs, *args, **kwargs):
    for doc in docs:
      yield doc

normalization.pipe = clean_pipe

@Language.factory('normalizer-factory')
def normalizer_factory(nlp, name):
  return normalization