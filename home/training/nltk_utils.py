import nltk
import numpy as np
# nltk.download('punkt') # take out the comment for the first time execution
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence): # separate a sentence into meaningful words
    return nltk.word_tokenize(sentence)

def stem(word): # reduce words to its roots # organize, organization, organizing -> organize
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words): # counts occurrences of words
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx,w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag