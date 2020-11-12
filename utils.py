import pandas as pd
import pickle
import spacy as sp

from num2words import num2words
from spellchecker import SpellChecker

spell = SpellChecker()
nlp = sp.load("en_core_web_sm")


def spellchecker(query):
    words = query.split()
    correction = []
    for word in words:
        correction.append(spell.correction(word))

    return " ".join(correction)


def preprocess(snippet):
    """
    Perform following cleaning on `snippet`:
        - Lower case
        - Lemmatization
        - Stop word removal
        - num2word
    Returns modified snippet
    """
    stopwords = nlp.Defaults.stop_words
    # lowercase
    snippet = snippet.lower()

    # remove stopwords; lemmatise
    snippet = nlp(snippet)
    cleanTokens = [word.lemma_ for word in snippet if word not in stopwords]

    for index in range(len(cleanTokens)):
        try:
            newToken = num2words(cleanTokens[index])
            cleanTokens[index] = newToken
        except:
            continue

    cleanSnippet = " ".join(cleanTokens)
    return cleanSnippet


def retrieveSnippetsFromFile(filePath):
    df = pd.read_csv(filePath, index_col=None, header=0)
    return df.Snippet.str.lower().values


def dump(obj, filename):
    with open(filename, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(filename):
    with open(filename, "rb") as handle:
        obj = pickle.load(handle)

    return obj
