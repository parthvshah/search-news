import pandas as pd
import pickle
import numpy as np
import spacy as sp

from nltk.stem import PorterStemmer
from num2words import num2words
from spellchecker import SpellChecker

spell = SpellChecker()
nlp = sp.load("en_core_web_sm")


def levenshteinDistance(s1, s2):
    """
    Calculates the Levenshtein Distance between two strings using DP
    """
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(
                    1 + min((distances[i1], distances[i1 + 1], distances_[-1]))
                )
        distances = distances_
    return distances[-1]


def spellchecker(query):
    """
    Returns the most probable corrected string
    """
    words = query.split()
    correction = []
    for word in words:
        correction.append(spell.correction(word))

    return " ".join(correction)


def preprocess(snippet):
    """
    Perform following cleaning on `snippet`:
        - Lower case
        - punctuation
        - Lemmatization
        - Stop word removal
        - num2word
    Returns modified snippet
    """
    stopwords = nlp.Defaults.stop_words
    stemmer = PorterStemmer()
    # lowercase
    snippet = snippet.lower()
    # punct
    symbols = '!"#$%&()*+-./:;<=>?@[\]^_`{|}~\n'
    for i in range(len(symbols)):
        snippet = np.char.replace(snippet, symbols[i], " ")
        snippet = np.char.replace(snippet, "  ", " ")
    snippet = np.char.replace(snippet, ",", "")

    # remove stopwords; stem
    # TODO: try with a lemmetizer
    snippet = str(snippet).split()
    cleanTokens = [stemmer.stem(word) for word in snippet if word not in stopwords]

    for index in range(len(cleanTokens)):
        try:
            newToken = num2words(cleanTokens[index])
            cleanTokens[index] = newToken
        except:
            continue

    cleanSnippet = " ".join(cleanTokens)
    return cleanSnippet


def retrieveFile(filePath):
    result = []
    df = pd.read_csv(filePath, index_col=None, header=0)
    # URL,MatchDateTime,Station,IAPreviewThumb,Snippet
    for index, row in df.iterrows():
        result.append(
            [
                row["URL"],
                row["MatchDateTime"],
                row["Station"],
                row["IAPreviewThumb"],
                row["Snippet"],
            ]
        )

    return result


def dump(obj, filename):
    """
    Pickle wrapper to dump
    """
    with open(filename, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(filename):
    """
    Pickle wrapper to load
    """
    with open(filename, "rb") as handle:
        obj = pickle.load(handle)

    return obj


def inspect(filename):
    """
    Pickle wrapper to inspect stored values
    """
    with open(filename, "rb") as handle:
        obj = pickle.load(handle)

    print(obj)
