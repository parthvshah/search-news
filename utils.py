import pandas as pd
import pickle
from spellchecker import SpellChecker

spell = SpellChecker()


def spellchecker(query):
    words = query.split()
    correction = []
    for word in words:
        correction.append(spell.correction(word))

    return " ".join(correction)


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
