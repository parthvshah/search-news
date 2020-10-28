import pandas as pd
import pickle


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
