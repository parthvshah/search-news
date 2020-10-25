import glob
import pandas as pd
import pickle_utils as pu
import spacy as sp
import time


nlp = sp.load("en_core_web_sm")


class invertedIndexDict:
    """
    Dictionary implementation of inverted index.
    """

    def __init__(self, dataDirectory):
        """
        `dataDirectory` is path to directory of corpus.
        """

        self.dataDir = dataDirectory

        # index is the inverted index
        # each term in the inverted index maps to a posting list
        # posting list is positional, i.e, it is a dictionary that maps docID to set of token positions
        self.index = {}

        # documentIndex contains the mapping of docIDs to docNames.
        self.documentIndex = {}

        self.constructIndex()

    def constructIndex(self):
        """
        Construct index from documents.
        """

        try:
            self.index = pu.load(r"./obj/invIdxDict.pk")
            self.documentIndex = pu.load(r"./obj/docIdxDict.pk")

        except (OSError, IOError):
            documents = glob.glob(self.dataDir + "/*.csv")
            docCount = 0
            for document in documents:
                docCount += 1
                self.documentIndex[docCount] = document

                df = pd.read_csv(document, index_col=None, header=0)
                a = str(df["Snippet"]).lower()
                spSnip = nlp(a)

                for token in spSnip:
                    tokLemma = token.lemma_
                    if not tokLemma in self.index:
                        self.index[tokLemma] = {}
                    if not docCount in self.index[tokLemma]:
                        self.index[tokLemma][docCount] = set()
                    self.index[tokLemma][docCount].add(token.i)

            pu.dump(self.index, r"./obj/invIdxDict.pk")
            pu.dump(self.documentIndex, r"./obj/docIdxDict.pk")

    def search(self, query):
        """
        Search for terms in `query`
        Currently scores documents based on number of terms from query that match a document.
        """
        # TODO: Correct spelling in query terms
        # TODO: Stop-word removal in query terms
        # TODO: Consider positional indexes while searching

        scoreIndex = {}
        queryTerms = nlp(query)

        for term in queryTerms:
            termLem = term.lemma_
            if termLem not in self.index:
                continue
            for doc in self.index[termLem]:
                if doc not in scoreIndex:
                    scoreIndex[doc] = 0
                scoreIndex[doc] += len(self.index[termLem][doc])

        rankedDict = {
            k: v
            for k, v in sorted(
                scoreIndex.items(), key=lambda item: item[1], reverse=True
            )
        }
        return rankedDict
