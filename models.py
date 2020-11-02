import glob
import pandas as pd
import spacy as sp

from collections import Counter, defaultdict
from itertools import islice
from math import log
from tqdm import tqdm
from utils import dump, load, retrieveSnippetsFromFile, spellchecker

nlp = sp.load("en_core_web_sm")


class InvertedIndexDict:
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
            self.index = load("./obj/invIdxDict.pk")
            self.documentIndex = load("./obj/docIdx.pk")

        except (OSError, IOError):
            documents = glob.glob(self.dataDir + "/*.csv")
            docCount = 0
            print("Constructing Inverted Index")
            for documentID in tqdm(range(len(documents))):
                document = documents[documentID]
                snippets = retrieveSnippetsFromFile(document)
                for snippet in snippets:
                    docCount += 1
                    self.documentIndex[docCount] = snippet
                    spSnip = nlp(snippet)
                    for token in spSnip:
                        tokLemma = token.lemma_
                        if not tokLemma in self.index:
                            self.index[tokLemma] = {}
                        if not docCount in self.index[tokLemma]:
                            self.index[tokLemma][docCount] = set()
                        self.index[tokLemma][docCount].add(token.i)

            print("Indexed", len(self.index), "tokens")
            dump(self.index, "./obj/invIdxDict.pk")
            dump(self.documentIndex, "./obj/docIdx.pk")

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
            if termLem in self.index:
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


class InvertedIndexTfIdf:
    """
    Dictionary implementation of inverted index with TFIDF similarity measure.
    """

    def __init__(self, dataDirectory):
        """
        `dataDirectory` is path to directory of corpus.
        """
        self.dataDir = dataDirectory
        # index is the inverted index
        # each term maps to posting list consisting of (docId, term_frequency) pairs
        self.index = {}
        # documentIndex contains the mapping of docIDs to docNames.
        self.documentIndex = {}
        self.constructIndex()

    def constructIndex(self):
        """
        Construct index from documents.
        """
        try:
            self.index = load("./obj/invIdxTfIdf.pk")
            self.documentIndex = load("./obj/docIdx.pk")

        except (OSError, IOError):
            documents = glob.glob(self.dataDir + "/*.csv")
            docCount = 0
            print("Constructing Inverted Index")
            for documentID in tqdm(range(len(documents))):
                document = documents[documentID]
                snippets = retrieveSnippetsFromFile(document)
                for snippet in snippets:
                    docCount += 1
                    self.documentIndex[docCount] = snippet
                    spSnip = nlp(snippet)
                    tokenCounts = Counter([token.lemma_ for token in spSnip])
                    for token in tokenCounts:
                        tf = round(1 + log(tokenCounts[token], 10), 3)
                        if token in self.index:
                            self.index[token][docCount] = tf
                        else:
                            self.index[token] = {docCount: tf}

            print("Indexed", len(self.index), "tokens")
            dump(self.index, "./obj/invIdxTfIdf.pk")
            dump(self.documentIndex, "./obj/docIdx.pk")

    def search(self, query, top=10):
        """
        Search for terms in `query`
        Currently scores documents based on tfidf(t,q)*tfidf(t,d) similarity
        """
        # TODO: Use further heuristics to reduce query search time such as Query Parser, Impact Ordered postings, Relevance and Authority

        correctedQuery = spellchecker(query)
        if correctedQuery != query:
            print("Did you mean:", correctedQuery)
            query = correctedQuery

        scoreIndex = defaultdict(int)
        queryTerms = nlp(query)
        queryTermCounts = Counter([token.lemma_ for token in queryTerms])
        noOfDocuments = len(self.documentIndex)
        for term in queryTermCounts:
            if term in self.index:
                idf = round(log(noOfDocuments / len(self.index[term])), 3)
                postingList = {
                    k: v
                    for k, v in sorted(
                        self.index[term].items(), key=lambda item: item[1], reverse=True
                    )
                }
                for docIndex in postingList:
                    scoreIndex[docIndex] += round(
                        queryTermCounts[term] * (idf * postingList[docIndex]), 3
                    )

        rankedDict = {
            k: v
            for k, v in sorted(
                scoreIndex.items(), key=lambda item: item[1], reverse=True
            )
        }

        return dict(islice(rankedDict.items(), top))
