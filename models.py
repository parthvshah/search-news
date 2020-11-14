import glob
import numpy as np
import pandas as pd
import spacy as sp
import sys

from collections import Counter, defaultdict
from itertools import islice
from math import log
from tqdm import tqdm
from utils import (
    dump,
    load,
    retrieveSnippetsFromFile,
    preprocess,
    spellchecker,
    levenshteinDistance,
)

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
                    self.documentIndex[docCount] = snippet
                    docCount += 1
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

    def __init__(self, dataDirectory, queryLog=[]):
        """
        `dataDirectory` is path to directory of corpus.
        `log` is an optional param to populate the log before querying
        """
        self.dataDir = dataDirectory
        # index is the inverted index - each term maps to posting list consisting of (docId, term_frequency) pairs
        self.index = {}
        # documentIndex contains the mapping of docIDs to docNames.
        self.documentIndex = {}
        # contains all the tokens of the corpus, used to create a vector
        self.totalVocab = []
        # contains n most recent queries for suggestions
        self.queryLog = queryLog

        self._constructIndex()

    def _constructIndex(self):
        """
        Construct index from documents.
        """
        try:
            self.index = load("./obj/invIdxTfIdf.pk")
            self.totalVocab = load("./obj/vocab.pk")
            self.documentIndex = load("./obj/docIdx.pk")
            self.queryLog = load("./obj/log.pk")

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
                    processed = preprocess(snippet).split()
                    self.totalVocab.extend(processed)
                    tokenCounts = Counter(processed)
                    for token in tokenCounts:
                        tf = round(1 + log(tokenCounts[token], 10), 3)
                        if token in self.index:
                            self.index[token][docCount] = tf
                        else:
                            self.index[token] = {docCount: tf}

            print("Indexed", len(self.index), "tokens")
            dump(self.index, "./obj/invIdxTfIdf.pk")
            dump(self.totalVocab, "./obj/vocab.pk")
            dump(self.documentIndex, "./obj/docIdx.pk")
            dump(self.queryLog[-100:], "./obj/log.pk")

    def _cosineSim(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _vectorize(self, snippet):
        tokens = preprocess(snippet)
        tokens = tokens.split()
        query = np.zeros(len(self.totalVocab))
        counter = Counter(tokens)
        wordCount = len(tokens)

        for token in np.unique(tokens):
            tf = counter[token] / wordCount
            if token not in self.index:
                print("Invalid Query")
                sys.exit()

            df = len(self.index[token])
            idf = np.log((len(self.documentIndex) + 1) / (df + 1))
            index = self.totalVocab.index(token)
            query[index] = tf * idf

        return query

    def _rankVectorSpace(self, documentDict, query):
        queryVector = self._vectorize(query)
        docScores = {}
        for docID in documentDict:
            docVector = self._vectorize(self.documentIndex[docID])
            docScores[docID] = self._cosineSim(docVector, queryVector)

        # sort
        result = {
            k: v
            for k, v in sorted(
                docScores.items(), key=lambda item: item[1], reverse=True
            )
        }
        return result

    def _log(self, query):
        """
        Saves only the last n (100) queries
        """
        if query not in self.queryLog:
            self.queryLog.append(query)
            dump(self.queryLog[-100:], "./obj/log.pk")

    def _suggestions(self, query, top=10):
        distances = []
        for ql in self.queryLog[::-1]:
            dist = levenshteinDistance(ql, query)
            if dist != 0:
                distances.append((ql, dist))

        suggestions = sorted(distances, key=lambda x: x[1])
        return suggestions[:top]

    def search(self, query, top=10):
        """
        Search for terms in `query`
        Currently scores documents based on tf(t,q)*tfidf(t,d) similarity
        """
        # TODO: Use further heuristics to reduce query search time such as Query Parser, Impact Ordered postings, Relevance and Authority
        correctedQuery = spellchecker(query)
        if correctedQuery != query:
            print("Did you mean:", correctedQuery)
            query = correctedQuery

        self._log(query)

        scoreIndex = defaultdict(int)
        processed = preprocess(query).split()
        queryTermCounts = Counter(processed)
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

        vectorSpaceRanked = self._rankVectorSpace(
            dict(islice(rankedDict.items(), 2 * top)), query
        )

        # find suggestions from query log
        suggestions = self._suggestions(query, 3)

        # returns a dict with k: v as docID: score and suggestions
        return (dict(islice(vectorSpaceRanked.items(), top)), suggestions)


class DocumentBlueprint:
    """
    A simple way to store documents uniformly
    """

    def __init__(self, docID):
        self.docID = docID
        self.snippet = ""
        self.tokens = []
        self.docVector = []


class VectorSpaceModel:
    """
    Vector Space Model to search with TFIDF similarity measure.
    """

    def __init__(self, dataDirectory):
        """
        `dataDirectory` is path to directory of corpus.
        """
        self.dataDir = dataDirectory
        # A dictionary of all documents as DocumentBlueprint objects
        self.documentDictionary = {}

        # document count; each snippet is counted as a new document
        self.docCount = 0
        self._totalVocab = []
        self._vocabSize = 0
        self._docFreq = {}
        self._constructIndex()

    def _documentFreq(self, word):
        """
        Return frequency of documents that contain `word`
        """
        if word not in self._docFreq:
            return 0
        return len(self._docFreq[word])

    def _constructIndex(self):
        """
        Construct index from documents.
        """
        try:
            self.documentDictionary = load("./obj/docDict.pk")
            self.docCount = len(self.documentDictionary)
            self._totalVocab = load("./obj/vocab.pk")
            self._vocabSize = len(self._totalVocab)
            self._docFreq = load("./obj/docFreq.pk")

        except (OSError, IOError):
            documents = glob.glob(self.dataDir + "/*.csv")

            # load, pre-process and count
            for docID in tqdm(range(len(documents))):
                document = documents[docID]
                snippets = retrieveSnippetsFromFile(document)

                for snippet in snippets:
                    # TODO: Lemmatization?
                    newDoc = DocumentBlueprint(docID)
                    newDoc.snippet = snippet
                    newDoc.tokens = preprocess(snippet).split()
                    self.documentDictionary[self.docCount] = newDoc
                    self.docCount += 1

            # TODO: all the following funcs should happen in batches
            for i in range(self.docCount):
                tokens = self.documentDictionary[i].tokens
                for token in tokens:
                    if token not in self._docFreq:
                        self._docFreq[token] = set()
                    self._docFreq[token].add(i)

            self._totalVocab = [x for x in self._docFreq]
            self._vocabSize = len(self._totalVocab)

            # calculate tf.idf
            tfidf = {}
            docNum = 0

            for i in range(self.docCount):
                tokens = self.documentDictionary[i].tokens
                counter = Counter(tokens)
                wordCount = len(tokens)

                for token in np.unique(tokens):
                    tf = counter[token] / wordCount
                    df = self._documentFreq(token)
                    idf = np.log((self.docCount + 1) / (df + 1))
                    tfidf[docNum, token] = tf * idf

                self.documentDictionary[i].docVector = np.zeros(self._vocabSize)
                docNum += 1

            # vectorize
            for i in tfidf:
                index = self._totalVocab.index(i[1])
                self.documentDictionary[i[0]].docVector[index] = tfidf[i]

            # TODO: dump multiple docVectors into diff files
            dump(self.documentDictionary, "./obj/docDict.pk")
            dump(self._totalVocab, "./obj/vocab.pk")
            dump(self._docFreq, "./obj/docFreq.pk")

    def _vectorize(self, tokens):
        tokens = tokens.split()
        query = np.zeros(self._vocabSize)
        counter = Counter(tokens)
        wordCount = len(tokens)
        for token in np.unique(tokens):
            tf = counter[token] / wordCount
            df = self._documentFreq(token)
            idf = np.log((self.docCount + 1) / (df + 1))
            index = self._totalVocab.index(token)
            query[index] = tf * idf
        return query

    def _cosineSim(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def search(self, query, top=10):
        """
        Search for terms in `query`
        """

        # TODO: Use further heuristics to reduce query search time such as Query Parser, Impact Ordered postings, Relevance and Authority
        correctedQuery = spellchecker(query)
        if correctedQuery != query:
            print("Did you mean:", correctedQuery)
            query = correctedQuery

        print("Query:", query)
        tokens = preprocess(query)
        docCosines = []
        queryVector = self._vectorize(tokens)

        # TODO: search in multiple docVector files
        for document in self.documentDictionary:
            docCosines.append(
                self._cosineSim(
                    queryVector, self.documentDictionary[document].docVector
                )
            )

        resultIDs = np.array(docCosines).argsort()[-top:][::-1]
        resultCosines = []
        for Id in resultIDs:
            resultCosines.append(docCosines[Id])

        # returns array of (docIDs, cosineScore)
        return zip(resultIDs, resultCosines)
