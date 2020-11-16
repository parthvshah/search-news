import glob
import numpy as np
import sys

from collections import Counter, defaultdict
from itertools import islice
from math import log
from tqdm import tqdm
from utils import (
    dump,
    levenshteinDistance,
    load,
    retrieveFile,
    preprocess,
    spellchecker,
)


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
        # document and all its metadata
        self.documentMetadata = {}
        # contains all the tokens of the corpus, used to create a vector
        self.totalVocab = []
        # contains n most recent queries for suggestions
        self.queryLog = queryLog

        self.noOfDocs = 0
        self.noOfTokens = 0

        self._constructIndex()

    def _constructIndex(self):
        """
        Construct index from documents.
        """
        try:
            self.index = load("./obj/invIdxTfIdf.pk")
            self.totalVocab = load("./obj/vocab.pk")
            self.documentIndex = load("./obj/docIdx.pk")
            self.documentMetadata = load("./obj/meta.pk")
            self.queryLog = load("./obj/log.pk")
            self.noOfDocs = len(self.documentIndex)
            self.noOfTokens = len(self.totalVocab)

        except (OSError, IOError):
            documents = glob.glob(self.dataDir + "/*.csv")
            docCount = 0
            print("Constructing Inverted Index")
            for documentID in tqdm(range(len(documents))):
                document = documents[documentID]
                values = retrieveFile(document)
                for value in values:
                    docCount += 1

                    self.documentIndex[docCount] = value[-1]
                    self.documentMetadata[docCount] = value
                    processed = preprocess(value[-1]).split()
                    self.totalVocab.extend(processed)

                    tokenCounts = Counter(processed)
                    for token in tokenCounts:
                        tf = round(1 + log(tokenCounts[token], 10), 3)
                        if token in self.index:
                            self.index[token][docCount] = tf
                        else:
                            self.index[token] = {docCount: tf}

            self.totalVocab = list(self.index.keys())
            self.noOfDocs = len(self.documentIndex)
            self.noOfTokens = len(self.totalVocab)

            print("Indexed", len(self.index), "tokens")
            dump(self.index, "./obj/invIdxTfIdf.pk")
            dump(self.totalVocab, "./obj/vocab.pk")
            dump(self.documentIndex, "./obj/docIdx.pk")
            dump(self.documentMetadata, "./obj/meta.pk")
            dump(self.queryLog[-100:], "./obj/log.pk")

    def _cosineSim(self, a, b):
        """
        Finds similarity between vectors `a` and `b` using cosine measure
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _vectorize(self, snippet):
        tokens = preprocess(snippet).split()
        query = np.zeros(self.noOfTokens + 1)
        counter = Counter(tokens)
        wordCount = len(tokens)

        for token in np.unique(tokens):
            tf = counter[token] / wordCount
            if token not in self.index:
                print("Invalid Query")
                sys.exit()

            df = len(self.index[token])
            idf = np.log((self.noOfDocs + 1) / (df + 1))
            index = self.totalVocab.index(token)
            query[index] = tf * idf

        return query

    def _rocchio(self, query, rankedList, rel=5, nonrel=5, a=1, b=0.75, g=0.15):
        """
        Extend `query` based on `rel` and `nonrel` documents from `rankedDict`, using Rocchio Algorithm.
        `a`: alpha; `b`: beta; `g`: gamma
        Returns modified query.
        """
        # map terms in rel/non-rel docs to total tf-scores
        relDocs = {}
        nonRelDocs = {}

        for i in range(rel):
            docID = rankedList[i]
            snippet = self.documentIndex[docID]
            processed = preprocess(snippet).split()
            for token in processed:
                if token not in relDocs:
                    relDocs[token] = 0
                relDocs[token] += self.index[token][docID]

        for i in range(rel, nonrel):
            docID = rankedList[i]
            snippet = self.documentIndex[docID]
            processed = preprocess(snippet).split()
            for token in processed:
                if token not in nonRelDocs:
                    nonRelDocs[token] = 0
                nonRelDocs[token] += self.index[token][docID]

        newQueryVec = self._vectorize(query) * a

        for token in relDocs:
            df = len(self.index[token])
            idf = np.log((self.noOfDocs + 1) / (df + 1))
            newQueryVec[self.totalVocab.index(token)] += (
                b * idf * (relDocs[token] / rel)
            )

        for token in nonRelDocs:
            df = len(self.index[token])
            idf = np.log((self.noOfDocs + 1) / (df + 1))
            newQueryVec[self.totalVocab.index(token)] -= (
                g * idf * (nonRelDocs[token] / nonrel)
            )

        return newQueryVec

    def _rankVectorSpace(self, documentDict, query, vectorizeQuery=True):
        """
        Creates ranking of documents in `documentDict` against `query`.
        If `vectorizeQuery` is set to False, `query` must be vector form.
        """
        queryVector = query
        if vectorizeQuery:
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
        """
        Returns `top` (max) suggestions against `query'
        """
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
            # print("Did you mean:", correctedQuery)
            query = correctedQuery

        self._log(query)

        scoreIndex = defaultdict(int)
        processed = preprocess(query).split()
        queryTermCounts = Counter(processed)

        for term in queryTermCounts:
            if term in self.index:
                idf = round(log((self.noOfDocs + 1) / len(self.index[term])), 3)
                postingList = {
                    k: v
                    for k, v in sorted(
                        self.index[term].items(), key=lambda item: item[1], reverse=True
                    )
                }
                # Can implement early-stopping by checking if TfIdf goes below a certain threshold
                # if idf * postingList[docIndex] > threshold:
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

        docList = list(rankedDict.keys())
        feedbackQuery = self._rocchio(query, docList[: 2 * top + 1])

        vectorSpaceRanked = self._rankVectorSpace(docList[: 2 * top + 1], query)
        vectorSpaceRocchio = self._rankVectorSpace(
            docList[: 3 * top + 1], feedbackQuery, False
        )

        # find suggestions from query log
        suggestions = self._suggestions(query, 3)

        # returns a dict with k: v as docID: score and suggestions
        return (
            correctedQuery,
            dict(islice(vectorSpaceRanked.items(), top)),
            dict(islice(vectorSpaceRocchio.items(), top)),
            suggestions,
        )
