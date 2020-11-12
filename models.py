import glob
import pandas as pd
import spacy as sp
import numpy as np
from collections import Counter

from collections import Counter, defaultdict
from itertools import islice
from math import log
from tqdm import tqdm
from utils import dump, load, retrieveSnippetsFromFile, preprocess, spellchecker

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


class VectorSpaceModel:
    """
    Dictionary implementation of inverted index with TFIDF similarity measure.
    """

    def __init__(self, dataDirectory):
        """
        `dataDirectory` is path to directory of corpus.
        """
        self.dataDir = dataDirectory
        self.documentTokens = {}
        self.docCount = 0
        self.totalVocab = []
        self.docVectors = []
        self._constructIndex()

    def _documentFreq(self, word):
        count = 0
        try:
            count = self.DF[word]
        except:
            pass

        return count

    def _constructIndex(self):
        """
        Construct index from documents.
        """
        try:
            self.index = load("./obj/vectorSpace.pk")
            self.documentTokens = load("./obj/docIdx.pk")

        except (OSError, IOError):
            documents = glob.glob(self.dataDir + "/*.csv")
            self.docCount = 0

            # load, pre-process and count
            for documentID in tqdm(range(len(documents))):
                document = documents[documentID]
                snippets = retrieveSnippetsFromFile(document)

                for snippet in snippets:
                    # TODO: Add preprocessing, try stemming, save original
                    snippet = preprocess(snippet)
                    self.documentTokens[self.docCount] = snippet
                    self.docCount += 1

            # create df
            DF = {}
            vocabSize = 0
            for i in range(self.docCount):
                tokens = self.documentTokens[i]
                for token in tokens:
                    try:
                        DF[token].add(i)
                    except:
                        DF[token] = {i}
                        vocabSize += 1

            for i in DF:
                DF[i] = len(DF[i])

            self.totalVocab = [x for x in DF]

            # calculate tf.idf
            tfidf = {}
            docNum = 0

            for i in range(self.docCount):
                tokens = self.documentTokens[i]
                counter = Counter(tokens)
                wordCount = len(tokens)

                for token in np.unique(tokens):
                    tf = counter[token] / wordCount
                    df = self._documentFreq(token)
                    idf = np.log((self.docCount + 1) / (df + 1))

                    tfidf[docNum, token] = tf * idf

                docNum += 1

            # vectorize
            self.docVectors = np.zeros((self.docCount, vocabSize))
            for i in tfidf:
                try:
                    index = self.totalVocab.index(i[1])
                    self.docVectors[i[0]][index] = tfidf[i]
                except:
                    pass

            dump(self.docVectors, "./obj/vectorSpace.pk")
            dump(self.documentTokens, "./obj/docIdx.pk")

    def _vectorize(self, tokens):
        query = np.zeros((len(self.totalVocab)))
        counter = Counter(tokens)
        wordCount = len(tokens)

        for token in np.unique(tokens):
            tf = counter[token] / wordCount
            df = self._documentFreq(token)
            idf = np.log((self.docCount + 1) / (df + 1))

            try:
                index = self.totalVocab.index(token)
                query[index] = tf * idf
            except:
                pass

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

        for document in self.docVectors:
            docCosines.append(self._cosineSim(queryVector, document))

        result = np.array(docCosines).argsort()[-top:][::-1]

        # returns array of docIDs
        return result
