import time

from utils import load
from models import InvertedIndexTfIdf, VectorSpaceModel

if __name__ == "__main__":
    idxObj = InvertedIndexTfIdf(r"./archive/TelevisionNews")

    query = input("Enter search query: ")
    startTime = time.time()
    results = idxObj.search(query)
    endTime = time.time()

    documentIndex = load("./obj/docIdx.pk")

    searchTime = round(endTime - startTime, 3)
    print(
        "Searched across "
        + str(len(documentIndex))
        + " documents in "
        + str(searchTime)
        + " seconds."
    )

    for docID in results:
        print("DocID:", docID, "Score:", results[docID])
        print(documentIndex[docID])
        print("----------------------")
