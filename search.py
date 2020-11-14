import time

from utils import load
from models import InvertedIndexTfIdf, VectorSpaceModel

if __name__ == "__main__":
    idxObj = InvertedIndexTfIdf(
        r"./archive/TelevisionNews",
        ["meat and dairy", "clouds", "Donald Trump", "women", "clouds", "clouds over"],
    )

    query = input("Enter search query: ")
    startTime = time.time()
    results, rocchioRes, suggestions = idxObj.search(query)
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

    print("Results:")
    # print(results)
    for docID in results:
        print("DocID:", docID, "Score:", results[docID])
        print(documentIndex[docID])
        print("----------------------")

    print("Rocchio Modification")
    # print(rocchioRes)
    for docID in rocchioRes:
        print("DocID:", docID, "Score:", rocchioRes[docID])
        print(documentIndex[docID])
        print("----------------------")

    print("Suggestions:")
    for suggestion in suggestions:
        print(suggestion[0], "-", suggestion[1])
