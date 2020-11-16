import time

from utils import load
from models import InvertedIndexTfIdf

if __name__ == "__main__":
    idxObj = InvertedIndexTfIdf(
        r"./archive/TelevisionNews",
        ["meat and dairy", "clouds", "Donald Trump", "women", "clouds over"],
    )
    documentIndex = load("./obj/meta.pk")

    query = input("Enter search query: ")
    startTime = time.time()
    spellCheck, results, rocchioRes, suggestions = idxObj.search(query)
    endTime = time.time()

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
    print("\n\n")
    print("Rocchio Modification")
    # print(rocchioRes)
    for docID in rocchioRes:
        print("DocID:", docID, "Score:", rocchioRes[docID])
        print(documentIndex[docID])
        print("----------------------")

    print("Suggestions:")
    for suggestion in suggestions:
        print(suggestion[0], "-", suggestion[1])
