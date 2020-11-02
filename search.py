import pickle as pu
import time

from models import InvertedIndexDict, InvertedIndexTfIdf

if __name__ == "__main__":
    # idxObj = invertedIndexDict(r"./archive/TelevisionNews")
    idxObj = InvertedIndexTfIdf(r"./archive/TelevisionNews")

    query = input("Enter search query: ")
    startTime = time.time()
    res = idxObj.search(query)
    endTime = time.time()

    searchTime = round(endTime - startTime, 3)
    print(
        "Searched across "
        + str(len(idxObj.documentIndex))
        + " documents in "
        + str(searchTime)
        + " seconds."
    )

    with open("./obj/docIdx.pk", "rb") as f:
        documentIndex = pu.load(f)
    for doc in res:
        print("Doc", doc, ":", res[doc], "Score")
        print(documentIndex[doc])
        print("----------------------")
