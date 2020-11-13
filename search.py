import time

from models import VectorSpaceModel
from utils import load

if __name__ == "__main__":
    idxObj = VectorSpaceModel(r"./archive/test")

    query = input("Enter search query: ")
    startTime = time.time()
    results = idxObj.search(query)
    endTime = time.time()

    documentIndex = load("./obj/docDict.pk")

    searchTime = round(endTime - startTime, 3)
    print(
        "Searched across "
        + str(len(documentIndex))
        + " documents in "
        + str(searchTime)
        + " seconds."
    )

    for result in results:
        print("DocID:", result[0])
        print("Score:", result[1])
        print(documentIndex[result[0]].snippet)
        print("----------------------")
