import pickle as pu
import time

from models import VectorSpaceModel

if __name__ == "__main__":
    idxObj = VectorSpaceModel(r"./archive/test")

    query = input("Enter search query: ")
    startTime = time.time()
    results = idxObj.search(query)
    endTime = time.time()

    with open("./obj/ogDocs.pk", "rb") as f:
        documentIndex = pu.load(f)

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
        print(documentIndex[result[0]])
        print("----------------------")
