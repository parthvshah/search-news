import time

from models import invertedIndexDict

if __name__ == "__main__":
    idxObj = invertedIndexDict(r"./archive/TelevisionNews")

    query = input("Enter search query: ")
    startTime = time.time()
    res = idxObj.search(query)
    endTime = time.time()

    searchTime = round(endTime - startTime, 3)
    print(
        "Searched for - "
        + query
        + " - across "
        + str(len(idxObj.documentIndex))
        + " documents in "
        + str(searchTime)
        + " seconds."
    )

    print(res)
