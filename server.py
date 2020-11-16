from flask import Flask, request, render_template
from numpy.core.fromnumeric import sort

app = Flask(__name__, static_folder="static")

import time
from datetime import datetime

from utils import load
from models import InvertedIndexTfIdf

print("Search engine init")
idxObj = InvertedIndexTfIdf(
    r"./archive/TelevisionNews",
    ["meat and dairy", "clouds", "Donald Trump", "women", "clouds over"],
)
documentIndex = load("./obj/meta.pk")
print("Init completed")


def searchUsingModel(query):
    global documentIndex
    startTime = time.time()
    spelling, results, rocchioRes, suggestions = idxObj.search(query)
    endTime = time.time()
    searchTime = round(endTime - startTime, 3)
    toRender = []

    for docID in rocchioRes:
        toRender.append(documentIndex[docID])

    return spelling, toRender, suggestions, searchTime


def sortByValue(documents, value):
    # 1 - added first, 2 - added last, 3 - group by
    if value == 1:
        documents.sort(key=lambda x: datetime.strptime(x[1].split()[0], "%m/%d/%Y"))
        # return sorted(documents, key=lambda x: x[1].split()[0])
        return documents
    if value == 2:
        documents.sort(
            key=lambda x: datetime.strptime(x[1].split()[0], "%m/%d/%Y"), reverse=True
        )
        # return sorted(documents,key=lambda x: x[1].split()[0], reverse=True)
        return documents

    if value == 3:
        documents.sort(key=lambda x: x[2])
        return documents

    else:
        return documents


@app.route("/", methods=["GET"])
def searchHome():
    query = request.args.get("query")
    sortBy = request.args.get("sortBy")
    if sortBy:
        sortBy = int(sortBy)

    print("Query:", query)
    if query:
        spelling, rocchioRes, suggestions, searchTime = searchUsingModel(query)

        sortedResults = sortByValue(rocchioRes, sortBy)

        return render_template(
            "index.html",
            query=query,
            spellCheck=spelling,
            results=sortedResults,
            suggestions=suggestions,
            timeTaken=searchTime,
        )

    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run()
