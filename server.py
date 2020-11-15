from flask import Flask, request, render_template

app = Flask(__name__, static_folder='static')

import time

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


@app.route("/", methods=["GET"])
def searchHome():
    query = request.args.get("query")
    print("Query:", query)
    if(query):
        spelling, rocchioRes, suggestions, searchTime = searchUsingModel(query)
        print("Rendering")
        return render_template(
            "index.html",
            query=query,
            spellCheck=spelling,
            results=rocchioRes,
            suggestions=suggestions,
            timeTaken=searchTime,
        )
    else: 
        return render_template("index.html")