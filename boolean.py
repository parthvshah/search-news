import pandas as pd
import glob
import time

from pickle_utils import load, dump

# returns a string after cleaning punctuation
def punct_clean(string):
    punc = """!()-[]{};:'"\, <>./?@#$%^&*_~"""

    for char in string:
        if char in punc:
            string = string.replace(char, "")

    return string


# returns all document 'Snippets' as a list
def create_text_list():
    path = r"./archive/TelevisionNews"
    all_files = glob.glob(path + "/*.csv")

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    text_list_df = pd.concat(li, axis=0, ignore_index=True)
    text_list = text_list_df.Snippet.str.lower().values
    return text_list


# create inverted index
def create_inverted_index(text_list):
    inverted = dict()

    for i, text in enumerate(text_list):
        # TODO: Add a tokenizer, correct spellings
        tokens = text.split(" ")

        # TODO: Use a B-Tree instead
        for token in tokens:
            token = punct_clean(token)
            if token not in inverted:
                inverted[token] = [i]
            else:
                inverted[token].append(i)

    return inverted


# splits boolean query
def process_query(query):
    query = query.split("|")
    print(query)
    return query


# given a query, returns respective documents
def search(index, original, query):
    query = process_query(query)
    targets = []
    for term in query:
        targets.extend(index[term])

    results = []

    for target in targets:
        results.append(original[target])

    return results


if __name__ == "__main__":
    text_list = create_text_list()

    # inverted = create_inverted_index(text_list)
    # dump(inverted, './obj/inverted.pk')

    inverted = load("./obj/inverted.pk")

    """
    Currently supported:
        | - OR
    """

    query = "cloud|trump"

    start_t = time.time()
    documents = search(inverted, text_list, query)
    end_t = time.time()

    search_time = round(end_t - start_t, 3)

    print("Query:", query)
    documents_len = len(documents)
    if documents_len != 0:
        for document in documents:
            print(document)
            print("---")
        print(documents_len, "results in", search_time, "seconds")
    else:
        print("N/A")
