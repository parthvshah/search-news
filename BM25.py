import glob
import pandas as pd
import spacy
import time

from rank_bm25 import BM25Okapi
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")

path = r"./archive/TelevisionNews"
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

text_list_df = pd.concat(li, axis=0, ignore_index=True)
text_list = text_list_df.Snippet.str.lower().values

tok_text = []

for doc in tqdm(nlp.pipe(text_list, disable=["tagger", "parser", "ner"])):
    tok = [t.text for t in doc if t.is_alpha]
    tok_text.append(tok)

bm25 = BM25Okapi(tok_text)

query = "Flood Defence"
print("Query:", query)

tokenized_query = query.lower().split(" ")
t0 = time.time()
results = bm25.get_top_n(tokenized_query, text_list_df.Snippet.values, n=3)
t1 = time.time()

print(
    f"Searched {len(text_list_df.Snippet.values)} records in {round(t1-t0,3)} seconds!\n"
)

for i in results:
    print(i)
    print("---")
