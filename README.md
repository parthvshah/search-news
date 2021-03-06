# Search Engine for Environmental News 
A search engine for the Environmental News NLP Dataset. Created by Team ID-11 for the Algorithms for Information Retrieval Course - UE17CS412 at PES University, 2020.

## Setup
1. Download dataset from [Kaggle](https://www.kaggle.com/amritvirsinghx/environmental-news-nlp-dataset). Unzip the file and save as `./archive/TelevisionNews`.
2. Run `./setup.sh`

## Running a query on the Search Engine
1. Run `python3 server.py`. When run for the first time, all the documents will be parsed and the inverted index for information retrieval will be built and stored locally. 
2. Open [localhost:5000](http://localhost:5000) in you local browser and start querying!

## About the Project
Firstly, all documents are parsed and preprocessed. Preprocessing involves the removal of symbols and stop words and conversion of numbers to words. We then build an inverted index that consists of the token and posting list. Each posting list consists of `(documentID, token_frequency)` pairs.  

To rank the documents for information retrieval, we have built the Advanced TfIdf Model - In this model, after a query is processed, we traverse just the relevant documents in the posting lists of the query tokens and compute the vector similarity score between the query and those documents. We also use the Rocchio Algorithm to modify the query vector to retrieve further relevant results by moving the TfIdf values of the vector towards the centroids of relevant documents and away from the centroids of irrelevant documents. 

Additionally, we built a spelling checker to correct wrong words in queries and a query log to provide suggestions based on the similarity of the query with recently searched queries. Retrieved documents can further be sorted by date or grouped by news source.

## Team Members
1. [Aditya Vinod Kumar](https://github.com/adityavinodk) - PES1201700138
2. [Parth Vipul Shah](https://github.com/parthvshah) - PES1201700134
3. [Gaurang Rao](https://github.com/Gaupeng) - PES1201701103
4. [Richa Sharma](https://github.com/richa13sharma) - PES1201700662

## Future Work
1. Develop Query Parser technique - make use of a combination phrase query, biword phrase query and TfIdf inverted index to provide accurate and relevant retrievals.
2. Use query log, natural language context and word similarity methods to find semantically similar alternate queries for further improving retrieval results. Possible implementation using Sequential Neural Networks.
3. Advanced search based on date, news source, etc.
