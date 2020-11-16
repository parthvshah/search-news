#!/bin/bash
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
mkdir archive obj
rm ./archive/TelevisionNews/CNN.200910.csv
export FLASK_APP=server.py
export FLASK_ENV=development
python3 -m flask run
