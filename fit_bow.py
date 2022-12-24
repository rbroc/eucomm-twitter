import pandas as pd
import numpy as np
from pathlib import Path
import json
import glob
import pickle as pkl
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
import os


def main():
     
    fs = glob.glob('data/derivatives/*')
    dfs = []
    for f in fs:
        df = pd.read_json(f, orient='records', lines=True)
        df['entity'] = f.split('/')[-1][:-16]
        dfs.append(df)
    data = pd.concat(dfs).reset_index(drop=True)
    
    modelpath = Path('models') / 'bow'
    outpath = Path('data') / 'annotated'
    
    fit_data = data.groupby('entity').sample(n=4000).reset_index()
    
    for dim in [250, 500, 1000]:
        
        stop_words = stopwords.words('english')
        tkpath = str(modelpath / f'tokenizer_bow-{dim}.pkl')
        c_vec = TfidfVectorizer(stop_words=stop_words,
                                max_features=dim)
        c_vec.fit(fit_data['text'].tolist())
        transformed = pd.DataFrame(c_vec.transform(data['text'].tolist()).toarray(),
                                   columns=[f'bow_{dim}_{w}' 
                                            for w in c_vec.vocabulary_.keys()])
        data = pd.concat([data, transformed], axis=1)
        pkl.dump(c_vec, open(tkpath, "wb"))
    data.to_json(outpath / 'data.jsonl', orient='records', lines=True)
    

if __name__=="__main__":
    main()