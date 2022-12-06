from pathlib import Path
import pandas as pd
import json
import random
import numpy as np


def _read_tweets(fs, metrics, fields):
    ''' Read tweets that come in json '''
    processed_tws = []
    for f in fs:
        js = json.load(open(f))
        if 'data' in js.keys():
            tws = json.load(open(f))['data']
            for i in range(len(tws)):
                item = {k: tws[i][k] for k in fields}
                item.update({k: tws[i]['public_metrics'][k] for k in metrics})
                item.update({'created_at': tws[i]['created_at'][:10]})
                tws[i] = item
            processed_tws += tws
    df = pd.DataFrame(processed_tws)
    df['created_at'] = pd.to_datetime(df['created_at'], 
                                      infer_datetime_format=True)
    return df

def _read_df(df, metrics, fields):
    ''' Read tweets that come in csv '''
    for m in metrics:
        df[m] = df['public_metrics'].apply(lambda x: eval(x)[m])
    df = df.drop('public_metrics', axis=1)
    for c in df.columns:
        if c not in fields + metrics + ['public_metrics', 'created_at']:
            df = df.drop(c, axis=1)
    df['created_at'] = df['created_at'].str[:10]
    df['created_at'] = pd.to_datetime(df['created_at'], 
                                          infer_datetime_format=True)
    return df


def language_detection(s):
    ''' Annotate "special" tweets and strip links '''
    try:
        return detect(s)
    except:
        return 'unk'
    
    
def _preprocessing(df, seed=42, splits=True, train_size=3000, val_size=500):
    ''' Annotate "special" tweets and strip links '''
    df['is_retweet'] = np.where(df['text'].str.startswith('RT'), 1, 0)
    df['is_mention'] = np.where(df['text'].str.startswith('@'), 1, 0)
    df['text'] = df['text'].str.replace(r'http.*', '', regex=True)
    df = df[df['text'].str.len() > 0]
    df['text'] = df['text'].str.replace(f'&amp;', '&', regex=True)
    df['lang_detected'] = df['text'].apply(language_detection)
    df['sum_count'] = df['like_count'] + df['retweet_count'] + df['quote_count']
    df['sum_count'] = df['sum_count'] + df['reply_count']
    
    # Remove retweets and mentions
    df = df[(df['is_retweet']==0) & (df['is_mention']==0)]
    
    
    # Make splits
    if splits is True:
        random.seed(seed)
        train_test = ['train'] * train_size + ['val'] * val_size 
        train_test += ['test'] * (df.shape[0] - train_size - val_size)
        random.shuffle(train_test)
        df['pretraining_splits'] = train_test
        
        engage_train_size = int(df.shape[0] * .6)
        engage_val_size = int(df.shape[0] * .2)
        engage_test_size = df.shape[0] - engage_train_size - engage_val_size
        idxs = ['train'] * engage_train_size
        idxs += ['val'] * engage_val_size        
        idxs += ['test'] * engage_test_size
        random.shuffle(idxs)
        df['engagement_split'] = idxs
    
    return df