from src.preprocessing import (_read_tweets, 
                               _preprocessing,
                               _read_df)
import argparse
import glob
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--entity', type=str, default='10DowningStreet')


FIELDS = ['text', 'lang', 'id']
METRICS = [f'{m}_count' 
           for m in ['like','quote','reply','retweet']]


def main(entity, splits, keep_en=True):
    fs = glob.glob(f'data/raw/{entity}/*')
    if entity in ['10DowningStreet', 'POTUS']:
        df = pd.read_csv(fs[0], index_col=0)
        df = _read_df(df, fields=FIELDS, metrics=METRICS)
    else:
        df = _read_tweets(fs, fields=FIELDS, metrics=METRICS)
    df = _preprocessing(df, splits=splits) 
    if keep_en:
        df = df[(df['lang']=='en')]
    df.to_json(f'data/preprocessed/{entity}.jsonl', 
               orient='records', lines=True)
    
    
if __name__=='__main__':
    args = parser.parse_args()
    if args.entity == 'EU_Commission':
        splits = True
    else:
        splits = False
    main(args.entity, splits)
