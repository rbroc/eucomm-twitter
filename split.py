import pandas as pd
import numpy as np
import random
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--entity', type=str, default='10DowningStreet')


def main(entity, style_subset_only=True):
    df = pd.read_json(f'data/derivatives/{entity}_annotated.jsonl', 
                      orient='records', lines=True)
    
    # Make pretraining split
    random.seed(42)
    for c in ['pretraining', 'topic', 'engagement']:
        
        # Make engagement splits
        train_size = int(df.shape[0] * .7)
        val_size = int(df.shape[0] * .15)
        test_size = df.shape[0] - train_size - val_size
        idxs = ['train'] * train_size
        idxs += ['val'] * val_size        
        idxs += ['test'] * test_size
        random.shuffle(idxs)
        df[f'{c}_split'] = idxs
        
    # Save
    df.to_json(f'data/derivatives/{entity}_annotated.jsonl', 
               orient='records', lines=True)

if __name__=='__main__':
    args = parser.parse_args()
    main(args.entity)


