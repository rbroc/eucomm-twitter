import pandas as pd
import numpy as np
from pliers.extractors import PredefinedDictionaryExtractor, merge_results
from pliers.stimuli import ComplexTextStim


ext = PredefinedDictionaryExtractor(variables=['concreteness'])
tc = ["PredefinedDictionaryExtractor#concreteness_Conc.M"]
names = ['concreteness']

def main(style_subset_only=True):
    ''' Main function '''
    df = pd.read_json(f'data/topic/preds.jsonl',
                      orient='records', 
                      lines=True)

    vals = dict(zip(names, [[], [], []]))
    for i, t in enumerate(df['text']):
        if i % 1000 == 0:
            print(i)
        results = merge_results(ext.transform(ComplexTextStim(text=t)))
        for idx, tf in enumerate(tc):
            try:
                mval = results[tf].astype(float).mean()
                vals[names[idx]].append(mval)
            except:
                vals[names[idx]].append(np.nan)
    for n in names:
        df[n] = vals[n]
    
    df.to_json(f'data/topic/preds_extended.jsonl', 
               orient='records', lines=True)

if __name__=='__main__':
    main()


