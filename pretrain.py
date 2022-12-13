from src.pretraining import Pretrainer
import pandas as pd
from pathlib import Path
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--eucomm-only', type=int, default=1)


def _save_results(rlist, eucomm_only):
    fname = Path('logs') / 'pretraining'
    if eucomm_only:
         fname = fname / 'eucomm'
    else:
         fname = fname / 'all'
    fname.mkdir(exist_ok=True, parents=True)
    fname = fname / 'performances.jsonl'
    try:
        rdf = pd.read_json(fname, 
                           orient="records",
                           lines=True)
        rdf = pd.concat([rdf, pd.DataFrame(rlist)])
    except:
        rdf = pd.DataFrame(rlist)
    rdf.to_json(fname, orient="records", lines=True)


def main(eucomm_only):
    
    # Read the data
    if eucomm_only:
        fs = glob.glob('data/derivatives/EU_Comm*')
    else:
        fs = glob.glob('data/derivatives/*')
    dfs = []
    for f in fs:
        df = pd.read_json(f, lines=True, orient='records')
        dfs.append(df)
    df = pd.concat(dfs)
    
    # Run pretraining loop
    models = ['distilbert-base-uncased']
    results = []

    # Run entire loop
    for lr in [2e-6, 2e-5, 2e-3]:
        for batch_size in [16]:
            for wu_epochs in [3]:
                for cs in [30, 50, 100]:
                    for m in models:
                        trainer = Pretrainer(m, df, 
                                              batch_size=batch_size, 
                                              lr=lr, 
                                              warmup=batch_size*wu_epochs,
                                              chunk_size=cs)
                        trainer.compile()
                        r = trainer.fit()
                        results.append(r)
                        if eucomm_only:
                            OUT_PATH = Path('models') / 'pretraining' / 'eucomm'
                        else:
                            OUT_PATH = Path('models') / 'pretraining' / 'all'
                        OUT_PATH.mkdir(exist_ok=True, parents=True)
                        trainer.save(str(OUT_PATH/f'{trainer.name}'))
                        _save_results(results, eucomm_only)   
                        
if __name__=="__main__":
    args = parser.parse_args()
    main(bool(args.eucomm_only))
