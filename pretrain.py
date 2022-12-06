from pretrain import Pretrainer
import pandas as pd


def _save_results(rlist):
    fname = 'logs/pretraining/performances.jsonl'
    try:
        rdf = pd.read_json(fname, 
                           orient="records",
                           lines=True)
        rdf = pd.concat([rdf, pd.DataFrame(rlist)])
    except:
        rdf = pd.DataFrame(rlist)
    rdf.to_json(fname, orient="records", lines=True)


def main():
    
    # Read the data
    df = pd.read_json('processed/all_tweets.jsonl', lines=True, orient='records')
    df = df[(df['lang']=='en') & (df['lang_detected']=='en')]
    
    # Run pretraining loop
    models = ['distilbert-base-uncased',
              'distilbert-base-uncased-finetuned-sst-2-english',
              'cardiffnlp/tweet-topic-21-multi']
    results = []

    # Run entire loop
    for lr in [2e-6, 2e-5, 2e-3]:
        for batch_size in [4, 16]:
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
                        trainer.save(f'models/pretrained/{trainer.name}')
                        _save_results(results)   
                        
if __name__=="__main__":
    main()
