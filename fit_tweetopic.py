from tweetopic import TopicPipeline, DMM
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from contextualized_topic_models.evaluation.measures import CoherenceNPMI
import json
import argparse
import pickle as pkl
import glob
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--eucomm-only', type=int, default=1)


def main(eucomm_only):
    if eucomm_only is False:
        fs = glob.glob('data/derivatives/*')
    else:
        fs = glob.glob('data/derivatives/EU_Comm*')
    dfs = []
    for f in fs:
        df = pd.read_json(f, orient='records', lines=True)
        df['entity'] = f.split('/')[-1][:-16]
        dfs.append(df)
    topic_df = pd.concat(dfs, ignore_index=True)
    
    train_texts = topic_df[topic_df['topic_split']=='train'].text.tolist()
    val_texts = topic_df[topic_df['topic_split']=='val'].text.tolist()
    test_texts = topic_df[topic_df['topic_split']=='test'].text.tolist()
    
    eucomm_train_texts = topic_df[(topic_df['topic_split']=='train') & 
                                  (topic_df['entity']=='EU_Commission')].text.tolist()
    eucomm_val_texts = topic_df[(topic_df['topic_split']=='val') & 
                                (topic_df['entity']=='EU_Commission')].text.tolist()
    eucomm_test_texts = topic_df[(topic_df['topic_split']=='test') &
                                 (topic_df['entity']=='EU_Commission')].text.tolist()
    
    scores = []
    for run in range(5):
        for min_df in [10, 100]:
            for max_df in [0.01, 0.1, 0.2]:
                for n_clusters in [5, 10, 15, 20, 30]:
                    for max_features in [100, 250, 500]:

                        # Fit
                        vectorizer = CountVectorizer(min_df=min_df, 
                                                     max_df=max_df, 
                                                     max_features=max_features)
                        dmm = DMM(n_components=n_clusters, 
                                  n_iterations=100, 
                                  alpha=0.1, beta=0.1)
                        pipeline = TopicPipeline(vectorizer, dmm)
                        pipeline.fit(train_texts)

                        # Eval
                        topic_dict = pipeline.top_words(top_n=10)
                        topics = [list(d.keys()) for d in topic_dict]
                        topic_save = {i: t for i, t in enumerate(topics)}
                        id_model = f'comps-{n_clusters}_vocab-{max_features}'
                        id_model += f'_mindf-{min_df}_maxdf-{max_df}'
                        if eucomm_only:
                            OUT_PATH = Path('models') / 'tweetopic' / 'eucomm' / id_model / f'run-{run}'
                        else:
                            OUT_PATH = Path('models') / 'tweetopic' / 'all' / id_model / f'run-{run}'
                        OUT_PATH.mkdir(exist_ok=True, parents=True)
                        json.dump(topic_save, 
                                  open(str(OUT_PATH / f'model.json'), 'w'))
                        pkl.dump(pipeline, 
                                  open(str(OUT_PATH / f'model.pkl'), 'wb'))
                        
                        words = pipeline.vectorizer.get_feature_names_out() 
                        for split, ds, ent in zip(['train','train',
                                                   'val','val',
                                                   'test','test'],
                                                   [train_texts, 
                                                    eucomm_train_texts, 
                                                    val_texts, 
                                                    eucomm_val_texts, 
                                                    test_texts, 
                                                    eucomm_test_texts],
                                                   ['all', 
                                                    'EU_Commission',
                                                    'all', 
                                                    'EU_Commission',
                                                    'all', 
                                                    'EU_Commission']):
                            out = pipeline.vectorizer.transform(ds)
                            
                            feats = [] 
                            for o in out.toarray(): 
                                f = [words[i] for i in range(len(words)) if o[i]==1]
                                feats.append(f)
                            metric = CoherenceNPMI(topics=topics, texts=feats)
                            cscore = metric.score()
                            score_dict = {'name': id_model,
                                          'split': split,
                                          'score': cscore,
                                          'entity': ent,
                                          'run': run}
                            scores.append(score_dict)
                            
                        pred_mat = pipeline.transform(topic_df['text'].tolist())
                        pred_mat = pd.DataFrame(pred_mat, columns=[f'topic_{i}' 
                                                                   for i in range(n_clusters)])
                        pred_mat = pd.concat([topic_df, pred_mat], axis=1, ignore_index=True)
                        if eucomm_only:
                            PRED_PATH = Path('logs') / 'tweetopic' / 'eucomm' / id_model / f'run-{run}'
                        else:
                            PRED_PATH = Path('logs') / 'tweetopic' / 'all' / id_model / f'run-{run}'
                        PRED_PATH.mkdir(exist_ok=True, parents=True)
                        pred_mat.to_json(str(PRED_PATH / f'preds.jsonl'),
                                         orient='records', 
                                         lines=True)

                    # Save
                    if eucomm_only:
                        DF_PATH = Path('logs') / 'tweetopic' / 'eucomm'
                    else:
                        DF_PATH = Path('logs') / 'tweetopic' / 'all'
                    pd.DataFrame(scores).to_json(str(DF_PATH / 'performances.jsonl'), 
                                                 orient='records', 
                                                 lines=True)
                

if __name__=='__main__':
    args = parser.parse_args()
    main(bool(args.eucomm_only))
