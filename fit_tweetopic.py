from tweetopic import TopicPipeline, DMM
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from contextualized_topic_models.evaluation.measures import CoherenceNPMI
import json

def main():
    topic_df = pd.read_json('data/EU_Commission.jsonl', 
                            orient='records', 
                            lines=True)
    train_texts = topic_df[topic_df['topic_split']=='train'].text.tolist()
    val_texts = topic_df[topic_df['topic_split']=='val'].text.tolist()
    test_texts = topic_df[topic_df['topic_split']=='test'].text.tolist()
    
    scores = []
    for min_df in [10, 100]:
        for max_df in [0.01, 0.1, 0.2]:
            for n_clusters in [10, 15, 20, 30]:
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
                    json.dump(topic_save, 
                              open(f'logs/tweetopic/model_{id_model}.json', 'w'))

                    words = pipeline.vectorizer.get_feature_names_out() 
                    for split, ds in zip(['train','val','test'],
                                         [train_texts, val_texts, test_texts]):
                        out = pipeline.vectorizer.transform(ds)
                        feats = [] 
                        for o in out.toarray(): 
                            f = [words[i] for i in range(len(words)) if o[i]==1]
                            feats.append(f) 
                        metric = CoherenceNPMI(topics=topics, texts=feats)
                        cscore = metric.score()
                        score_dict = {'name': id_model,
                                      'split': split,
                                      'score': cscore}
                        scores.append(score_dict)
                    pred_mat = pipeline.transform(topic_df['text'].tolist())
                    pred_mat = pd.DataFrame(pred_mat, columns=[f'topic_{i}' 
                                                               for i in range(n_clusters)])
                    pred_mat = pd.concat([topic_df, pred_mat], axis=1)
                    pred_mat.to_json(f'logs/tweetopic/predictions_{id_model}.jsonl',
                                     orient='records', lines=True)

                # Save
                pd.DataFrame(scores).to_json('logs/tweetopic/performances.jsonl', 
                                             orient='records', 
                                             lines=True)
                
if __name__=='__main__':
    main()
