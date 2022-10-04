from tweetopic import TopicPipeline, DMM
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from contextualized_topic_models.evaluation.measures import CoherenceNPMI
import json

def main(responses=False):
    if responses is False:
        topic_df = pd.read_json('processed/pre_topic_tweets.jsonl', 
                                orient='records', 
                                lines=True)
        topic_df['text'] = topic_df['text'].str.replace(r'&amp', 
                                                        'and', 
                                                        regex=True)
    else:
        topic_df = pd.read_json('processed/pre_topic_responses_sentiment.jsonl', 
                                orient='records', 
                                lines=True).drop_duplicates('response_text')
        topic_df = topic_df[topic_df['response_lang']=='en']
        #topic_df['response_text'] = topic_df['response_text'].str.replace(r'&amp', 
        #                                                                  'and', 
        #                                                                  regex=True)
    if responses is False:
        train_texts = topic_df[topic_df['topic_split']=='train'].text.tolist()
        val_texts = topic_df[topic_df['topic_split']=='val'].text.tolist()
        test_texts = topic_df[topic_df['topic_split']=='test'].text.tolist()
    else:
        train_texts = topic_df[topic_df['topic_split']=='train'].response_text.tolist()
        val_texts = topic_df[topic_df['topic_split']=='val'].response_text.tolist()
        test_texts = topic_df[topic_df['topic_split']=='test'].response_text.tolist()        
    scores = []
    for min_df in [10, 100]:
        for max_df in [0.01, 0.1, 0.2]:
            for n_clusters in [10, 15, 20, 30]:
                for max_features in [250, 500]:

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
                    if responses is False:
                        id_model = f'components-{n_clusters}_vocab-{max_features}_mindf-{min_df}_maxdf-{max_df}'
                    else:
                        id_model = f'responses_components-{n_clusters}_vocab-{max_features}_mindf-{min_df}_maxdf-{max_df}'
                    json.dump(topic_save, 
                              open(f'logs/baselines/model_{id_model}.json', 'w'))

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
                    if responses is False:
                        pred_mat = pipeline.transform(topic_df['text'].tolist())
                    else:
                        pred_mat = pipeline.transform(topic_df['response_text'].tolist())
                    pred_mat = pd.DataFrame(pred_mat, columns=[f'topic_{i}' 
                                                               for i in range(n_clusters)])
                    pred_mat = pd.concat([topic_df, pred_mat], 
                                          axis=1)
                    print(pred_mat.columns)
                    pred_mat.to_json(f'logs/baselines/predictions_{id_model}.jsonl',
                                     orient='records', lines=True)

                # Save
                if responses is False:
                    pd.DataFrame(scores).to_json('logs/baselines/performances.jsonl', 
                                                 orient='records', 
                                                 lines=True)
                else:
                    pd.DataFrame(scores).to_json('logs/baselines/responses_performances.jsonl', 
                                                 orient='records', 
                                                 lines=True)

if __name__=='__main__':
    #main(responses=False)
    main(responses=True)
    