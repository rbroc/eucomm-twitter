from tweetopic import TopicPipeline, DMM
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from contextualized_topic_models.evaluation.measures import CoherenceNPMI


def main():
    topic_df = pd.read_json('processed/pre_topic_tweets.jsonl', 
                            orient='records', 
                            lines=True)
    train_texts = topic_df[topic_df['topic_split']=='train'].text.tolist()
    val_texts = topic_df[topic_df['topic_split']=='val'].text.tolist()
    test_texts = topic_df[topic_df['topic_split']=='test'].text.tolist()
    scores = []
    for n_clusters in [10, 20, 40, 50]:
        for max_features in [250, 500]:
            
            # Fit
            vectorizer = CountVectorizer(min_df=15, max_df=0.1, max_features=max_features)
            dmm = DMM(n_clusters=n_components, n_iterations=100, alpha=0.1, beta=0.1)
            pipeline = TopicPipeline(vectorizer, dmm)
            pipeline.fit(texts)
            
            # Eval
            topic_dict = pipeline.top_words(top_n=10)
            topics = [list(d.keys()) for d in topic_dict]
            
            words = pipeline.vectorizer.get_feature_names_out() 
            for split, ds in enumerate(zip(['train','val','test'],
                                           [train_texts, val_texts, test_texts])):
                out = pipeline.vectorizer.transform(ds)
                feats = [] 
                for o in out.toarray(): 
                    f = [words[i] for i in range(len(words)) if o[i]==1]
                    feats.append(f) 
                metric = CoherenceNPMI(topics=topics, texts=feats)
                cscore = metric.score()
                score_dict = {'name': f'components-{n_clusters}_vocab-{max_features}',
                              'split': split,
                              'score': cscore}
                scores.append(score_dict)
        
        # Save
        pd.DataFrame(scores).to_json('logs/baselines/performances.jsonl', 
                                     orient='records', 
                                     lines=True)
        
if __name__=='__main__':
    main()
    