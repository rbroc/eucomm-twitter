from ctm import CTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessingStopwords
from contextualized_topic_models.evaluation.measures import CoherenceNPMI, InvertedRBO
import nltk
from nltk.corpus import stopwords as stop_words
import pandas as pd
import numpy as np


def _save_results(rlist):
    fname = 'logs/topic/performances.jsonl'
    try:
        rdf = pd.read_json(fname, 
                           orient="records",
                           lines=True)
        rdf = pd.concat([rdf, pd.DataFrame(rlist)])
    except:
        rdf = pd.DataFrame(rlist)
    rdf.to_json(fname, orient="records", lines=True)

    
def _compute_metrics(split, ds, tlists, scores):
    npmi = CoherenceNPMI(texts=[d.split() for d in ds], 
                         topics=tlists)
    rbo = InvertedRBO(topics=tlists)
    scores[f'name'] = ctm.model_name
    scores[f'{split}_rbo'] = rbo.score()
    scores[f'{split}_npmi'] = npmi.score()
    

def main():
    topic_df = pd.read_json('processed/pre_topic_tweets.json', 
                            orient='records', 
                            lines=True)
    
    # Get training indices
    train_idx = set(np.where(topic_df['topic_split']=='train')[0].tolist())
    val_idx = set(np.where(topic_df['topic_split']=='val')[0].tolist())
    test_idx = set(np.where(topic_df['topic_split']=='test')[0].tolist())

    # Preprocess
    nltk.download('stopwords')
    stopwords = list(stop_words.words("english"))
    documents = topic_df.text.tolist()
    logpath = 'logs/topic'
    vocabulary_sizes = [500, 1000, 2000]
    score_list = []
    
    # Parameters
    for vs in vocabulary_sizes:
        sp = WhiteSpacePreprocessingStopwords(documents, 
                                              stopwords_list=stopwords, 
                                              vocabulary_size=vs)
        prepped, unprepped, vocab, retained_idx = sp.preprocess()

        prepped_train = [prepped[i] for i in range(len(prepped)) 
                         if retained_idx[i] in train_idx]
        prepped_val = [prepped[i] for i in range(len(prepped)) 
                       if retained_idx[i] in val_idx]
        prepped_test = [prepped[i] for i in range(len(prepped)) 
                        if retained_idx[i] in test_idx]
        unprepped_train = [unprepped[i] for i in range(len(unprepped)) 
                           if retained_idx[i] in train_idx]
        unprepped_val = [unprepped[i] for i in range(len(unprepped)) 
                         if retained_idx[i] in val_idx]
        unprepped_test = [unprepped[i] for i in range(len(unprepped)) 
                          if retained_idx[i] in test_idx]

        # Set parameters and prepare
        models = ["all-mpnet-base-v2"]
        n_comps = [5, 10, 20, 50, 100]
        ctx_size = 768 
        batch_sizes = [4, 16, 64]
        lrs = [2e-2, 2e-3, 2e-5]
        n_epochs = [10, 20, 100]

        for model in models:
            for n_components in n_comps:
                    for batch_size in batch_sizes:
                        for lr in lrs:
                            for num_epochs in n_epochs:
                                
                                # Preparation
                                scores = {}
                                tp = TopicModelDataPreparation(model)
                                train_dataset = tp.fit(unprepped_train, 
                                                       prepped_train)
                                val_dataset = tp.transform(unprepped_val, 
                                                           prepped_val)
                                test_dataset = tp.transform(unprepped_test, 
                                                            prepped_test)
                                
                                # Fit and predict
                                ctm = CombinedTM(bow_size=vs, 
                                                 contextual_size=ctx_size, 
                                                 n_components=n_components, 
                                                 num_epochs=num_epochs,
                                                 batch_size=batch_size,
                                                 lr=lr,
                                                 activation='softplus',
                                                 vocabulary_size=vs)
                                ctm.fit(training_dataset)
                                pred_train_topics = ctm.get_thetas(training_dataset, 
                                                                   n_samples=20)
                                pred_val_topics = ctm.get_thetas(val_dataset, 
                                                                 n_samples=20)
                                pred_test_topics = ctm.get_thetas(test_dataset, 
                                                                  n_samples=20)

                                # Save topics
                                model_out = f'{logpath}/{ctm.model_name}'
                                topic_out = f'{model_out}/topic_map.json'
                                json.dump(open(topic_out), ctm.get_topics())

                                # Merge predicted topics with tweets table
                                texts = pd.DataFrame(unprepped_train + \
                                                     unprepped_val + \
                                                     unprepped_test, 
                                                     columns=['text'])
                                pred_mat = np.vstack([pred_train_topics,
                                                      pred_val_topics,
                                                      pred_test_topics]).round(4)
                                col_names = [f'topic_{i}' 
                                             for i in range(n_components)]
                                preds = pd.DataFrame(pred_mat,
                                                     columns=col_names)
                                merged = topic_df.merge(pd.concat([texts, 
                                                                   preds], 
                                                                   axis=1))
                                merged.to_json(f'{model_out}/topic_preds.json',
                                               orient='records', 
                                               lines=True)

                                # Evaluate model
                                tlists = ctm.get_topic_lists(10)
                                _compute_metrics('train', 
                                                 prepped_train, 
                                                 tlists, 
                                                 scores)
                                _compute_metrics('val', 
                                                 prepped_train, 
                                                 tlists, 
                                                 scores)
                                _compute_metrics('test', 
                                                 prepped_train, 
                                                 tlists, 
                                                 scores) 
                                score_list.append(scores)

                                # Save model
                                ctm.save(models_dir='models/topic')
                                _save_results(scores)

    
if __name__=="__main__":
    main()