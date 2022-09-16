from ctmodel import CTModel
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessingStopwords
from contextualized_topic_models.evaluation.measures import CoherenceNPMI, InvertedRBO
from evaluation import CoherenceWordEmbeddings
import nltk
from nltk.corpus import stopwords as stop_words
import pandas as pd
import numpy as np
from pathlib import Path
import json
import glob
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _save_results(rlist):
    fname = 'logs/topic/performances_final.jsonl'
    try:
        rdf = pd.read_json(fname, 
                           orient="records",
                           lines=True)
        rdf = pd.concat([rdf, pd.DataFrame(rlist)])
        rdf['unique_name'] = rdf['name'] + '_' + rdf['run'].astype(str)
        rdf = rdf.drop_duplicates('unique_name').drop('unique_name', axis=1)
    except:
        rdf = pd.DataFrame(rlist)
    rdf.to_json(fname, orient="records", lines=True)

    
def _compute_metrics(split, ds, tlists, tlists_20, scores, ctm):
    scores[f'name'] = ctm.model_name
    for n, tl in zip([10, 20], [tlists, tlists_20]):
        npmi = CoherenceNPMI(texts=[d.split() for d in ds], 
                             topics=tl)
        rbo = InvertedRBO(topics=tl)
        cwe = CoherenceWordEmbeddings(topics=tl) 
        scores[f'{split}_npmi_{n}'] = npmi.score(topk=n).round(4)
        if split == 'train':
            scores[f'cwe_{n}'] = cwe.score(topk=n).round(4)
            scores[f'rbo_{n}'] = rbo.score(topk=n).round(4)
    

def main():
    topic_df = pd.read_json('processed/pre_topic_tweets.jsonl', 
                            orient='records', 
                            lines=True)
    best_models = pd.read_json('logs/topic/best_models.jsonl', 
                               orient='records', 
                               lines=True).to_dict('records')
    
    # Get training indices
    train_idx = set(np.where(topic_df['topic_split']=='train')[0].tolist()) 
    val_idx = set(np.where(topic_df['topic_split']=='val')[0].tolist())
    test_idx = set(np.where(topic_df['topic_split']=='test')[0].tolist())

    # Preprocess
    nltk.download('stopwords')
    stopwords = list(stop_words.words("english"))
    documents = topic_df.text.tolist()
    logpath = 'logs/topic'
    score_list = []
    
    # Parameters
    for p in best_models:
        for run in range(1,5):
            vs = p['vocab']
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
            model = p['id']
            if model != 'all-mpnet-base-v2':
                model = f'models/sent_transformers/{model}'
            n_components = p['components']
            ctx_size = p['esize']
            batch_size = 64
            lr = 2e-3
            bow = p['bow']

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
            ctm = CTModel(model=model,
                          bow_size=bow, 
                          contextual_size=ctx_size, 
                          n_components=n_components, 
                          num_epochs=100,
                          batch_size=batch_size,
                          lr=lr,
                          activation='softplus',
                          vocabulary_size=vs,
                          num_data_loader_workers=5)
            ctm.fit(train_dataset, 
                    validation_dataset=val_dataset)
            pred_train_topics = ctm.get_thetas(train_dataset, 
                                               n_samples=20)
            pred_val_topics = ctm.get_thetas(val_dataset, 
                                             n_samples=20)
            pred_test_topics = ctm.get_thetas(test_dataset, 
                                              n_samples=20)

            # Save topics
            model_out = f'{logpath}/{ctm.model_name}'
            Path(model_out).mkdir(parents=True, exist_ok=True)
            topic_out = f'{model_out}/topic_map_{run}.json'
            with open(topic_out, "w") as fh:
                json.dump(ctm.get_topics(k=20), fh)

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
            preds = pd.concat([texts, preds], axis=1)
            to_be_merged = topic_df.iloc[retained_idx, :]
            assert preds.shape[0] == to_be_merged.shape[0]
            merged = to_be_merged.merge(preds)
            merged.to_json(f'{model_out}/topic_preds_{run}.jsonl',
                           orient='records', 
                           lines=True)

            # Evaluate model
            tlists = ctm.get_topic_lists(10)
            tlists_20 = ctm.get_topic_lists(20)
            _compute_metrics('train', 
                             prepped_train, 
                             tlists, 
                             tlists_20,
                             scores,
                             ctm)
            _compute_metrics('val', 
                             prepped_val, 
                             tlists, 
                             tlists_20,
                             scores,
                             ctm)
            _compute_metrics('test', 
                             prepped_test, 
                             tlists, 
                             tlists_20,
                             scores,
                             ctm) 
            scores['run'] = run
            score_list.append(scores)

            # Save model
            ctm.save(models_dir='models/topic', final=True, run=run)
            _save_results(score_list)

    
if __name__=="__main__":
    main()
