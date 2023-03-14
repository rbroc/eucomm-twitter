import pandas as pd
import json
import numpy as np
import shap
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier
import pickle as pkl
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import (GridSearchCV, 
                                     RandomizedSearchCV,
                                     PredefinedSplit)
from sklearn.metrics import (r2_score, 
                             mean_absolute_error,
                             mean_squared_error)

from nltk.corpus import stopwords
import argparse
from src.colnames import (topic_col, emotion_col, 
                          style_col, exclude_col)


# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument('--early-stopping', type=int, default=5)
parser.add_argument('--out-metric', type=str, default='sum_count')


# Parameters for grid search
def _make_estimator_params():
    params = {
              'learning_rate': [2e-5, 2e-3, 2e-2, 2e-1],
              'min_child_weight': [1, 5, 10, 50],
              'gamma': [0, .5, 1., 2.],
              'subsample': [.6, .8, 1],
              'colsample_bytree': [.3, .5, .7, 1],
              'max_depth': [2, 3, 5, 10, 20],
              'reg_alpha' :[0, .1, 5.],
              'reg_lambda': [0, .1, 1.],
              'n_estimators': [1, 5, 10, 30, 50],
              'tweedie_variance_power': [1.01, 1.99, 1.2, 1.4, 1.6, 1.8]
              }
    return params


def _save_predictions(preds, split, labels, exs,
                      logpath, model_name):
    pred_df = pd.DataFrame(zip(exs,
                               labels,
                               preds, 
                               [model_name]*len(exs),
                               [split]*len(exs)), 
                            columns=['trial_id',
                                     'label',
                                     'prediction',
                                     'model_name',
                                     'split'])
    ofile = logpath / f'pred_{model_name}_{split}.pkl'
    pred_df.to_pickle(ofile)


def _save_scores(X, y, logpath, ofile, split, exs, grid, model_name):
    ''' Save scores '''
    preds = grid.best_estimator_.predict(X)
    _save_predictions(preds, split, y, exs, logpath, model_name)
    r2 = r2_score(y, preds)
    mae = mean_absolute_error(y, preds)
    mse = mean_squared_error(y, preds)
    outs = dict(zip([f'{split}_mse', 
                     f'{split}_mae',
                     f'{split}_r2'],
                    [mse, mae, r2]))
    return outs
    

def fit_predict(logpath,
                out_metric,
                model_type,
                n_words=None,
                eval_on_test=True,
                es=10):
    
    # Get the data
    data = pd.read_json(f'data/topic/data.jsonl',
                        orient='records', lines=True)
    data = data[data['entity']=='EU_Commission'] # TODO: fit without this
    data['n_mentions'] = data['text'].replace('[^@]','', regex=True).str.len()
    data['n_hashtag'] = data['text'].replace('[^#]','', regex=True).str.len()
    for c in ['n_hashtag', 'n_mentions', 'n_emojis']:
        data[c] = data[c] / data['benoit_sentence-length-words']
    
    # Set up data
    train_data = data[data['topic_split']=='train']
    val_data = data[data['topic_split']=='val']
    test_data = data[data['topic_split']=='test']
    
    if model_type == 'topic':
        train_X = train_data[topic_col].values
        val_X = val_data[topic_col].values
        test_X = test_data[topic_col].values   
    
    elif model_type == 'sentiment':
        train_X = train_data[emotion_col].values
        val_X = val_data[emotion_col].values
        test_X = test_data[emotion_col].values
        
    elif model_type == 'style_short':
        cols = [c for c in data.columns if 'rauh' in c or 'benoit' in c]
        train_X = train_data[cols].fillna(0).values
        val_X = val_data[cols].fillna(0).values
        test_X = test_data[cols].fillna(0).values
        
    elif model_type == 'style_full': # TODO: Better feature set?
        style_targets = [c for c in data.columns if any(['rauh' in c, 
                                                         'benoit' in c, 
                                                         'alpha_ratio' in c])] + \
                        ['n_hashtag', 'n_mentions', 'is_link', 'n_emojis']
        train_X = train_data[style_targets].fillna(0).values
        val_X = val_data[style_targets].fillna(0).values
        test_X = test_data[style_targets].fillna(0).values
    
    elif model_type == 'combined': # TODO: Better feature set?
        style_targets = [c for c in data.columns if any(['rauh' in c, 
                                                         'benoit' in c, 
                                                         'alpha_ratio' in c])] + \
                        ['n_hashtag', 'n_mentions', 'is_link', 'n_emojis']
        cols = topic_col + emotion_col + style_targets
        train_X = train_data[cols].fillna(0).values
        val_X = val_data[cols].fillna(0).values
        test_X = test_data[cols].fillna(0).values
    
    elif model_type == 'bow':
        stop_words = stopwords.words('english')
        tkpath = str(logpath / f'tokenizer_bow-{n_words}.pkl')
        c_vec = TfidfVectorizer(stop_words=stop_words,
                                max_features=n_words)
        train_X = c_vec.fit_transform(train_data['text'].tolist()).toarray()
        val_X = c_vec.transform(val_data['text'].tolist()).toarray()
        test_X = c_vec.transform(test_data['text'].tolist()).toarray()
        pkl.dump(c_vec, open(tkpath, "wb"))
        
    else:
        raise ValueError(f'{model_type} is not a valid model type')
    
    # Get outcomes
    train_y = train_data[out_metric].values
    val_y = val_data[out_metric].values
    test_y = test_data[out_metric].values
    train_ex = train_data['text'].tolist()
    val_ex = val_data['text'].tolist()
    test_ex = val_data['text'].tolist()

    # Set up XGBoost
    objective = 'reg:tweedie'
    est_class = XGBRegressor(objective=objective,
                             n_jobs=20)
        
    grid = RandomizedSearchCV(estimator=est_class,
                              param_distributions=_make_estimator_params(),
                              cv= 5, #TODO: None,
                              verbose=2,
                              return_train_score=True,
                              refit=False,
                              n_iter=2000)
    
    # Refit
    grid.fit(train_X,
             train_y,
             verbose=False)
    
    # Get best model and fit on training data only
    model = XGBRegressor(**grid.best_params_, 
                         objective=objective, 
                         n_jobs=20)
    model.fit(train_X, train_y, 
              eval_set=[(val_X, val_y)],
              early_stopping_rounds=3,
              verbose=False)

    grid.best_estimator_ = model

    # Make output file names
    ofile_train = logpath / f'train.txt'
    ofile_val = logpath / f'val.txt'
    ofile_test = logpath / f'test.txt'
    result_path = logpath / f'grid.csv'
    model_name = f'{out_metric}_{model_type}_{n_words}'
    model_path = logpath / f'{model_name}.pkl'
    shap_path = logpath / f'shap_{model_name}.pkl'

    # Predict and evaluate on train, val and set
    outs = _save_scores(train_X, train_y, 
                        logpath, 
                        ofile_train, 'train', train_ex, 
                        grid, model_name)
    pkl.dump(grid.best_estimator_, open(str(model_path), "wb"))
    pd.DataFrame(grid.cv_results_).to_csv(str(result_path))
    val_outs = _save_scores(val_X, val_y, 
                            logpath, 
                            ofile_val, 'val', val_ex, 
                            grid, model_name)
    outs.update(val_outs)
    
    # Predict and evaluate on test set
    if eval_on_test:
        test_outs = _save_scores(test_X, test_y, 
                                 logpath,
                                 ofile_test, 'test', test_ex, 
                                 grid, model_name)
        outs.update(test_outs)
        
    # Also store shap values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(test_X)
    pkl.dump(shap_values, open(str(shap_path), "wb"))
    
    # Print results
    print(out_metric)
    outs.update({'model': model_name})
    return outs
    

if __name__=='__main__':
    args = parser.parse_args()
    logpath = Path('logs') / 'engagement' 
    logpath = logpath / args.out_metric
    logpath.mkdir(parents=True, exist_ok=True)
    pm = list(zip([logpath] * 6,
                  [args.out_metric] * 6,
                  ['combined', 
                   'topic', 
                   'sentiment', 
                   'style_short',
                   'style_full',
                   'bow'],
                  [None, None, None, None, 
                   None, 500],
                  [True] * 6,
                  [args.early_stopping] * 6))
    results = [fit_predict(*p) for p in pm]
    try:
        old_results = json.load(open(str(logpath)+'.json', 'rb'))
    except:
        old_results = []
    old_results += results
    with open(str(logpath)+'.json', 'w') as of:
        of.write(json.dumps(old_results))
    
