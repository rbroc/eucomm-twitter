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
from logs.best_topic_names import topic_col, emotion_col, style_col


# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument('--early-stopping', type=int, default=5)
parser.add_argument('--out-metric', type=str, default=None)


# Parameters for grid search
def _make_estimator_params():
    params = {
              'eta': [.0001, .001, .01, .1],
              'min_child_weight': [5, 10, 50],
              'gamma': [.5, 1., 2.],
              'subsample': [.6, .8],
              'colsample_bytree': [.3, .5, .7],
              'max_depth': [3, 5],
              'reg_alpha' :[0, .1, 5.],
              'reg_lambda': [.1, 1.],
              'n_estimators': [5, 10, 20],
              'tweedie_variance_power': np.arange(1.1,2.1,0.1)
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
    model_base = 'distilbert-base-uncased-finetuned-sst-2-english'
    best_model = model_base + '_vocab-500_bow-499_comp-20_esize-768_batch-64_lr-0.002_epochs-100_act-softplus'
    data = pd.read_json(f'processed/post_topic_tweets_style_and_sent_pca.jsonl',
                        orient='records', lines=True)
    
    # Set up data
    train_data = data[data['topic_split']=='train']
    val_data = data[data['topic_split']=='val']
    test_data = data[data['topic_split']=='test']
    if model_type == 'topic':
        train_X = train_data[topic_col].values
        val_X = val_data[topic_col].values
        test_X = test_data[topic_col].values
    elif model_type == 'emotions':
        train_X = train_data[emotion_col].values
        val_X = val_data[emotion_col].values
        test_X = test_data[emotion_col].values
    elif model_type == 'style':
        train_X = train_data[style_col].values
        val_X = val_data[style_col].values
        test_X = test_data[style_col].values
    elif model_type == 'combined':
        cols = topic_col+emotion_col+style_col
        train_X = train_data[cols].values
        val_X = val_data[cols].values
        test_X = test_data[cols].values
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
    train_y = train_data[out_metric].values
    val_y = val_data[out_metric].values
    test_y = test_data[out_metric].values
    train_ex = train_data['text'].tolist()
    val_ex = val_data['text'].tolist()
    test_ex = val_data['text'].tolist()
    
    # Pass explicit validation set
    train_idx = np.full((train_X.shape[0],), -1, dtype=int)
    val_idx = np.full((val_X.shape[0],), 0, dtype=int)
    fold_idx = np.append(train_idx, val_idx)
    ps = PredefinedSplit(fold_idx)

    # Set up XGBoost
    objective = 'reg:tweedie'
    eval_metric = f'rmse' # 'rmse'
    est_class = XGBRegressor(objective=objective,
                             eval_metric=eval_metric,
                             n_jobs=20, 
                             #tree_method='gpu_hist', 
                             #gpu_id=1
                            )
    grid = RandomizedSearchCV(estimator=est_class,
                              param_distributions=_make_estimator_params(),
                              cv=ps,
                              verbose=2,
                              return_train_score=True,
                              refit=False,
                              n_iter=1000)
    grid.fit(np.concatenate([train_X,val_X], axis=0),
             np.concatenate([train_y,val_y], axis=0),
             verbose=False)
    
    # Get best model and fit on training data only
    model = XGBRegressor(**grid.best_params_, 
                         objective=objective, 
                         eval_metric=eval_metric,
                         n_jobs=20)
    model.fit(train_X, train_y, 
              eval_set=[(val_X, val_y)],
              early_stopping_rounds=5,
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
    shap_values = explainer(test_X) #.shap_values
    pkl.dump(shap_values, open(str(shap_path), "wb"))
    
    # Print results
    print(out_metric)
    outs.update({'model': model_name})
    return outs
    

if __name__=='__main__':
    args = parser.parse_args()
    logpath = Path('logs') / 'metrics' 
    logpath = logpath / args.out_metric
    logpath.mkdir(parents=True, exist_ok=True)
    pm = list(zip([logpath] * 7,
                  [args.out_metric] * 7,
                  ['combined','topic', 'emotions', 'style',
                   'bow', 'bow', 'bow',],
                  [None, None, None, None, 
                   100, 250, 500],
                  [True] * 7,
                  [args.early_stopping] * 7))
    results = [fit_predict(*p) for p in pm]
    with open(str(logpath)+'.json', 'w') as of:
        of.write(json.dumps(results))
    
