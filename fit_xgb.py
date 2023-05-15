import pandas as pd
import json
import numpy as np
import shap
from xgboost import XGBRegressor
import pickle as pkl
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (r2_score, 
                             mean_absolute_error,
                             mean_squared_error)
import argparse
from src.colnames import (topic_col, 
                          style_col)


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
              'tweedie_variance_power': [1.01, 1.3, 1.6, 1.8, 1.99]
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
    ofile = logpath / f'pred_{split}.pkl'
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
    data = pd.read_json(str(Path('data') / 'topic' / 'preds_reduced.jsonl'),
                        orient='records', lines=True)
    data = data[data['entity']=='EU_Commission']
    data = data.rename(dict(zip(topic_col, new_topic_col)), axis=1)
    
    # Set up data
    train_data = data[data['topic_split']=='train']
    val_data = data[data['topic_split']=='val']
    test_data = data[data['topic_split']=='test']

    cols = topic_col + style_col
    train_X = train_data[cols].fillna(0).values
    val_X = val_data[cols].fillna(0).values
    test_X = test_data[cols].fillna(0).values
    
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
                              cv=5,
                              verbose=2,
                              return_train_score=True,
                              refit=False,
                              n_iter=1000)
    
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
    model_name = f'{model_type}'
    model_path = logpath / f'model.pkl'
    shap_path = logpath / f'shap.pkl'

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
    logpath = Path('logs') / 'engagement' / 'results'
    logpath.mkdir(parents=True, exist_ok=True)
    perfpath = Path('logs') / 'engagement' / 'perf.json'
    pm = list(zip([logpath] * 1,
                   [args.out_metric] * 1,
                   ['combined'],
                   [None] * 1,
                   [True] * 1,
                   [args.early_stopping] * 1))
    results = [fit_predict(*p) for p in pm]
    try:
        old_results = json.load(open(str(perfpath), 'rb'))
    except:
        old_results = []
    old_results += results
    with open(str(perfpath), 'w') as of:
        of.write(json.dumps(old_results))
    
