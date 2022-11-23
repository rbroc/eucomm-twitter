from pathlib import Path
from transformers import (TrainingArguments,
                          AutoModelForSequenceClassification,
                          EarlyStoppingCallback)
from utils import make_dataset
import numpy as np
import pandas as pd
import json
import argparse
from sklearn.metrics import (r2_score, 
                             mean_absolute_error,
                             mean_squared_error)
import torch
from transformers import EarlyStoppingCallback

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument('--metric', type=str, default=None)
parser.add_argument('--model-id', type=str, default=None)
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--train-examples-per-device', type=int, default=8)
parser.add_argument('--eval-examples-per-device', type=int, default=8)
parser.add_argument('--learning-rate', type=float, default=5e-5)
parser.add_argument('--warmup-steps', type=int, default=500)
parser.add_argument('--weight-decay', type=float, default=0.001)
parser.add_argument('--logging-steps', type=int, default=100)
parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
parser.add_argument('--early-stopping-patience', type=int, default=10)
parser.add_argument('--freeze-layers', type=int, default=1)
parser.add_argument('--tweedie-p', type=float, default=1.5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRED_COLUMNS = ['trial_id', 'label', 'prediction', 'model_name', 'split']
OUTPUT_PATH = Path('logs') / 'transformers'


def compute_metrics(pred):
    preds, labels = pred
    mse = mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds)   
    return {"mse": mse, 
            "mae": mae, 
            "r2": r2}


# Training module
def _make_trainer(model_id,
                  checkpoint, 
                  train_dataset, val_dataset,
                  epochs, 
                  train_examples_per_device, 
                  eval_examples_per_device,
                  learning_rate,
                  warmup_steps, 
                  weight_decay,
                  logging_steps,
                  gradient_accumulation_steps,
                  early_stopping_patience,
                  freeze_layers,
                  tweedie_p,
                  metric):
    ''' Train model 
    Args:
        model_id: unique model id
        checkpoint: path to model checkpoint or pretrained from model hub
        train_dataset: training dataset
        val_dataset: validation dataset
        epochs: training epochs
        train_examples_per_device: examples per device at training
        eval_examples_per_device: examples per device at eval
        warmup_steps: nr optimizer warmup steps
        weight_decay: weight decay for Adam optimizer
        logging_steps: how often we want to log metrics
    '''

    # Make directories
    fstr = 'freeze' if freeze_layers == 1 else 'nofreeze'
    bstr = f'batch-{train_examples_per_device}'
    mid = f'{model_id}_lr-{learning_rate}_wdecay-{weight_decay}_wsteps-{warmup_steps}_{fstr}'
    mid += f'_{bstr}_tweediep-{tweedie_p}'
    logpath = Path('logs') / 'transformers' / metric / mid 
    respath = Path('models') / 'transformers' / metric / mid
    logpath.mkdir(exist_ok=True, parents=True)
    respath.mkdir(exist_ok=True, parents=True)

    
    # Set up trainer
    training_args = TrainingArguments(
        output_dir=logpath,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_examples_per_device,
        per_device_eval_batch_size=eval_examples_per_device,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        logging_dir=str(logpath),
        evaluation_strategy='steps',
        save_strategy='steps',
        logging_strategy='steps',
        logging_steps=logging_steps,
        save_steps=logging_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        run_name=mid,
        load_best_model_at_end=True,
        #fp16=True,
        save_total_limit=1
    )

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint,
                                                               num_labels=1
                                                               problem_type='regression').to(device)
    if freeze_layers==1:
        modules = [model.base_model.embeddings]
        try:
            modules += model.base_model.encoder.layer
        except:
            modules += model.base_model.transformer.layer
        for module in modules:
            for p in module.parameters():
                p.requires_grad = False
    
    trainer = make_trainer(tweedie_p, 
                           model=model,
                           args=training_args,
                           train_dataset=train_dataset,
                           eval_dataset=val_dataset,
                           compute_metrics=compute_metrics,
                           callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)])
    return trainer, respath, mid


def evaluate(trainer, dataset, split, model_id, metric, odict, txt):
    # Get predictions
    outs = trainer.predict(test_dataset=dataset)
    predictions = torch.max(torch.tensor(outs.predictions), axis=1)
    # Extract metrics
    labels = [l for l in outs.label_ids]
    mid = f'{metric}_{model_id}'
    model_names = [mid] * len(predictions)
    # Hyperparameters
    splits = [split] * len(predictions)
    scores = outs.predictions
    # Make dataframe
    ### TODO: Add text (trial id)
    pdf = pd.DataFrame(zip(txt,
                           labels, 
                           predictions, 
                           model_names, splits),
                           columns=PRED_COLUMNS)
    # Make output path
    output_path = OUTPUT_PATH / metric
    output_path.mkdir(parents=True, exist_ok=True)
    outfile = str(output_path / f'pred_{mid}_{split}.pkl')
    # Save
    pdf.to_pickle(outfile) 
    # Also log absolute metrics
    for m in ['mae', 'mse', 'r2']:
        odict[f'{split}_{m}'] = outs.metrics[f'test_{m}']
    return odict


# Execute
if __name__=='__main__':
    args = parser.parse_args()
    train, val, test = (make_dataset(args.checkpoint, args.metric, s) 
                                 for s in ['train', 'validation', 'test'])
    train_ds, train_txt = train
    val_ds, val_txt = val
    test_ds, test_txt = test
    trainer, respath, mid = _make_trainer(args.model_id,
                                          args.checkpoint,
                                          train_ds, 
                                          val_ds, 
                                          args.epochs, 
                                          args.train_examples_per_device, 
                                          args.eval_examples_per_device,
                                          args.learning_rate,
                                          args.warmup_steps,
                                          args.weight_decay,
                                          args.logging_steps,
                                          args.gradient_accumulation_steps,
                                          args.early_stopping_patience,
                                          args.freeze_layers,
                                          args.tweedie_p,
                                          args.metric)
    trainer.train()
    trainer.save_model(str(respath))
    
    metrics_path = OUTPUT_PATH / f'{args.metric}.json'
    results = []
    if metrics_path.is_file():
        results = json.load(open(str(metrics_path)))
    odict = {}
    odict['model'] = f'{args.metric}_{mid}'
    for ds, txt, spl in zip([train_ds, val_ds, test_ds],
                            [train_txt, val_txt, test_txt],
                            ['train', 'val', 'test']):
        odict = evaluate(trainer, ds, spl, mid, args.metric, odict, txt)
    results.append(odict)
    
    with open(str(metrics_path), 'w') as of:
        of.write(json.dumps(results))

    