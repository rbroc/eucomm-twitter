from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer, Trainer
from transformers.trainer_utils import speed_metrics 
import torch
from torch import nn
import tensorflow.keras.backend as K
from torch.utils.data import Dataset
import time
from torchmetrics import TweedieDevianceScore
from typing import Dict, List, Optional
import math


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def make_dataset(tknzr, metric, split='train'):
    ''' Make dataset from transcripts and train / val ids 
    Args:
        tknzr: pretrained tokenizer name
    '''
    DPATH = Path('processed') / 'post_topic_tweets_style_and_sent.jsonl'
    data = pd.read_json(DPATH, orient='records', lines=True)
    data = data[data['topic_split']==split]
    txt, lab = zip(*data[['text', metric]].to_records(index=False))
    tokenizer = AutoTokenizer.from_pretrained(tknzr)
    enc = tokenizer(list(txt), truncation=True, padding=True)
    dataset = TextDataset(enc, lab)
    return dataset, txt
    
def make_trainer(tweedie_p, **kwargs):
    ''' Create trainer with custom loss and metrics loop '''
    
    class TextTrainer(Trainer):

        def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
        ) -> Dict[str, float]:
            # memory metrics - must set up as early as possible
            self._memory_tracker.start()

            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            start_time = time.time()

            eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                prediction_loss_only=True if self.compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

            total_batch_size = self.args.eval_batch_size * self.args.world_size
            output.metrics.update(
                speed_metrics(
                    metric_key_prefix,
                    start_time,
                    num_samples=output.num_samples,
                    num_steps=math.ceil(output.num_samples / total_batch_size),
                )
            )
            self.log(output.metrics)
            self.control = self.callback_handler.on_evaluate(self.args, 
                                                             self.state,
                                                             self.control, 
                                                             output.metrics)
            self._memory_tracker.stop_and_update_metrics(output.metrics)
            return output.metrics
        
        def compute_loss(self, model, inputs, return_outputs=False):
            relu = nn.ReLU()
            loss_fn = TweedieDevianceScore(power=tweedie_p)
            labels = inputs.get("labels").squeeze()
            outputs = model(**inputs)
            logits = outputs.get('logits').squeeze()
            logits = relu(logits)
            loss = loss_fn(logits, labels) # or MSE loss
            return (loss, outputs) if return_outputs else loss

    return TextTrainer(**kwargs)


def _read_tweets(fs, metrics, fields):
    processed_tws = []
    for f in fs:
        tws = json.load(open(f))['data']
        for i in range(len(tws)):
            item = {k: tws[i][k] for k in fields}
            item.update({k: tws[i]['public_metrics'][k] for k in metrics})
            item.update({'created_at': tws[i]['created_at'][:10]})
            tws[i] = item
        processed_tws += tws
        df = pd.DataFrame(processed_tws)
        df['created_at'] = pd.to_datetime(df['created_at'], 
                                          infer_datetime_format=True)
    return df


def language_detection(s):
    try:
        return detect(s)
    except:
        return 'unk'
    
    
def _preprocessing(df):
    df['is_retweet'] = np.where(df['text'].str.startswith('RT'), 1, 0)
    df['is_mention'] = np.where(df['text'].str.startswith('@'), 1, 0)
    df['text'] = df['text'].str.replace(r'http.*', '', regex=True)
    df = df[df['text'].str.len() > 0]
    df['lang_detected'] = df['text'].apply(language_detection)
    return df