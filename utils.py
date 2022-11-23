from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer
import torch
from tensorflow import keras as K


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
    DPATH = Path('processed') / 'post_topic_tweets_style_and_emo_pca.jsonl'
    data = pd.read_csv(DPATH/f'{split}.csv')
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
        
        def tweedieloss(y_true, y_pred, tweedie_p):
            dev = 2 * (K.pow(y_true, 2-tweedie_p)/((1-tweedie_p) * (2-tweedie_p)) -
                           y_true * K.pow(y_pred, 1-tweedie_p)/(1-tweedie_p) +
                           K.pow(y_pred, 2-tweedie_p)/(2-tweedie_p))
            return K.mean(dev)
        
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get('logits')
            loss = tweedieloss(labels, outputs, tweedie_p)
            return (loss, outputs) if return_outputs else loss

    return TextTrainer(**kwargs)