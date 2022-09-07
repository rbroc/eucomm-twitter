from numpy import save
from transformers import TFAutoModelForMaskedLM, AutoTokenizer
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling
from transformers import create_optimizer
import tensorflow as tf
import math
import pandas as pd
import wandb

wandb.login()
wandb.init(entity="rbroc", project="eu-twitter")

class Pretrainer:
    """ Helper class for MLM pretraining
    Parameters
    ----------
    model_checkpoint: str,
        Name of huggingface model checkpoint.
    df: pd.DataFrame,
        Full input dataframe.
    train_prop: float, default 0.9,
        Proportion of the training set used for training.
    chunk_size: int, default 50,
        Number of tokens in training chunks.
    mlm_prob: float, default .15,
        Proportion of tokens masked for MLM training.
    n_epochs: int, default 100,
        Training epochs
    lr: float, default 2e-5,
        Initial learning rate
    es_patience: int, default 5,
        Early stopping patience
    """

    def __init__(self, 
                 model_checkpoint: str, 
                 df: pd.DataFrame, 
                 train_prop: float =.9, 
                 chunk_size: int = 50, 
                 mlm_prob: float = .15,
                 n_epochs: int = 100, 
                 lr: float = 2e-5,
                 es_patience: int = 5):
        self.name = f'{model_checkpoint}-finetuned-lr_{lr}-chunks_{chunk_size}'
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = TFAutoModelForMaskedLM.from_pretrained(model_checkpoint)
        self.chunk_size = chunk_size
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, 
                                                             mlm_probability=mlm_prob)
        train_d, eval_d, test_d = self._make_dataset(df, train_prop)
        self.train_ds = train_d
        self.eval_ds = eval_d
        self.test_ds = test_d
        self.n_epochs = n_epochs
        self.lr = lr
        self.es_patience = es_patience


    def _make_dataset(self, df, train_prop):
        train_ds = Dataset.from_pandas(df[df['pretraining_splits']=='train'][['text']], split='train')
        test_ds = Dataset.from_pandas(df[df['pretraining_splits']=='val'][['text']], split='test')
        tokenized_train = train_ds.map(self._tokenizer_function,
                                       batched=True, 
                                       remove_columns=train_ds.features)
        tokenized_test = test_ds.map(self._tokenizer_function,
                                     batched=True, 
                                     remove_columns=test_ds.features)
        train_lm = tokenized_train.map(self._concat_texts, batched=True)
        test_lm = tokenized_test.map(self._concat_texts, batched=True)
        train_size = int(train_prop * train_lm.num_rows)
        test_size = int((1-train_prop) * train_lm.num_rows)
        train_with_splits = train_lm.train_test_split(train_size=train_size, 
                                                      test_size=test_size,
                                                      seed=42)
        tf_train = train_with_splits["train"].to_tf_dataset(
            columns=["input_ids", "attention_mask", "labels"],
            collate_fn=self.data_collator,
            shuffle=True,
            batch_size=32,
            )
        tf_eval = train_with_splits["test"].to_tf_dataset(
            columns=["input_ids", "attention_mask", "labels"],
            collate_fn=self.data_collator,
            shuffle=False,
            batch_size=32,
            )
        tf_test = test_lm.to_tf_dataset(
            columns=["input_ids", "attention_mask", "labels"],
            collate_fn=self.data_collator,
            shuffle=False,
            batch_size=32,
            )
        return tf_train, tf_test, tf_eval


    def _concat_texts(self, examples):
        concat_ex = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concat_ex[list(examples.keys())[0]])
        total_length = (total_length // self.chunk_size) * self.chunk_size
        out = {
            k: [t[i : i + self.chunk_size] 
                for i in range(0, total_length, self.chunk_size)]
            for k, t in concat_ex.items()
        }
        out["labels"] = out["input_ids"].copy()
        return out        

    def _tokenizer_function(self, examples):
        out = self.tokenizer(examples["text"])
        if self.tokenizer.is_fast:
            out["word_ids"] = [out.word_ids(i) for i in range(len(out["input_ids"]))]
        return out


    def _make_optimizer(self):
        num_train_steps = len([t for t in self.train_ds])
        optimizer, _ = create_optimizer(
            init_lr=self.lr,
            num_warmup_steps=num_train_steps,
            num_train_steps=num_train_steps * self.n_epochs,
            weight_decay_rate=0.01,
        )
        wandb.config = {
            "learning_rate": self.lr,
            "epochs": self.n_epochs,
            "batch_size": 32
            }
        return optimizer


    def compile(self):
        optimizer = self._make_optimizer()
        self.model.compile(optimizer=optimizer)


    def fit(self):
        wandb_cb = wandb.keras.WandbCallback(monitor='val_loss',
                                             save_model=False,
                                             log_weights=True,
                                             log_evaluation=True,
                                             validation_steps=5)
        es_cb = tf.keras.callbacks.EarlyStopping(patience=self.es_patience)

        train_loss_pre = self.model.evaluate(self.train_ds)
        eval_loss_pre = self.model.evaluate(self.eval_ds)
        test_loss_pre = self.model.evaluate(self.test_ds)
        print(f"Pre-training perplexity: {math.exp(eval_loss_pre):.2f}")
        
        self.model.fit(self.train_ds, validation_data=self.eval_ds,
                       epochs=self.n_epochs,
                       callbacks=[es_cb, wandb_cb]
                       )
        train_loss_post = self.model.evaluate(self.train_ds)
        eval_loss_post = self.model.evaluate(self.eval_ds)
        test_loss_post = self.model.evaluate(self.test_ds)
        print(f"Post-training perplexity: {math.exp(eval_loss_post):.2f}")

        results = {
            'name': self.name,
            'pre_train_perplexity': math.exp(train_loss_pre),
            'pre_val_perplexity': math.exp(eval_loss_pre), 
            'pre_test_perplexity': math.exp(test_loss_pre), 
            'post_train_perplexity': math.exp(train_loss_post),
            'post_val_perplexity': math.exp(eval_loss_post), 
            'post_test_perplexity': math.exp(test_loss_post)
            }

        return results


    def evaluate(self, split='test'):
        ds = getattr(self, f'{split}_ds')
        eval_loss = self.model.evaluate(ds)
        print(f"Perplexity on {split}: {math.exp(eval_loss):.2f}")


    def save(self, name=None):
        if name is None:
            name = self.name
        self.model.save_pretrained(name)
    
