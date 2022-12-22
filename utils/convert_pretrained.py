import glob
from sentence_transformers import SentenceTransformer, models
from transformers import AutoModel
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

base_models = ['sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
               'distilbert-base-uncased',
               'distilbert-base-uncased-finetuned-sst-2-english',
               'cardiffnlp/tweet-topic-21-multi']

def main():
    model_paths = glob.glob('../models/pretraining/*/*')
    for model_name in model_paths + base_models:
        if model_name in base_models:
            eucomm = 'pretrained'
        else:
            eucomm = model_name.split('/')[-2]
        if 'cardiffnlp/' in model_name or 'distiluse' in model_name or 'paraphrase' in model_name:
            model_id = '/'.join(model_name.split('/')[-2:])
        else:
            model_id = model_name.split('/')[-1]
        if 'distilbert' in model_id:
            tok = '-'.join(model_id.split('-')[:3])
        else:
            tok = 'cardiffnlp/tweet-topic-21-multi'
        if model_name in model_paths:
            AutoModel.from_pretrained(model_name, 
                                      from_tf=True).save_pretrained(model_name)
        we_model = models.Transformer(model_name, 
                                      max_seq_length=768,
                                      tokenizer_name_or_path=tok)
        pooling = models.Pooling(we_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[we_model, pooling])
        OUT_PATH = Path('..') / 'models' / 'sent_transformers' / eucomm
        OUT_PATH.mkdir(exist_ok=True, parents=True)
        model.save(str(OUT_PATH / model_id.replace("/","-")))
        
if __name__=='__main__':
    main()
    