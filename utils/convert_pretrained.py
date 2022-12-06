import glob
from sentence_transformers import SentenceTransformer, models
from transformers import AutoModel
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model_paths = glob.glob('models/pretrained/*')
model_paths = list(set(model_paths) - set(['models/pretrained/legacy']))
base_models = ['sentence-transformers/distiluse-base-multilingual-cased-v1',
               'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
               'distilbert-base-uncased',
               'distilbert-base-uncased-finetuned-sst-2-english',
               'cardiffnlp/tweet-topic-21-multi']
emo_models = []

def main():
    for model_name in base_models: # model_paths + base_models or emo_models
        if 'cardiffnlp/' in model_name or 'distiluse' in model_name or 'paraphrase' in model_name:
            model_id = '/'.join(model_name.split('/')[-2:])
        else:
            model_id = model_name.split('/')[-1]
        if 'distilbert' in model_id:
            tok = '-'.join(model_id.split('-')[:3])
        else:
            tok = 'cardiffnlp/tweet-topic-21-multi'
        if 'pretrained' in model_id:
            AutoModel.from_pretrained(model_name, 
                                      from_tf=True).save_pretrained(model_name)
        we_model = models.Transformer(model_name, 
                                      max_seq_length=768,
                                      tokenizer_name_or_path=tok)
        pooling = models.Pooling(we_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[we_model, pooling])
        model.save(f'models/sent_transformers/{model_id.replace("/","-")}')
        
if __name__=='__main__':
    main()
    