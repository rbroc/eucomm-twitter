import pandas as pd
import numpy as np
import spacy
import emojis
from pliers.extractors import PredefinedDictionaryExtractor, merge_results
from pliers.stimuli import ComplexTextStim
import textdescriptives as td
from transformers import pipeline
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--entity', type=str, default='10DowningStreet')


# Set up classifier pipeline
def get_emo(lst):
    columns = [l['label'] for l in lst]
    vals = [l['score'] for l in lst]
    return pd.DataFrame([vals], columns=columns)

classifier = pipeline("text-classification", 
                      model='cardiffnlp/twitter-roberta-base-sentiment',
                      return_all_scores=True, device=0,
                      function_to_apply='softmax')


# Set up style extractors
nlp = spacy.load("en_core_web_md")
nlp.add_pipe("textdescriptives")
ext = PredefinedDictionaryExtractor(variables=['subtlexusfrequency'])
target_cols = ['PredefinedDictionaryExtractor#subtlexusfrequency_Lg10WF',
               'PredefinedDictionaryExtractor#subtlexusfrequency_SUBTLWF', 
               'PredefinedDictionaryExtractor#subtlexusfrequency_Zipf-value']


def main(entity, style_subset_only=True):
    ''' Main function '''
    if style_subset_only is False:
        df = pd.read_json(f'data/preprocessed/{entity}.jsonl',
                          orient='records', 
                          lines=True)

        # Extract styles
        dfs = [] 
        for i, t in enumerate(df['text']):
            if i % 1000 == 0:
                print(i)
            try: 
                out = nlp(t)
                extracted = td.extract_df(out)
                for i, c in enumerate(list(out.doc._.pos_proportions.keys())):
                    extracted[c] = list(out.doc._.pos_proportions.values())[i]
                results = merge_results(ext.transform(ComplexTextStim(text=t)))
                for tc in target_cols:
                    freq = results[tc].astype(float).mean()
                    extracted[tc.split('#')[1]] = freq
                dfs.append(extracted) 
            except:
                dfs.append(pd.DataFrame([[np.nan]*len(extracted.columns)],
                                        columns=extracted.columns))

        df = pd.concat([df, 
                        pd.concat(dfs).reset_index(drop=True).drop('text',
                                                                   axis=1)], axis=1)

        # Extract emotions
        emos = pd.concat(df['text'].apply(lambda x: 
                                          get_emo(classifier(x)[0])).tolist())
        df = pd.concat([df, emos.reset_index(drop=True)], axis=1)
        df.rename({'LABEL_0': 'negative_sentiment', 
                   'LABEL_1': 'neutral_sentiment',
                   'LABEL_2': 'positive_sentiment'}, 
                  axis=1,
                  inplace=True)
    else:
        df = pd.read_json(f'data/derivatives/{entity}_annotated.jsonl', 
                          orient='records', lines=True)
    
    # Compute Rauh metrics
    df['rauh_frequency'] = df['subtlexusfrequency_Lg10WF']
    df['rauh_verb-to-noun'] = df.fillna(0)['pos_prop_VERB'] / df.fillna(0)['pos_prop_NOUN']
    df['rauh_readability'] = df['flesch_kincaid_grade']

    # Compute metrics from Benoit et al., 2019
    df['benoit_readability'] = df['flesch_reading_ease']
    df['benoit_overall-length'] = df['n_tokens']
    df['benoit_sentence-length-words'] = df['sentence_length_mean']
    df['benoit_sentence-length-characters'] = df['n_characters'] / df['n_sentences']
    df['benoit_word-length-syllables'] = df['syllables_per_token_mean']
    df['benoit_word-length-characters'] = df['token_length_mean']
    df['benoit_prop-noun'] = df['pos_prop_NOUN']
    
    # Compute alphanumeric metrics: mentions, hashtags, emojis, is link
    df['n_hashtag'] = df['text'].replace('[^#]', '').str.len()
    df['n_mentions'] = df['text'].replace('[^@]', '').str.len()
    df['n_emojis'] = df['text'].apply(lambda x: emojis.count(x))
    df['emojis'] = df['text'].apply(lambda x: emojis.get(x))
    
    # Save
    df.to_json(f'data/derivatives/{entity}_annotated.jsonl', 
               orient='records', lines=True)

if __name__=='__main__':
    args = parser.parse_args()
    main(args.entity)


