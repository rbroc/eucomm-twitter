# Dynamic topic modeling of EU-related communication


#### TODO
- Engagement XGB + refit transformer-based classifier
- Plots & wrap-up

### Style metrics
#### From Benoit et al. (2019)
Best metrics
- Flesch Reading Ease only - OK
- Mean sentence length - OK + Mean word syllables - OK
- Frequency - OK, mean sentence characters - OK, proportion of nouns - OK, mean word characters

**ALL METRICS**
Long words
- Mean characters per word [token_length_mean, excludes alphanumeric, or 'mean_word_length', treats alphanumeric as independent]
- Mean syllables per word - [syllables_per_token_mean, excludes alphanumeric]

Rarity
- Google Books baseline usage (min, mean) + Brown - [SubtlexUS available, to be extracted]

Long Sentence
- Mean characters per sentence [can be computed as n_characters / n_sentences]
- Mean sentence length in words [sentence_length_mean]
- Number of sentences per character [can be computed as n_sentences / n_character]
- Mean sentence length in syllables - [can be computed as syllables_per_token * sentence_length_mean]

Complex content
- Proportion of nouns
- Proportion of verbs
- Proportion of adjectives
- Proportion of adverbs
- Average subordinate clauses (exclude) 

Twitter-specific:
- symbol_\#_2_word_ratio

#### From Rauh, 2022
- Verb-to-noun ratio
- Google book n-gram frequency
- Flesch-Kincaid reading score