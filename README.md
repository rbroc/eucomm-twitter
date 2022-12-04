# Dynamic topic modeling of EU-related communication
We use topic modeling to analyse discourse from EU institutions on Twitter, focusing particularly on the European Commission.
We use contextualized topic modeling and compare it with simple models (tweetopic) and qualitative inspection.

Motivation:
- Describing EC identity and communication strategy
- Topic as proxy for identity
- Investigate engagement
- Analyze other aspects of EC commmunication (e.g., emotions and style)

** TL;DR **
- What topics does the European Commission communicate about? How does this change over time? How do they relate with engagement?
- How does the style of the EC compare with other institutions? How does it change over time? How does style relate with engagement?
- How does the sentiment of the EC communication compare with other institutions? How does the sentiment change over time? How does it relate with engagement?
- What is the best predictor of engagement? What are the most important features? How do they contribute to the model?

To do:
- Check XGB results and run transformers [in progress]
- Compare metrics with other data  [TBD]
- Public favorability analysis per topic [TBD]
- Tidy up plots, lit review, and intro [TBD]
- Paraphrase example [TBD]


### Metrics
#### From Benoit et al. (2019)
Best metrics
- Flesch Reading Ease only
- Mean sentence length + Mean word syllables
- Frequency, mean sentence characters, proportion of nouns, mean word characters

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