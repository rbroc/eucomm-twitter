# Dynamic topic modeling of EU-related communication
We use topic modeling to analyse discourse from EU institutions on Twitter, focusing particularly on the European Commission.
We use contextualized topic modeling and compare it with simple models (tweetopic) and qualitative inspection.

Motivation:
- Describing dimensions of EU top-down identity as conveyed through online platforms
- Topics as a proxy for focus areas
- Reception in the public: how much and which kind of reactions do different topics convey?

Outline:
- Tuning and description of topic structure for EU commission data
- Describing and predicting engagement as a function of topics
- Describing sentiment of responses as a function of topics

To do:
- Run transformer-based engagement [in progress]
- Run XGBoost with negative binomials
    - Raw style indicators, PCA, informed from Rauh et al. & Benoit
    - Extract aggregate measure
    - Extract number of positive comments as a metric!
- Disentangle engagement from positive responses ("public favorability")
- Tidy up plots
- Lit review and intro
- Paraphrase example



### Metrics
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
- symbol_\#_2_word_ratio [exists]
- Verbs / Nouns [to be computed]
- Readability indices [exist]