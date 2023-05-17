# A data-driven analysis of the European Commission’s Twitter communication


### Brief description
This repository contains code and data for the paper "From economy to identity: A data-driven analysis of the European Commission’s Twitter communication between 2010 and 2022".

### Data
Due to Twitter policy, we do not share raw text or metrics data accessed through the API. We only share derivatives of the data, including Twitter IDs and extracted metrics, but no text nor engagement metrics. Note that the `data/derivatives` and `data/topic` include derivatives of the data uniquely including extracted features. Some of the files are zipped because their raw versions are too large. Data in the `data/topic` folder are the most complete version of the data, as they include both style and topic annotations.

### Processing steps
- `preprocess.py` performs basic preprocessing steps, described in the paper (raw data are used as input to this, thus input and output of this steps are not available in the `data` repository);
- `annotate.py` extracts metrics used throughout the analyses, especially style metrics;
- `split.py` splits all input files into training, validation and test set;
- `pretrain.py` is used to tune a standard DistilBERT models on our data. The resulting model is one of the models used as basis for contextualized topic modeling;
- `annotate_concreteness.py` extracts concreteness estimates (separate file due to error in the first iteration of data processing). It is only applied post-hoc, after all data have been processed and merged, to avoid recomputing topic annotations;

### Analyses
- `fit_topics.py` fits topic models with a range of parameters. The best topic model is selected after inspection (see `0. Explore Topics.ipynb`);
- `fit_xgb.py` fits the XGBoost models, and logs outputs and evalution outcomes

### Other useful information
- Each notebook in the root of the repository maps onto a section of the paper
- `summaries` and `figs` contain outputs that are used in the paper (tables and figures)
- `src` contains useful functions, used throughout the analysis
- `utils.py` containts utils for visualization, used for plotting time series

### Trained models
Trained topic models are available under `models/topic`. The configuration and training script for our pretrained transformer are shared under `models/pretraining`, and the model is available at: https://huggingface.co/rbroc/twitter-eupol-model/. In the `utils` folder, you can find a script that transforms files (weights and architecture) standard bidirectional transformer encoder architectures into sentence transformers, required for contextualized topic modeling. Note, however, that the standard DistilBERT model performed better than our fine-tuned architecture.
