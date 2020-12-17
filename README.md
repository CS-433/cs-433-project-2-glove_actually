# Project Text Sentiment Classification

This project was developed as part of the EPFL Machine Learning course (2020).

## Authors
- Marie Biolková
- Sena Necla Cetin
- Robert Pieniuta

## Summary
This repository contains code used for building a classifier for text sentiment analysis. The task was performed on a large corpus of tweets where the goal was to determine whether the tweet contained a positive or negative smiley (before it was removed) from the remaining text. More information about the challenge and the data can be found [here](https://www.aicrowd.com/challenges/epfl-ml-text-classification).

## File structure 
```
.
├── README.md
├── __init__.py
├── data
|   ├── preprocessed_tweets.txt
|   ├── preprocessed_tweets_full.txt
|   ├── preprocessed_tweets_test.txt
|   ├── test_data.txt
|   ├── train_neg.txt
|   ├── train_neg_full.txt
|   ├── train_pos.txt
|   ├── train_pos_full.txt
|   ├── weights_gru.pt
|   └── weights_lstm.pt
├── notebooks
│   ├── bow-tfidf-baselines.ipynb
│   ├── eda.ipynb
│   ├── fasttext.ipynb
│   ├── glove_base.ipynb
│   └── test-preprocessing.ipynb
└── src
    ├── __init__.py
    ├── consts.py
    ├── ft_helpers.py
    ├── get_embeddings.py
    ├── glove
    │   ├── build_vocab.sh
    │   ├── consts_glove.py
    │   ├── cooc.py
    │   ├── cut_vocab.sh
    |   ├── embeddings.txt
    │   ├── glove_solution.py
    │   ├── pickle_vocab.py
    │   └── tmp
    │       ├── cooc.pkl
    │       ├── vocab.pkl
    │       ├── vocab_cut.txt
    │       └── vocab_full.txt
    ├── load.py
    ├── predict_helpers.py
    ├── preprocessing.py
    ├── representations.py
    ├── rnn.py
    ├── rnn_classifier.py
    └── run.py
```

### File description

- `reprocessed_tweets.txt`, `preprocessed_tweets_full.txt`, `preprocessed_tweets_test.txt`: tweets from the development set, full dataset and test set respectivelt which have been pre-processed
- `test_data.txt`: unlabelled tweets to be predicted
- `train_neg.txt`, `train_neg_full.txt`: development and full set of negative tweets
- `train_pos.txt`, `train_pos_full.txt`: development and full set of positive tweets
- `weights_gru.pt`, `weights_lstm.pt`: weights of the best GRU and LSTM model 
-  `bow-tfidf-baselines.ipynb`: code for exploration and tuning of baselines with Tf-Idf and Bag-of-Words
- `eda.ipynb`: exploratory data analysis
- `fasttext.ipynb`: exploration and tuning of fastText
- `glove_base.ipynb`: code for exploration and tuning of baselines using GloVe embeddings
- `test-preprocessing.ipynb`: test file to check whether preprocessing was done correctly
- `consts.py`,`const_glove.py` : contain paths to files used
- `ft_helpers.py`: helper files for fastText training
- `get_embeddings.py`: executing this script from the command line will train GloVe embeddings on the preprocessed dataset 
- `build_vocab.sh`, `cooc.py`, `cut_vocab.sh`, `pickle_vocab.py`, `glove_solution.py`: scripts for training GloVe embeddings;produce the `embeddings.txt` once executed
- `cooc.pkl`, `vocab.pkl`, `vocab_cut.txt`, ` vocab_full.txt`: intermediate files for training GloVe embeddings
-  `load.py`: helper functions for loading datasets and outputing predictions
- `predict_helpers.py`: helper functions for making predictions for the best model
- `preprocessing.py`: methods for preprocessing
- `representations.py`: methods for generating and mapping GloVe embeddings
- `rnn.py`: methods for training RNNs and predicting their outputs
- `rnn_classifier.py`: defines the recurrent neural network class
- `run.py`: script to produce our best submission


## Requirements
- Python 3
  - `numpy`
  - `pandas`
  - `nltk`
  - `wordcloud`
  - `fasttext`
  - `sklearn`
  - `pytorch`
  - `matplotlib` and `seaborn`
  
## Usage

Place the data in the `data` folder. The data, as well as the embeddings we trained can be downloaded [here](https://drive.google.com/file/d/1YQP_vVieTj4LGfx3lJvpvsEHCVRBfhPf/view?usp=sharing).

In order to generate our final submission file, you have to run : 

```
cd src
python run.py
```

This will generate the `src/submission.csv` file.

## Results
Our best model is an ensemble of fastText, LSTM and GRU classifiers. It yielded a classification accuracy of 88.6% on AIcrowd (and an F1-score of 88%).

Please note that since it is not possible to set a seed in fastText, the outputs may vary slightly.

For more details, please read the `report.pdf` file.
