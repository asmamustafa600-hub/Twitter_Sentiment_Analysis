# Twitter Sentiment Classifier

A Streamlit app that classifies a tweet as **Positive**, **Negative**, **Neutral**, or
**Irrelevant**, using either a Multinomial Naive Bayes (TF-IDF) model or a SimpleRNN
(Keras) model. Includes the training script used to produce both models.

## Project structure

```
.
├── app.py            # Streamlit app
├── train.py           # Trains both models and saves them to models/
├── preprocess.py       # Shared text-cleaning function (used by both app.py and train.py)
├── requirements.txt
├── data/               # Put twitter_training.csv / twitter_validation.csv here (not committed)
└── models/             # Trained model artifacts land here after running train.py
```

## Setup

```bash
git clone <your-repo-url>
cd <your-repo-name>
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data

This project uses the [Twitter Entity Sentiment Analysis dataset](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
from Kaggle. Download `twitter_training.csv` and `twitter_validation.csv` and place
them in the `data/` folder:

```
data/twitter_training.csv
data/twitter_validation.csv
```

## Train the models

```bash
python train.py
```

This will:
- Clean and dedupe the training/validation data
- Train a Multinomial Naive Bayes classifier on TF-IDF features
- Train a SimpleRNN classifier on tokenized/padded sequences
- Save both models (plus the vectorizer, tokenizer, and label encoder) to `models/`
- Save an accuracy comparison chart to `outputs/accuracy_comparison.png`

## Run the app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Notes

- `models/` is not excluded by `.gitignore` in this template — if your trained
  model files are large (the `.keras` file in particular), consider using
  [Git LFS](https://git-lfs.com/) or excluding `models/` and adding a step to
  your deployment that runs `train.py` first.
- Raw CSV data is excluded from version control via `.gitignore`; download it
  fresh using the Kaggle link above.
