"""
Train a Multinomial Naive Bayes model and a SimpleRNN model on the
Twitter sentiment dataset, and save both to the models/ directory.

Usage:
    python train.py

Expects the following files to already be in ./data/:
    data/twitter_training.csv
    data/twitter_validation.csv

(Get them from Kaggle: "Twitter Entity Sentiment Analysis" dataset.)
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Embedding, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from preprocess import clean_text

DATA_DIR = "data"
MODEL_DIR = "models"
COLUMN_NAMES = ["id", "entity", "sentiment", "text"]

VOCAB_SIZE = 20000
MAX_LEN = 40
EMBED_DIM = 128
RNN_UNITS = 128


def prepare_dataset(train_path, val_path):
    train_df = pd.read_csv(train_path, header=None, names=COLUMN_NAMES)
    val_df = pd.read_csv(val_path, header=None, names=COLUMN_NAMES)

    for df in (train_df, val_df):
        df.dropna(subset=["text"], inplace=True)
        df["clean_text"] = df["text"].apply(clean_text)
        df.drop_duplicates(subset=["clean_text"], inplace=True)
        df.query("clean_text != ''", inplace=True)
        df.reset_index(drop=True, inplace=True)

    return train_df, val_df


def train_naive_bayes(train_df, val_df, y_train, y_val, label_encoder):
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(train_df["clean_text"])
    X_val_tfidf = vectorizer.transform(val_df["clean_text"])

    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)

    nb_pred = nb_model.predict(X_val_tfidf)
    nb_accuracy = accuracy_score(y_val, nb_pred)

    print(f"Multinomial Naive Bayes validation accuracy: {nb_accuracy:.4f}")
    print(classification_report(y_val, nb_pred, target_names=label_encoder.classes_))

    joblib.dump(nb_model, f"{MODEL_DIR}/naive_bayes_model.joblib")
    joblib.dump(vectorizer, f"{MODEL_DIR}/tfidf_vectorizer.joblib")

    with open(f"{MODEL_DIR}/nb_accuracy.json", "w") as f:
        json.dump({"model": "Multinomial NB", "accuracy": nb_accuracy}, f)

    return nb_accuracy


def train_rnn(train_df, val_df, y_train, y_val, label_encoder):
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["clean_text"])

    X_train_seq = pad_sequences(
        tokenizer.texts_to_sequences(train_df["clean_text"]),
        maxlen=MAX_LEN, padding="post", truncating="post",
    )
    X_val_seq = pad_sequences(
        tokenizer.texts_to_sequences(val_df["clean_text"]),
        maxlen=MAX_LEN, padding="post", truncating="post",
    )

    num_classes = len(label_encoder.classes_)

    rnn_model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM, input_length=MAX_LEN),
        SimpleRNN(RNN_UNITS, return_sequences=True),
        Dropout(0.3),
        SimpleRNN(64),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])

    rnn_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    rnn_model.summary()

    early_stop = EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)

    rnn_model.fit(
        X_train_seq, y_train,
        validation_data=(X_val_seq, y_val),
        epochs=15,
        batch_size=128,
        callbacks=[early_stop],
        verbose=2,
    )

    rnn_pred = np.argmax(rnn_model.predict(X_val_seq, verbose=0), axis=1)
    rnn_accuracy = accuracy_score(y_val, rnn_pred)
    print(f"SimpleRNN validation accuracy: {rnn_accuracy:.4f}")

    rnn_model.save(f"{MODEL_DIR}/rnn_model.keras")
    joblib.dump(tokenizer, f"{MODEL_DIR}/rnn_tokenizer.joblib")

    with open(f"{MODEL_DIR}/rnn_accuracy.json", "w") as f:
        json.dump({"model": "SimpleRNN", "accuracy": rnn_accuracy}, f)

    return rnn_accuracy


def plot_comparison(nb_accuracy, rnn_accuracy, out_path="outputs/accuracy_comparison.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    names = ["Multinomial NB", "SimpleRNN"]
    scores = [nb_accuracy * 100, rnn_accuracy * 100]

    plt.figure(figsize=(6, 5))
    bars = plt.bar(names, scores, color=["#4C72B0", "#DD8452"])
    plt.ylabel("Validation Accuracy (%)")
    plt.title("ML vs DL Model Accuracy Comparison")
    plt.ylim(0, 100)
    plt.axhline(80, color="gray", linestyle="--", linewidth=1, label="80% target")

    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, score + 1.5,
                  f"{score:.2f}%", ha="center", fontweight="bold")

    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_path = os.path.join(DATA_DIR, "twitter_training.csv")
    val_path = os.path.join(DATA_DIR, "twitter_validation.csv")

    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        raise FileNotFoundError(
            f"Expected {train_path} and {val_path} to exist. "
            "Download the Twitter Entity Sentiment Analysis dataset from Kaggle "
            "and place the two CSV files in the data/ directory."
        )

    train_df, val_df = prepare_dataset(train_path, val_path)
    print("Training rows:", len(train_df))
    print("Validation rows:", len(val_df))

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["sentiment"])
    y_val = label_encoder.transform(val_df["sentiment"])
    joblib.dump(label_encoder, f"{MODEL_DIR}/label_encoder.joblib")

    nb_accuracy = train_naive_bayes(train_df, val_df, y_train, y_val, label_encoder)
    rnn_accuracy = train_rnn(train_df, val_df, y_train, y_val, label_encoder)

    plot_comparison(nb_accuracy, rnn_accuracy)
    print("Done. Models saved to", MODEL_DIR)


if __name__ == "__main__":
    main()
