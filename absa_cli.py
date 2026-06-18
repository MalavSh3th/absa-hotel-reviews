#!/usr/bin/env python3
"""Command-line ABSA tool: trains aspect + sentiment classifiers from a
Review/Aspect/Sentiment CSV and predicts both for new review text.

Usage:
    python absa_cli.py --data data/output.csv \
        --review "The food was great but the service was really slow"

    python absa_cli.py --data data/restaurant_reviews.csv --reviews-file my_reviews.txt --tune

    python absa_cli.py --data data/output.csv   # train + report metrics only
"""

import argparse
import re
import sys

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVC

RF_PARAM_GRID = {
    "max_depth": [100, 110],
    "min_samples_leaf": [1, 2, 3],
    "min_samples_split": [5, 6, 7],
    "n_estimators": [2300, 2400, 2500],
}

SVC_PARAM_GRID = {
    "C": [2, 2.5, 3],
    "gamma": [0.1, 0.2, 0.3],
    "degree": [1, 2, 3],
    "kernel": ["linear", "rbf", "sigmoid"],
}


def ensure_nltk_data():
    for resource in ("stopwords",):
        try:
            stopwords.words("english")
        except LookupError:
            nltk.download(resource)


def clean_review(text, stemmer, all_stopwords):
    text = re.sub("[^a-zA-Z]", " ", str(text))
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in all_stopwords]
    return " ".join(words)


def build_corpus(reviews):
    stemmer = PorterStemmer()
    all_stopwords = set(stopwords.words("english"))
    all_stopwords.discard("not")
    return [clean_review(r, stemmer, all_stopwords) for r in reviews]


def make_model(model_name):
    if model_name == "rf":
        return RandomForestClassifier(random_state=0)
    return SVC(random_state=0)


def fit_model(model_name, tune, X_train, y_train, n_iter, random_state):
    base_model = make_model(model_name)
    if not tune:
        return base_model.fit(X_train, y_train)

    param_grid = RF_PARAM_GRID if model_name == "rf" else SVC_PARAM_GRID
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        scoring="accuracy",
        cv=5,
        n_iter=n_iter,
        random_state=random_state,
    )
    search.fit(X_train, y_train)
    print(f"  best params: {search.best_params_}")
    print(f"  best CV score: {search.best_score_:.4f}")
    return search.best_estimator_


def report(label_name, model, X_test, y_test, plot, plot_path):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n==== {label_name} classifier ====")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    if plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix — {label_name}")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved confusion matrix plot to {plot_path}")

    return acc


def collect_reviews(args):
    reviews = list(args.review or [])
    if args.reviews_file:
        with open(args.reviews_file, encoding="utf-8") as f:
            reviews.extend(line.strip() for line in f if line.strip())
    return reviews


def main():
    parser = argparse.ArgumentParser(description="Train and run an Aspect-Based Sentiment Analysis classifier.")
    parser.add_argument("--data", default="data/output.csv", help="Path to CSV with Review,Aspect,Sentiment columns")
    parser.add_argument("--max-rows", type=int, default=5000, help="Max rows to load from the dataset")
    parser.add_argument("--max-features", type=int, default=151, help="Max Bag-of-Words features")
    parser.add_argument("--model", choices=["svc", "rf"], default="svc", help="Classifier to train")
    parser.add_argument("--tune", action="store_true", help="Run RandomizedSearchCV hyperparameter tuning (slower)")
    parser.add_argument("--n-iter", type=int, default=10, help="RandomizedSearchCV iterations when --tune is set")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--review", action="append", help="Review text to classify. Can be passed multiple times.")
    parser.add_argument("--reviews-file", help="Path to a text file with one review per line to classify")
    parser.add_argument("--plot", action="store_true", help="Save confusion matrix plots as PNG files")
    parser.add_argument("--plot-dir", default=".", help="Directory to save confusion matrix plots into")
    args = parser.parse_args()

    ensure_nltk_data()

    full_df = pd.read_csv(args.data, nrows=args.max_rows)
    for required_col in ("Review", "Aspect", "Sentiment"):
        if required_col not in full_df.columns:
            sys.exit(f"Dataset is missing required column: {required_col}")

    print(f"Loaded {len(full_df)} reviews from {args.data}")
    corpus = build_corpus(full_df["Review"])

    cv = CountVectorizer(max_features=args.max_features)
    X = cv.fit_transform(corpus).toarray()
    y_aspect = full_df["Aspect"].values
    y_sentiment = full_df["Sentiment"].values

    X_train, X_test, y_aspect_train, y_aspect_test, y_sentiment_train, y_sentiment_test = train_test_split(
        X, y_aspect, y_sentiment, test_size=args.test_size, random_state=args.random_state
    )

    print(f"\nTraining {args.model.upper()} aspect classifier{' (tuned)' if args.tune else ''}...")
    aspect_model = fit_model(args.model, args.tune, X_train, y_aspect_train, args.n_iter, args.random_state)
    report("Aspect", aspect_model, X_test, y_aspect_test, args.plot, f"{args.plot_dir}/confusion_matrix_aspect.png")

    print(f"\nTraining {args.model.upper()} sentiment classifier{' (tuned)' if args.tune else ''}...")
    sentiment_model = fit_model(args.model, args.tune, X_train, y_sentiment_train, args.n_iter, args.random_state)
    report(
        "Sentiment", sentiment_model, X_test, y_sentiment_test, args.plot, f"{args.plot_dir}/confusion_matrix_sentiment.png"
    )

    reviews = collect_reviews(args)
    if not reviews:
        return

    print("\n==== Predictions ====")
    cleaned = build_corpus(reviews)
    X_new = cv.transform(cleaned).toarray()
    aspect_preds = aspect_model.predict(X_new)
    sentiment_preds = sentiment_model.predict(X_new)

    for review, aspect, sentiment in zip(reviews, aspect_preds, sentiment_preds):
        print(f"- \"{review}\"\n    Aspect: {aspect} | Sentiment: {sentiment}")


if __name__ == "__main__":
    main()
