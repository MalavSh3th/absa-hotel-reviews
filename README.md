# Aspect-Based Sentiment Analysis (ABSA) — Hotel Reviews

A machine learning project that performs **Aspect-Based Sentiment Analysis** on hotel reviews, classifying sentiment across three distinct aspects: **Food**, **Ambience**, and **Service**.

## Overview

Traditional sentiment analysis assigns a single polarity score to an entire review. ABSA goes deeper — it extracts specific aspects from text and determines the sentiment expressed toward each one independently.

**Example:**  
> *"The food was amazing but the service was really slow."*  
> → Food: Positive | Service: Negative

## Dataset

Hotel reviews annotated with aspect labels (`Food`, `Ambience`, `Service`) and sentiment polarity.

## Approach

### Text Preprocessing
- Lowercasing and punctuation removal
- Stopword removal (NLTK)
- Stemming (Porter Stemmer) and Lemmatization (WordNet)

### Feature Engineering
- Bag of Words model using `CountVectorizer` (top 151 features)
- Aspect-based grouping: reviews split into per-aspect subsets

### Models
| Model | Notes |
|---|---|
| Random Forest Classifier | Ensemble, handles non-linear boundaries |
| Support Vector Classifier (SVC) | Tuned via Randomized Search CV |

### Hyperparameter Tuning
`RandomizedSearchCV` used to optimize:
- `max_depth`, `min_samples_leaf`, `min_samples_split` (RF)
- SVC kernel and regularization parameters

## Results

Final model: **Tuned SVC**  
Metrics reported: Accuracy, Classification Report, Confusion Matrix

## Tech Stack

- Python 3.8
- pandas, NumPy
- scikit-learn
- NLTK, spaCy
- Jupyter Notebook (Google Colab)

## Setup

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet averaged_perceptron_tagger
```

Then open `hotel_review_absa.ipynb` in Jupyter or Google Colab.

> **Note:** A sample dataset is included at `data/output.csv`. The notebook loads it automatically from that path.

> **Try it on restaurant reviews:** A second sample dataset with the same schema is included at `data/restaurant_reviews.csv`. To use it, change `DATA_PATH` near the top of the notebook to `"data/restaurant_reviews.csv"`.

## CLI Tool

`absa_cli.py` trains aspect and sentiment classifiers from the command line (retrained fresh on every run, no saved model) and can classify new review text immediately:

```bash
# Train and report metrics only
python absa_cli.py --data data/output.csv

# Classify one or more reviews
python absa_cli.py --data data/restaurant_reviews.csv \
    --review "The food was great but the service was really slow"

# Classify reviews from a file (one per line)
python absa_cli.py --data data/output.csv --reviews-file my_reviews.txt

# Use Random Forest instead of SVC, and run hyperparameter tuning
python absa_cli.py --data data/output.csv --model rf --tune

# Save confusion matrix plots
python absa_cli.py --data data/output.csv --plot --plot-dir ./plots
```

Run `python absa_cli.py --help` for all options.

## Author

Malav Sheth — [LinkedIn](https://linkedin.com/in/malavshethdp)
