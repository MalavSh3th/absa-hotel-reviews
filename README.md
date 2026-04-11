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

> **Note:** The dataset file is not included in this repo. Add your hotel reviews CSV to the working directory and update the data loading cell accordingly.

## Author

Malav Sheth — [LinkedIn](https://linkedin.com/in/malavshethdp)
