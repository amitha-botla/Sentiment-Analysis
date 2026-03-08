# Sentiment-Analysis
Comparing rule-based (VADER) and transformer-based (FinBERT) approaches
for sentiment classification on financial news headlines.

## Overview
This project evaluates two sentiment analysis models on 4,846 financial
news headlines labeled as positive, negative, or neutral. VADER is a
general-purpose rule-based tool while FinBERT is a BERT model fine-tuned
specifically on financial text. The goal is to understand whether a
domain-specific model meaningfully outperforms a general-purpose one
on financial language, and where each model breaks down.

## Results

| Model   | Accuracy |
|---------|----------|
| VADER   | 54.3%    |
| FinBERT | 88.9%    |

## VADER Analysis

VADER achieves 54.3% accuracy and shows a clear positive bias. The actual
dataset has 1,363 positive headlines but VADER predicted 2,398, a 76%
overestimation. At the same time it predicted only 2,003 neutral headlines
against an actual 2,879, missing nearly a third of neutral cases.

The negative class is where VADER fails most severely. Only 180 out of 604
negative headlines were correctly identified. 239 negative headlines were
misclassified as positive and 185 as neutral. This is a fundamental problem
with using a general-purpose lexicon on financial text. VADER scores words
in isolation, so a headline like "operating profit fell" can still score
positively because "profit" carries positive weight regardless of context.

Confusion matrix:

|                 | Predicted Negative | Predicted Neutral | Predicted Positive |
|-----------------|--------------------|-------------------|--------------------|
| Actual Negative | 180                | 185               | 239                |
| Actual Neutral  | 197                | 1487              | 1195               |
| Actual Positive | 68                 | 331               | 964                |

## FinBERT Analysis

FinBERT achieves 88.9% accuracy, a 35 percentage point improvement. Being
trained on financial text, it understands context rather than surface-level
keywords, which directly addresses VADER's core weakness.

The improvement on the negative class is the most significant finding.
FinBERT correctly identifies 586 out of 604 negative headlines compared to
VADER's 180. Only 7 negative headlines were misclassified as positive.

The neutral class remains the weakest area for FinBERT. 287 neutral
headlines were misclassified as positive and 123 as negative. This is
expected as neutral financial headlines are inherently ambiguous and sit
between clear sentiment categories.

Confusion matrix:

|                 | Predicted Negative | Predicted Neutral | Predicted Positive |
|-----------------|--------------------|-------------------|--------------------|
| Actual Negative | 586                | 11                | 7                  |
| Actual Neutral  | 123                | 2469              | 287                |
| Actual Positive | 22                 | 86                | 1255               |

## Methodology
1. Loaded 4,846 labeled financial headlines
2. Applied VADER scoring using NLTK's SentimentIntensityAnalyzer, converting
compound scores to labels (positive >= 0.05, negative <= -0.05, neutral in between)
3. Loaded FinBERT via HuggingFace Transformers pipeline with batch size 10
and ran inference across all headlines on CPU (no GPU available)
4. Evaluated both models using accuracy score, confusion matrix, and
classification report

## Tools and Libraries
- Python
- NLTK (VADER)
- HuggingFace Transformers (FinBERT)
- scikit-learn
- pandas
- seaborn

## Dataset
Financial news headlines with sentiment labels (positive, negative, neutral).
Source: Kaggle Sentiment Analysis Dataset by Ankur Sinha
