# Automated Climate Fact Verification Project

This project focuses on automated fact verification of climate-related claims using advanced Natural Language Processing (NLP) and Machine Learning (ML) techniques. The primary goal is to evaluate the veracity of claims based on associated evidence using a range of NLP methods, including tokenization, lemmatization, TF-IDF, and deep learning approaches like Sentence Transformers and BERT models.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

The code in this repository is written in Python. You will need Python 3.6 or later to run it. You will also need the following Python libraries:

- pandas
- numpy
- torch
- transformers
- sentence_transformers
- nltk
- Google Colab (for easy setup and GPU support)

You can install these libraries using pip:

```bash
pip install pandas numpy torch transformers sentence_transformers nltk
```

## Model Training
- Utilize Sentence Transformers for sentence embedding, fine-tuning with `facebook/bart-large-mnli`.
- Employ BERT-like models for sequence classification.
- Train models using custom datasets and DataLoader in PyTorch.

## Fact Verification Process
- Retrieve relevant evidence using TF-IDF, BM25, and pretrained models.
- Perform predictions using majority voting based on the evidence.
- Support multiple modes of evidence retrieval (`tfidf`, `bm25`, `bm25+pretrained`).

## Usage
- Prepare the data as per the instructions in the script.
- Train the model using the provided training functions.
- Make predictions by providing claims and evaluating them against the evidence.

## Additional Notes
- Ensure you have sufficient computational resources (preferably Google Colab with GPU support).
- The training process can be computationally intensive and may take significant time.
- The model's performance is highly dependent on the quality and quantity of the training data.
