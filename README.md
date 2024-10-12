# NLP Coursework and Mini-Projects
This repository includes practical work (TPs) and mini-projects related to Natural Language Processing (NLP) as part of the second year of the Master’s in Artificial Intelligence and Data Science (IASD) at the École Nationale Supérieure d'Informatique (ESI), Sidi Bel Abbes.

# Table of Contents
### - Course Material
### - Practical Work (TPs)
### - Mini Projects
### - How to Use
### - Tools & Technologies

## Course Material
This section includes the course notes and summaries for key NLP topics covered during the semester.

## Practical Work (TPs)
### TP-01: Preprocessing in NLP
This TP focuses on essential preprocessing techniques such as:

##### Tokenization
##### Lowercasing
##### Stopword removal
##### Lemmatization and stemming

### TP-02: Feature Extraction and Embeddings
This TP introduces feature extraction techniques and word embeddings like:

##### Bag of Words (BoW)
##### TF-IDF
##### Word2Vec
##### GloVe

### TP-03: Improvement of TP-02
In this TP, the goal is to enhance the feature extraction techniques from TP-02 by improving the model performance and experimenting with advanced embedding techniques.

### TP-04: Sequence Models
This TP delves into sequence models used in NLP, such as:

##### Recurrent Neural Networks (RNNs)
##### Long Short-Term Memory Networks (LSTMs)
##### Gated Recurrent Units (GRUs)

### TP-05: Machine Translation (English to French)
The goal of this TP is to build a basic machine translation model from English to French using sequence models or transformers.

## Mini Projects
### Project 1: Flickr8k Image Captioning using CNNs & LSTMs
This project involves generating image captions using a combination of Convolutional Neural Networks (CNNs) for feature extraction and LSTMs for sequence generation. It is based on the Flickr8k dataset.

### Project 2: Image Captioning with Transformers (Flickr8k)
This project enhances image captioning by using Transformers, a more modern architecture compared to CNN-LSTM models, leveraging the attention mechanism to improve the captioning process on the Flickr8k dataset.
U'll find an attached report of it.

## How to Use
To get started with the code and examples in this repository, follow these steps:

### Clone the repository:

- git clone https://github.com/mohammedhachoud/Natural-Language-Processing
- cd Natural-Language-Processing

### Install the necessary Python packages:

pip install -r requirements.txt

Explore the TPs and mini projects by navigating to the corresponding directories.

## Tools & Technologies

Python: Core language used for the coursework and projects
NLTK: For text preprocessing and tokenization
spaCy: NLP library for advanced text processing
Gensim: For word embeddings like Word2Vec
scikit-learn: Used for feature extraction techniques like TF-IDF
TensorFlow / PyTorch: Deep learning frameworks for sequence models and image captioning
Transformers: Used in Project 2 for image captioning
