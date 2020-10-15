from nltk.stem import WordNetLemmatizer, PorterStemmer
import warnings
import logging
import glob
import string
from string import digits

import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# NLTK Stop words
from nltk.corpus import stopwords

wnl = WordNetLemmatizer()
ps = PorterStemmer()

# Enable logging for gensim - optional
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

warnings.filterwarnings("ignore", category=DeprecationWarning)


def read_file_to_text(filename):

    lines = []
    with open(filename) as file:
        for l in file:
            lines.append(l.strip())
    text = ' '.join(lines)

    return text


def stemming(text):
    words = word_tokenize(text)
    words = [sent for sent in words if not(
        any([ss for ss in sensitive_words if sent.find(ss) >= 0]))]

    words_stemmed = []
    for w in words:
        #print(w, " : ", ps.stem(w))
        #words_stemmed.append(wnl.lemmatize(w) if wnl.lemmatize(w).endswith('e') else ps.stem(w))
        words_stemmed.append(wnl.lemmatize(w))
    return ' '.join(words_stemmed)


def rm_special_chars(data):

    # Remove Emails
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]

    # Remove distracting single/double quotes
    data = [re.sub("\'", "", sent) for sent in data]
    data = [re.sub("\“", "", sent) for sent in data]
    data = [re.sub("\‘", "", sent) for sent in data]

    data = [sent.translate(string.punctuation) for sent in data]
    #remove_digits = str.maketrans('', '', digits)
    #data = [sent.translate(remove_digits) for sent in data]
    # data = [re.sub('[^a-zA-Z#]+', ' ', sent) for sent in data]
    data = [sent.lower() for sent in data]

    sensitive_words = ['vmware', 'vmw']

    data = [stemming(sent) for sent in data]

    return data


def sent_to_words(sentences):
    '''Tokenize words and Clean-up text'''
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


if __name__ == '__main__':

    pth = "/Users/zruxi/Downloads/EDST/euc_ds_pages/"

    # read in all files from path
    list_txt = []
    for filename in glob.glob(pth + "*.txt"):
        text = read_file_to_text(filename)
        list_txt.append(text)

    print('=== There are in total : {} files to process'.format(len(list_txt)))

    # Run in python console
    import nltk
    nltk.download('stopwords')
    # Prep stop words
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

    # Remove special characters
    data = rm_special_chars(list_txt)
    print('=== special characters removed sample text : {}'.format(data[0]))

    # Tokenize words and Clean-up text
    data_words = list(sent_to_words(data))
    print('=== Tokenize words and Clean-up text : {}'.format(data_words[:1]))

    # Creating Bigram and Trigram Models
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # See trigram example
    print(trigram_mod[bigram_mod[data_words[0]]])

    # Remove Stopwords, Make Bigrams and Lemmatize
    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    print('=== Do lemmatization keeping only noun, adj, vb, adv : {}'.format(data_lemmatized[:1]))

    # Create the Dictionary and Corpus needed for Topic Modeling
    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # View
    print('=== Create the Dictionary and Corpus needed for Topic Modeling : {}'.format(corpus[:1]))

    # Building the Topic Model
    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=10,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=20,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
