from sklearn.manifold import TSNE
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.models import Phrases
from copy import deepcopy
import plotly.express as px
import numpy as np
import pandas as pd
import json
import os
import logging
import pytextrank
import spacy
import re
import time
import string

# ===== START LOGGER =====
logger = logging.getLogger(__name__)
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
sh.setFormatter(formatter)
root_logger.addHandler(sh)


nlp = spacy.load("en_core_web_sm")

desired_width = 320
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", desired_width)

if "cord-nlp-cytoscape" in os.listdir("."):
    os.chdir("cord-nlp-cytoscape")

#datadir = "/Users/zruxi/Downloads/EDST/euc_ds_pages/"
#data_files = [i for i in os.listdir(datadir) if ".txt" in i]
#
df_input_medium = pd.read_csv('../inputs/medium_texts.csv')
#df_input_confu = pd.read_csv('../inputs/confu_texts.csv')
df_input_medium['src'] = 'medium.com'
#df_input_confu['src'] = 'confluence'
#df_input = pd.concat([df_input_medium, df_input_confu])
df_input = df_input_medium
print(len(df_input))
df_input = df_input[~df_input['text'].isnull()]
print(len(df_input))

# For testing the script with a smaller dataset
# data_files = data_files[:500]

# util functions

'''
def read_file_to_text(filename):

    lines = []
    with open(filename) as file:
        for l in file:
            lines.append(l.strip())
    text = ' '.join(lines)

    return text
'''


def has_digits(string):

    RE_D = re.compile('\d')
    res = RE_D.search(string)
    return res is not None


def remove_special_chars(data):

    # Remove Emails
    data = re.sub('\S*@\S*\s?', '', data)
    # Remove new line characters
    data = re.sub('\s+', ' ', data)
    # Remove distracting single/double quotes
    data = re.sub("\'", "", data)
    data = re.sub("\“", "", data)
    data = re.sub("\‘", "", data)

    return data

class Summarize:

    def __init__(self, text, limit_phrases, limit_sentences):

        self._text = text
        self._limit_phrases = limit_phrases
        self._limit_sentences = limit_sentences

    def getSummary(self):

        # load a spaCy model, depending on language, scale, etc.
        nlp = spacy.load("en_core_web_sm")

        # add PyTextRank into the spaCy pipeline
        tr = pytextrank.TextRank(logger=None)
        nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)
        doc = nlp(self._text)

        list_outcome = []
        for phrase in doc._.textrank.summary(
                limit_phrases=self._limit_phrases,
                limit_sentences=self._limit_sentences):

            list_outcome.append(phrase)
        return list_outcome


chars = re.escape(string.punctuation)
start = time.time()
# ========== CONVERT STRINGS TO BoW ==========
logger.info(f"Tokenising & Lemmatising {len(df_input)} files")
unigram_docs = []  # This will be the list of lemmatised abstracts
unigram_datafiles = []
docs_titles = []
filenames = []
#docs_abstract = []
docs_src = []
counter = 0

for row in df_input.iterrows():
#for datafile in data_files:
    # with open(os.path.join(datadir, datafile), "r") as f:
    #    data_dict = json.load(f)
    #text = read_file_to_text(datadir + datafile)

    # For using either the abstract data of the title data
    # abs_text = " ".join(
    #    [i["text"] for i in data_dict["abstract"]]
    # )  # Abstract could be in multiple sections
    #title_text = data_dict["metadata"]["title"].lower()

    #getSumm = Summarize(text, 15, 5)
    #abs_text = getSumm.getSummary()

    #title_text = re.sub(r'[' + chars + ']', ' ', datafile[:-4])
    title_text = row[1]['title']

    # raw_text = title_text
    raw_text = row[1]['text']
    src = row[1]['src']

    raw_text = remove_special_chars(raw_text)
    # Tokenise & lemmatise data w/ spacy
    doc = nlp(raw_text)
    lemmas = [
        token.lemma_.lower()
        for token in doc
        if (
            not token.is_stop
            and token.pos is not "SYM"
            and token.pos_ is not "PUNCT"
            and len(token) > 1
        )
    ]
    if len(lemmas) > 10:
        unigram_docs.append(lemmas)
        #unigram_datafiles.append(datafile)
        docs_titles.append(title_text)
        #docs_abstract.append(abs_text)
        #filenames.append(datafile)
        docs_src.append(src)
    counter += 1
    if counter % 1000 == 0:
        logger.info(
            f"Tokenised/Lemmatised {counter} of {len(df_input)} files")

print("Tokenising & Lemmatising took" + str(int(time.time() - start)))

# ========== Add n-grams ==========
logger.info(f"Generating n-grams and adding them to the corpus")
docs = deepcopy(unigram_docs)  # Make new instance before manipulating docs

# Build bigrams
bigram = Phrases(docs, min_count=30)
trigram = Phrases(bigram[docs], min_count=15)
for i in range(len(docs)):
    doc = docs[i]
    bigrams_ = [b for b in bigram[doc] if b.count("_") == 1]
    trigrams_ = [t for t in trigram[bigram[doc]] if t.count("_") == 2]
    # print(f'Found bigrams {bigrams_}')
    # print(f'Found trigrams {trigrams_}')
    docs[i] = doc + bigrams_ + trigrams_


# ===== Inspect n-grams =====
def vis_ngrams(docs_in, n_ngrams=20):

    from collections import Counter

    frequencies = Counter([])
    for text in docs_in:
        frequencies += Counter(text)

    unigram_df = pd.DataFrame(
        [{"ngram": k, "count": v}
            for k, v in frequencies.items() if "_" not in k]
    ).sort_values("count", ascending=False)
    bi_gram_df = pd.DataFrame(
        [{"ngram": k, "count": v} for k, v in frequencies.items() if "_" in k]
    ).sort_values("count", ascending=False)
    trigram_df = pd.DataFrame(
        [{"ngram": k, "count": v}
            for k, v in frequencies.items() if k.count("_") == 2]
    ).sort_values("count", ascending=False)
    print(unigram_df[:n_ngrams])
    print(bi_gram_df[:n_ngrams])
    print(trigram_df[:n_ngrams])

    # Visualise counts of top n-grams
    fig = px.bar(
        unigram_df[:n_ngrams],
        x="ngram",
        y="count",
        title="Counts of top unigrams",
        template="plotly_white",
        labels={"ngram": "Unigram", "count": "Count"},
    )
    fig.show()
    fig = px.bar(
        bi_gram_df[:n_ngrams],
        x="ngram",
        y="count",
        title="Counts of top bi-grams",
        template="plotly_white",
        labels={"ngram": "Bigram", "count": "Count"},
    )
    fig.show()
    fig = px.bar(
        trigram_df[:n_ngrams],
        x="ngram",
        y="count",
        title="Counts of top trigrams",
        template="plotly_white",
        labels={"ngram": "Trigram", "count": "Count"},
    )
    fig.show()
    return True


vis_ngrams(docs, n_ngrams=20)  # TODO - THIS IS SLOW AS SHIT

# ========== Further filtering ==========
logger.info(
    f"Further cleaning the text by removing custom stopwords & rare/common words"
)
# Remove custom stopwords
custom_stopwords = [
    "play_important_role",
    "play_critical_role",
    "play_key_role",
    "95_confidence_interval",
    "provide_new_insight",
    "et_al",
    "pubmed_abstract",
    "publisher_text",
    "present_study",
    "results_suggest",
    "result_suggest",
    "95_ci",
    "play_important",
    "study",
    "result",
    "analysis",
    "method",
]
docs = [
    [token for token in doc if token.lower() not in custom_stopwords] for doc in docs
]

# # Visualis n-grams again
# vis_ngrams(docs)
# write out the docs for UI iterations
with open('../outputs/list_input_lda.txt', 'w') as f:
    for doc in docs:
        f.write("%s\n" % doc)

# Create a dictionary representation of the documents.
dictionary = Dictionary(docs)

# Filter out words that occur less than 20 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=20, no_above=0.5)

# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(doc) for doc in docs]

print("Number of unique tokens: %d" % len(dictionary))
print("Number of documents: %d" % len(corpus))

# ========== LDA - train our model with Gensim ==========
logger.info(f"Training LDA model")
# Set training parameters.
num_topics = 12
chunksize = 2000
passes = 20
iterations = 400
eval_every = None  # Don't evaluate model perplexity, takes too much time.

# Make a index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

lda = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha="auto",
    eta="auto",
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every,
)

logger.info(f"Take a look at the results")
top_topics = lda.top_topics(corpus)  # , num_words=20)

# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
pyLDAvis.save_html(vis, '../assets/lda.html')

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print("Average topic coherence: %.4f." % avg_topic_coherence)

# # ========== Visualise the data with pyLDAvis ==========
# from pprint import pprint
# pprint(top_topics)
#
# import pyLDAvis.gensim
# import pyLDAvis
#
# vis = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
# with open('outputs/lda_vis.html', 'w') as f:
#     pyLDAvis.save_html(vis, f)

# ========== T-SNE transform ==========
lda_vals = list()
for d in corpus:
    topics_tup = lda.get_document_topics(
        d
    )  # This should be a N by K matrix where N = corpus size, K = topics
    temp_dict = {i: 0 for i in range(num_topics)}
    for t in topics_tup:
        temp_dict[t[0]] = t[1]
    lda_vals.append(temp_dict)

lda_df = pd.DataFrame(lda_vals)
lda_arr = lda_df.values

lda_topics = {i[0]: i[1].split(" + ") for i in lda.print_topics(-1)}
topics_txt = [lda_topics[i] for i in range(num_topics)]
topics_txt = [[j.split("*")[1].replace('"', "")
               for j in i] for i in topics_txt]
topics_txt = ["; ".join(i) for i in topics_txt]

lda_df = lda_df.assign(topic_id=[str(lda_arr[i].argmax())
                                 for i in range(len(lda_arr))])
lda_df = lda_df.assign(
    topic_txt=[topics_txt[lda_arr[i].argmax()] for i in range(len(lda_arr))]
)
lda_df = lda_df.assign(
    topics=["Topic: " + str(lda_arr[i].argmax()) for i in range(len(lda_arr))]
)
lda_df = lda_df.assign(title=docs_titles)
lda_df = lda_df.assign(filename=docs_titles)
lda_df = lda_df.assign(src=docs_src)

# for tsne_perp in [20, 35, 50, 100, 200]:  # Test out different perplexity values
for tsne_perp in [40]:  # Test out different perplexity values
    tsne_embeds = TSNE(
        n_components=2,
        perplexity=tsne_perp,
        n_iter=350,
        n_iter_without_progress=100,
        learning_rate=500,
        random_state=42,
    ).fit_transform(lda_arr)
    lda_df = pd.concat([lda_df, pd.DataFrame(
        tsne_embeds, columns=["x", "y"])], axis=1)

    # Visualise the t-SNE topics
    topic_ids = "Topic: " + lda_df["topic_id"].astype(str).values
    fig = px.scatter(
        lda_df,
        title="t-SNE test, perplexity: " + str(tsne_perp),
        x="x",
        y="y",
        color=topic_ids,
        color_discrete_sequence=px.colors.qualitative.Light24,
        hover_name="title",
        hover_data=["topic_txt"],
        template="plotly_white",
    )
    fig.show()

lda.save("../outputs/lda_model")
lda_df.to_csv("../outputs/lda_df.csv", index=False)
with open("../outputs/lda_topics.json", "w") as f:
    json.dump(lda_topics, f)
