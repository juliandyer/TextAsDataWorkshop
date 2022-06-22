'''
Code for a Text-As-Data workshop for economists at Uni. Exeter. Partly based on Text-As-Data PhD course by Elliott Ash.
'''

'''
Pandas and Loading text data in a CSV file:
'''
import pandas as pd
UKHansard_1960_1962 = pd.read_csv('UK_Hansard_1960_1962_sample.csv')
print(UKHansard_1960_1962.head())

'''
Cleaning text:
'''
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
hansard_stopwords = ['hon', 'member']

def clean_text(input_string):
    words = nltk.word_tokenize(input_string)
    alpha_tokens = [word for word in words if word.isalpha()]
    tokens_lower = [token.lower() for token in alpha_tokens]
    tokens_no_stopwords = [token for token in tokens_lower if token not in stop_words and token not in hansard_stopwords]
    clean_sentence = ' '.join(tokens_no_stopwords)
    return clean_sentence

UKHansard_1960_1962['clean_text'] = UKHansard_1960_1962.apply(lambda x: clean_text(x['member_speech']), axis=1)

print(UKHansard_1960_1962['member_speech'][0:5])
print(UKHansard_1960_1962['clean_text'][0:5])

'''
Text as a Bag-Of-Words:
'''
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=10000)
tf = vectorizer.fit_transform(UKHansard_1960_1962['clean_text'])
df_bow = pd.DataFrame(tf.toarray(),columns=vectorizer.get_feature_names_out())

#df_bow['total_tokens'] = df_bow.sum(axis=1)
print(df_bow.head())

'''
Topic Modelling:
'''
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation

tfidf = TfidfTransformer()
tfidf_sparse = tfidf.fit_transform(df_bow)
df_tfidf = pd.DataFrame(tfidf_sparse.toarray(), columns=tfidf.get_feature_names_out())

lda = LatentDirichletAllocation(
    n_components=5,
    max_iter=5,
    learning_method="online",
    learning_offset=50.0,
    random_state=520,
)
lda.fit(tf)

tf_feature_names = vectorizer.get_feature_names_out()

import matplotlib.pyplot as plt
import numpy as np
def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(1, 5, figsize=(10, 5), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        raw_weights = topic[top_features_ind]
        weights = raw_weights/np.mean(raw_weights)

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx}", fontdict={"fontsize": 15})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=10)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=0)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()
plot_top_words(lda, tf_feature_names, 10, "Topics in LDA model")

lsa = lda.fit_transform(tf)
df_lsa = pd.DataFrame(lsa).add_prefix('topic_')

UKHansard_1960_1962_topics = pd.concat([UKHansard_1960_1962, df_lsa], axis=1)
UKHansard_1960_1962_topics.to_csv('UKHansard_1960_1962_topics.csv', index=False)

'''
Sentiment Analysis:
'''
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

def sentiment_score(sentence):
    sentiment_dict = sentiment_analyzer.polarity_scores(sentence)
    compound_sentiment = sentiment_dict['compound']
    return compound_sentiment

UKHansard_1960_1962_topics['sentiment'] = UKHansard_1960_1962_topics.apply(lambda x: sentiment_score(x['member_speech']), axis=1)

y = UKHansard_1960_1962_topics['sentiment']
X = UKHansard_1960_1962_topics[['topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4']]

import statsmodels.api as sm
X2 = sm.add_constant(X)
model = sm.OLS(y, X2).fit()
print(model.summary())

'''
Word Vector Embedding
'''
import gensim
model = gensim.models.Word2Vec(UKHansard_1960_1962['clean_text'].str.split(), sorted_vocab=1, min_count=2, workers=1,
                                   vector_size=300, seed=5430)
model.wv.most_similar(positive='minister')
model.wv.most_similar(positive='tanganyika')

model1 = gensim.models.Word2Vec(UKHansard_1960_1962[0:2500]['clean_text'].str.split(), sorted_vocab=1, min_count=10, workers=1,
                                   vector_size=300, seed=5430)
print(model1.wv.most_similar(positive='government', topn=2))
print(model1.wv.similarity('government', 'business'))

model2 = gensim.models.Word2Vec(UKHansard_1960_1962[2500:5000]['clean_text'].str.split(), sorted_vocab=1, min_count=10, workers=1,
                                   vector_size=300, seed=5430)
print(model2.wv.most_similar(positive='government', topn=2))
print(model2.wv.similarity('government', 'business'))

'''
Semantic Similarity & Semantic Arithmetic
'''
import gensim.downloader as api
pretrained_model = api.load("glove-wiki-gigaword-50")

print(pretrained_model.most_similar(positive=['king'], topn=1))
print(pretrained_model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1))

print(pretrained_model.most_similar(['disease','epidemic'], topn=5))

'''
Other tools:
'''
nltk.download('maxent_ne_chunker')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')

ne_tree = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize('London is the capital of the United Kingdom. Boris Johnson is the prime minister (for now).')))
print(ne_tree)