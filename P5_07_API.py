import streamlit as st
import pandas as pd
import nltk
from sklearn.neural_network import MLPClassifier
import numpy  as np
import ast
from nltk.corpus import wordnet
from nltk.tokenize import WhitespaceTokenizer
import re
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import string
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
from nltk import pos_tag
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize

st.title('Recommandations de tags Stackoverflow')
st.markdown("Bienvenue dans cette interface. Vous utilisez Stackoverflow? Vous ne savez pas quel est le tag pertinent le plus pertinent à utiliser pour votre question? Nous avons la solution! Suivez les étapes ci-dessous. ")

@st.cache
def loadData():
	df = pd.read_csv("corpusnoverbs.csv")
	return df
df = loadData()
st.header("Nettoyage de votre question:")

replacements = {
    '<': '',
    '><': ' ',
    '>': ''}
lemmatizer = WordNetLemmatizer()
porter = PorterStemmer()
stop = stopwords.words("english")
stop.extend(["'d", "'ll", "'ve", "like","ex","etc","when","whenever","will","would","could","how","why","who","what","good", "big", "fine","'m", "'s","nt","n't","one","two","three","four","five","six","seven","eight","nine","ten", "per", "etc"])
punctuations = string.punctuation
numbers = '0123456789'
tags_to_remove = []


@st.cache
def remove_stopwords(text_to_modify):
    stop_words = set(stop)
    words = word_tokenize(text_to_modify)
    filtered = [w for w in words if not w in stop_words]
    return ' '.join(map(str, filtered))
@st.cache
def remove_punctuations(text_to_modify):
    for x in text_to_modify:
            if x in punctuations:
                text_to_modify = text_to_modify.replace(x, "")
            if x in numbers:
                text_to_modify = text_to_modify.replace(x, "")
    return text_to_modify
@st.cache
def withoutverbs(text):
    tokenizer = WhitespaceTokenizer()
    sent = nltk.pos_tag(tokenizer.tokenize(text))
    return [x for (x,y) in sent if ((y not in ('VBN')) and (y not in ('VBG'))and (y not in ('VBP'))and (y not in ('VB'))and (y not in ('VBD'))and (y not in ('RB')))]

def untokenize(text):
    return " ".join(text)
@st.cache
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
@st.cache
def lemmatize_sentence(sentence):
    nltk_tagged = nltk.pos_tag(sentence)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)
@st.cache
def tokenization(text_to_modify):
    tokenizer = WhitespaceTokenizer()
    words = tokenizer.tokenize(text_to_modify)
    return [w for w in words]
@st.cache
def stemSentence(text_to_modify):
    stem_sentence=[]
    for word in text_to_modify:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append("")
    return "".join(stem_sentence)

@st.cache
def remove_low_freq_noverbs(text_to_modify):
    lists =  df['Body']
    words2 = []
    for wordList in lists:
        words2 += wordList
    fdist = FreqDist(words2)
    fdist_min = sorted(w for w in set(words2) if fdist[w] < 21)
    sw_low_noverbs = set()
    sw_low_noverbs.update(fdist_min)
    stopwords = sw_low_noverbs
    tokenizer = WhitespaceTokenizer()
    words = tokenizer.tokenize(text_to_modify)
    return [word for word in words if word not in stopwords]

def cleaning_yourquestion(text):
    text = str.lower(text)
    text = BeautifulSoup(text).get_text()
    text = text.replace('\n', ' ')
    text = text.replace('  ', ' ')
    text = remove_stopwords(text)
    text = remove_punctuations(text)
    text = withoutverbs(text)
    text = lemmatize_sentence(text)
    text = stemSentence(text)
    text = remove_low_freq_noverbs(text)
    text = untokenize(text)
    return text

text = st.text_input('Entrez votre question à poser à la communauté')
st.write(cleaning_yourquestion(text))

def removestring(text):
    return ast.literal_eval(text)
corpus = df.copy()
corpus[['Body', 'Tags']] = corpus[['Body', 'Tags']].applymap(removestring)

@st.cache
def recommend_tags(text):
    multilabel_binarizer = MultiLabelBinarizer()
    y_tags = multilabel_binarizer.fit_transform(corpus['Tags'])
    X_train = corpus['Body']
    X_train = [" ".join(x) for x in X_train]
    vectorizer = TfidfVectorizer(lowercase=False, max_features=1000)
    X_tfidf_train = vectorizer.fit_transform(X_train)
    X_tfidf_test = vectorizer.transform([text])
    mlp_classifier = MLPClassifier(solver= 'lbfgs', learning_rate= 'adaptive', hidden_layer_sizes=(203,), activation='relu', alpha= 0.696991203689759)
    mlp_classifier.fit(X_tfidf_train, y_tags)
    mlp_predictions = mlp_classifier.predict_proba(X_tfidf_test)
    thresh = 0.35
    pred = (mlp_predictions>thresh).astype(int)
    predictionns = multilabel_binarizer.inverse_transform(pred)
    predictionns = removestring(''.join(map(str, predictionns)))
    return ' '.join(predictionns)

st.header("Vos tag(s) recommandés: ")
st.write(recommend_tags(text))
