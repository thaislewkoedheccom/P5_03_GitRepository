import pandas as pd
import string
import numpy as np
import nltk
import cufflinks as cf
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from gensim import corpora
from nltk.probability import FreqDist
import seaborn as sns
import ast
from tqdm import tqdm
from gensim import models
from sklearn.preprocessing import MultiLabelBinarizer
from bs4 import BeautifulSoup
from sklearn.neural_network import MLPClassifier
import gensim
from nltk.corpus import wordnet
from nltk import pos_tag
from sklearn.metrics import jaccard_score
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import scattertext as st

########################
## CLEANING
#######################

data = pd.read_csv('P5_0.5_data_query.csv')


#########################
# fonctions de nettoyage
#########################


_replacements = {
    '<': '',
    '><': ' ',
    '>': ''
}
def _do_replace(match):
    return _replacements.get(match.group(0))

def replace_tags(text, _re=re.compile('|'.join(re.escape(r) for r in _replacements))):
    return _re.sub(_do_replace, text)

def remove_tags_lowfreq(text_to_modify):
    stop_words = set(tags_to_remove)
    words = tokenizer.tokenize(text_to_modify)
    filtered = [w for w in words if not w in stop_words]
    return ' '.join(map(str, filtered))

def remove_stopwords(text_to_modify):
    stop = stopwords.words("english")
    stop.extend(["'d", "'ll", "'ve", "like","ex","etc","when","whenever","will","would","could","how","why","who","what","good", "big", "fine","'m", "'s","nt","n't","one","two","three","four","five","six","seven","eight","nine","ten", "per", "etc"])
    stop_words = set(stop)
    words = word_tokenize(text_to_modify)
    filtered = [w for w in words if not w in stop_words]
    return ' '.join(map(str, filtered))

def remove_punctuations(text_to_modify):
    for x in text_to_modify:
            if x in punctuations:
                text_to_modify = text_to_modify.replace(x, "")
            if x in numbers:
                text_to_modify = text_to_modify.replace(x, "")
    return text_to_modify

def withoutverbs(text):
    tokenizer = WhitespaceTokenizer()
    sent = nltk.pos_tag(tokenizer.tokenize(text))
    return [x for (x,y) in sent if ((y not in ('VBN')) and (y not in ('VBG'))and (y not in ('VBP'))and (y not in ('VB'))and (y not in ('VBD'))and (y not in ('RB')))]

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

def lemmatize_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    nltk_tagged = nltk.pos_tag(sentence)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)

def tokenization(text_to_modify):
    words = tokenizer.tokenize(text_to_modify)
    return [w for w in words]

def stemSentence(text_to_modify):
    porter = PorterStemmer()
    stem_sentence=[]
    for word in text_to_modify:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append("")
    return "".join(stem_sentence)

def remove_low_freq_noverbs(text_to_modify):
    stopwords = sw_low_noverbs
    tokenizer = WhitespaceTokenizer()
    words = tokenizer.tokenize(text_to_modify)
    return [word for word in words if word not in stopwords]

#########################
# nettoyage  tags
#########################

data[['Tags']] = data[['Tags']].applymap(replace_tags)
toptags = pd.Series(np.concatenate([x.split() for x in data.Tags])).value_counts()
tags_to_remove = toptags[101:len(toptags)+1].index.tolist()
tokenizer = WhitespaceTokenizer()
data[['Tags']] = data[['Tags']].applymap(remove_tags_lowfreq)
data.replace(r'^\s*$', np.nan, regex=True, inplace=True)
data = data[data['Tags'].notna()]

#########################
# nettoyage  texte
#########################
data[['Body','Tags']] = data[['Body','Tags']].applymap(str.lower)
data['Body'] = data['Body'].apply(lambda x: BeautifulSoup(x).get_text())
data['Body'] = data['Body'].apply(lambda x: x.replace('\n', ' '))
data['Body'] = data['Body'].apply(lambda x: x.replace('  ', ' '))
data[['Body']] = data[['Body']].applymap(remove_stopwords)
data[['Body']] = data[['Body']].applymap(remove_punctuations)
data[['Body']]= data[['Body']].applymap(withoutverbs)
data[['Body']]= data[['Body']].applymap(lemmatize_sentence)
data[['Body', 'Tags']]= data[['Body', 'Tags']].applymap(tokenization)
data[['Body']]= data[['Body']].applymap(stemSentence)


lists =  data['Body']
words2 = []
for wordList in lists:
    words2 += wordList
fdist = FreqDist(words2)
fdist_min = sorted(w for w in set(words2) if fdist[w] < 21)
sw_low_noverbs = set()
sw_low_noverbs.update(fdist_min)
data[['Body']] = data[['Body']].applymap(remove_low_freq_noverbs)


#########################
# modélisation
#########################
