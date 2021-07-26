from newspaper import fulltext
import requests
from nltk.tokenize import regexp_tokenize, sent_tokenize
from nltk.cluster.util import cosine_distance
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import re
import networkx as nx
from yake import KeywordExtractor
import pickle

def detect_language(text):
    '''
    This is a function to run the trained language detection model to indentify the language of the text

    Parameters:
        text: This is the input text of the article

    Returns:
        A list of ISO name and ISO 639-1 code of the identified language
        [ISO-name, ISO-code] 
    '''

    model = pickle.load(open('ml-models/language-detection/language_detection_model.sav', 'rb'))
    vectorizer = pickle.load(open('ml-models/language-detection/language_detection_dict.sav', 'rb'))
    encoder = pickle.load(open('ml-models/language-detection/language_detection_classes.sav', 'rb'))

    language_dict = {
        'English': ['english', 'en'],
        'Arabic': ['arabic', 'ar'],
        'German': ['german', 'de'],
        'French': ['french', 'fr'],
        'Portugeese': ['portuguese', 'pt'],
        'Italian': ['italian', 'it'],
        'Spanish': ['spanish', 'es'],
        'Dutch': ['dutch', 'nl'],
        'Turkish': ['turkish', 'tr']
    }

    X = vectorizer.transform([text]).toarray()
    y = model.predict(X)
    result = encoder.inverse_transform(y)
    return language_dict[result[0]]

def get_word_count(text):
    '''
    This function is used to count the number of words in an article

    Parameters:
        text: This is the input text of the article

    Returns:
        Number of words in the article (Int)
    '''
    tokens = regexp_tokenize(text, r'\w+')
    return len(tokens)

def get_full_text_from_url(url):
    '''
    This function is used to fetch article text from a url

    Parameters:
        url: This is the url of the article

    Returns:
        It return the full text of the article found at url (Str)
    '''
    response = requests.get(url)
    html = response.text
    return fulltext(html)

def clean_data(text, lang='english'):
    '''
    This function is used to clean input text

    Parameters:
        text: This is the input text to be cleaned
        lang: This is the ISO language name of the text language; default value is 'english'

    Returns:
        It returns clean text which is free of stopwords and is lemmatized (Str)
    '''
    tokens = regexp_tokenize(text, r'\w+')
    lemmatizer = WordNetLemmatizer()
    tokens_no_stop = [lemmatizer.lemmatize(t.lower(), pos='a') for t in tokens if t not in stopwords.words(lang)]
    clean_text = " ".join(tokens_no_stop)
    return clean_text

def get_keywords(text, lang='en'):
    '''
    This function is used to extract keywords from text using YAKE

    Parameters:
        text: This is the input text to be cleaned
        lang: This is the ISO 639-1 code of the text language; default value is 'en'

    Returns:
        It returns a list of keywords extracted from the text
    '''
    extractor = KeywordExtractor(lan=lang, n=3, top=10)
    keywords_raw = extractor.extract_keywords(text)
    keywords = [k[0] for k in keywords_raw]
    return keywords

# Functions for generating the summary
def generate_sentences(text, lang='english'):
    '''
    This function is used to generate sentences broken into words from original article text

    Parameters:
        text: The input text of the article
        lang: The ISO language name of the input text language; default value is 'english'

    Returns:
        It returns a list of sentences where each sentence is represented as a list of words
    '''
    article = sent_tokenize(text, language=lang)
    sentences = []

    for sent in article:
        sentences.append(re.sub("[^\w\"\'\,]", " ", sent).split())

    return sentences

def get_sentence_similarity(sent1, sent2, lang='english'):
    '''
    This function is used to calculate the cosine distance(similarity) between two sentences

    Parameters:
        sent1: This is sentence 1
        sent2: This is sentence 2
        lang: The ISO language name of the input text language; default value is 'english'

    Returns:
        Cosine distance between the two sentences (Float)
    '''
    stop_words = stopwords.words(lang)

    punc_list = [',', '.', '-', '=', '/', ']', '[', '@', '_', '(', ')', '{', '}']

    sent1 = [w.lower() for w in sent1 if w not in punc_list]
    sent2 = [w.lower() for w in sent2 if w not in punc_list]

    word_list = list(set(sent1 + sent2))

    v1 = [0] * len(word_list)
    v2 = [0] * len(word_list)

    for w in sent1:
        if w not in stop_words:
            v1[word_list.index(w)] += 1

    for w in sent2:
        if w not in stop_words:
            v2[word_list.index(w)] += 1

    return 1 - cosine_distance(v1, v2)

def generate_similarity_matrix(sentences, lang='english'):
    '''
    This function is used to generate a similarity matrix for all sentences in a sequence/article

    Parameters:
        sentences: A list of sentences where each sentence is a list of words
        lang: The ISO language name of the input text language; default value is 'english'

    Returns:
        It return a similarity matrix for all sentences in an article (2D numpy array)
    '''
    # creating empty similarity matrix
    similarity_mat = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 != idx2:
                similarity_mat[idx1][idx2] = get_sentence_similarity(
                    sentences[idx1], sentences[idx2], lang)

    return similarity_mat

def generate_summary(text, n=5, lang="english"):
    '''
    This function is used to generate the summary of input text

    Parameters:
        text: This is the input text of the article
        n: Number of sentences in the summary
        lang: ISO language name of the input text language; default value is 'english'

    Returns:
        A summary of the input text (Str)
    '''
    summary_text = []

    sentences = generate_sentences(text, lang)

    similarity_mat = generate_similarity_matrix(sentences, lang)

    # Ranking sentences using the similarity matrix
    similarity_graph = nx.from_numpy_array(similarity_mat)
    scores = nx.pagerank(similarity_graph)

    ranked_sentences = sorted(((scores[i], s, i) for i, s in enumerate(
        sentences)), reverse=True, key=lambda x: x[0])
    print(ranked_sentences)

    shorlisted_sentences = ranked_sentences[0:n]
    shorlisted_sentences = sorted(shorlisted_sentences, key=lambda x: x[2])
    print(shorlisted_sentences)

    for i in range(n):
        summary_text.append(" ".join(shorlisted_sentences[i][1]))

    summary_text = ". ".join(summary_text) + "."
    return summary_text

def main():
    pass
    
if __name__=="__main__":
    main()