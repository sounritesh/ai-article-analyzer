from newspaper import fulltext
import requests
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from yake import KeywordExtractor
import pickle

def detect_language(text):
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
    tokens = regexp_tokenize(text, r'\w+')
    return len(tokens)

def get_full_text_from_url(url):
    response = requests.get(url)
    html = response.text
    return fulltext(html)

def clean_data(text, lang):
    tokens = regexp_tokenize(text, r'\w+')
    lemmatizer = WordNetLemmatizer()
    tokens_no_stop = [lemmatizer.lemmatize(t.lower(), pos='a') for t in tokens if t not in stopwords.words(lang)]
    clean_text = " ".join(tokens_no_stop)
    return clean_text

def get_keywords(text, lang):
    extractor = KeywordExtractor(lan=lang, n=3, top=10)
    keywords_raw = extractor.extract_keywords(text)
    keywords = [k[0] for k in keywords_raw]
    return keywords

def main():
    pass
    
if __name__=="__main__":
    main()