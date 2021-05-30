from newspaper import fulltext
import requests
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter

def get_word_count(text):
    tokens = regexp_tokenize(text, r'\w+')
    return len(tokens)

def get_full_text_from_url(url):
    response = requests.get(url)
    html = response.text
    return fulltext(html)

def clean_data(text):
    tokens = regexp_tokenize(text, r'\w+')
    lemmatizer = WordNetLemmatizer()
    tokens_no_stop = [lemmatizer.lemmatize(t.lower(), pos='a') for t in tokens if t not in stopwords.words('english')]
    clean_text = " ".join(tokens_no_stop)
    return clean_text

def main():
    url = input("Enter URL: ")
    text = get_full_text_from_url(url)
    tokens = clean_data(text)
    counter = Counter(tokens)
    print(counter.most_common(10))

if __name__=="__main__":
    main()