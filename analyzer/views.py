from django.shortcuts import render
from django.http import HttpResponse
from .ml_helpers import *
import json
import re

# Create your views here.
def index_view(request):
    try:
        url = request.GET.get('url')
        text = get_full_text_from_url(url)
        lang = detect_language(text)
        clean_text = clean_data(text, lang[0])
        word_count = get_word_count(text)
        language = f"{lang[0]} ({lang[1]})"
        keywords = get_keywords(text, lang[1])

    except:
        try:
            text = request.GET.get('text')
            lang = detect_language(text)
            clean_text = clean_data(text, lang[0])
            word_count = get_word_count(text)
            language = f"{lang[0]} ({lang[1]})"
            keywords = get_keywords(text, lang[1])
        except:
            text = ""
            clean_text = ""
            word_count = 0
            language = ""
            keywords = []
    
    context = {
        "text": text,
        "clean_text": clean_text,
        "word_count": word_count,
        "language": language,
        "keywords": keywords
    }
    return render(request, 'analyzer/index.html', context)

def analysis_result_view(request):
    pass
    # return render(request, 'analyzer/analysis_result.html', context)