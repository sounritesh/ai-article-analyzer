from django.shortcuts import render
from django.http import HttpResponse
from .ml_helpers import *
import json

# Create your views here.
def index_view(request):
    try:
        url = request.GET.get('url')
        text = get_full_text_from_url(url)
        clean_text = clean_data(text)
        word_count = get_word_count(text)
        language = "English (en)"

    except:
        text = ""
        clean_text = ""
        word_count = 0
        language = ""
    
    context = {
        "text": text,
        "clean_text": clean_text,
        "word_count": word_count,
        "language": language
    }
    return render(request, 'analyzer/index.html', context)

def analysis_result_view(request):
    pass

    # return render(request, 'analyzer/analysis_result.html', context)