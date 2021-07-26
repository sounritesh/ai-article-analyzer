from django.shortcuts import render
from .ml_helpers import *

# Create your views here.
def index_view(request):
    try:
        url = request.GET.get('url')
        # n = request.GET.get('n')
        n = 5
        text = get_full_text_from_url(url)
        lang = detect_language(text)
        summary = generate_summary(text, n, lang[0])
        clean_text = clean_data(text, lang[0])
        word_count = get_word_count(text)
        language = f"{lang[0]} ({lang[1]})"
        keywords = get_keywords(text, lang[1])

    except:
        try:
            text = request.GET.get('text')
            # n = request.GET.get('n')
            n = 5
            lang = detect_language(text)
            summary = generate_summary(text, n, lang[0])
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
            summary = ""
    
    context = {
        "text": text,
        "clean_text": clean_text,
        "word_count": word_count,
        "language": language,
        "keywords": keywords,
        "summary": summary
    }
    return render(request, 'analyzer/index.html', context)