import nltk
# nltk.download('punkt')
# nltk.download('corpora/wordnet')
# from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

def polarity_1(text):
    score = SentimentIntensityAnalyzer().polarity_scores(text)
    compound_score = score['compound']
    if compound_score >= 0.05:
        sentiment = 'positive'
    elif compound_score <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    return sentiment