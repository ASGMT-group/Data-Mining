import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Sample reviews (replace with your dataset)
reviews = [
    "This product is amazing! I love it.",
    "Terrible experience. I would not recommend it.",
    "Decent product, but it could be better.",
    "The worst service I have ever received.",
]

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())

    # Remove stop words and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

    # Stemming (optional)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    return ' '.join(tokens)

def analyze_sentiment(review):
    sid = SentimentIntensityAnalyzer()
    compound_score = sid.polarity_scores(review)['compound']

    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Preprocess and analyze each review
for review in reviews:
    preprocessed_review = preprocess_text(review)
    sentiment = analyze_sentiment(preprocessed_review)
    print(f"Review: {review}\nSentiment: {sentiment}\n")
