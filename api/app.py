import os
import io
import nltk
import speech_recognition as sr
from nltk.sentiment import SentimentIntensityAnalyzer
from flask import Flask, render_template, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Initialize NLTK and SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Function to convert audio to text and get sentiment score
def analyze_sentiment(audio_file):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_file) as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        sentiment_scores = sia.polarity_scores(text)
        return text, sentiment_scores
    except sr.UnknownValueError:
        return None, None

# Define route for the homepage
@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment_score = None
    transcribed_text = None
    sentiment = None

    if request.method == 'POST':
        # Check if an audio file was submitted
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'})

        audio_file = request.files['audio']
        
        # Check if the file has a valid extension
        if audio_file and (audio_file.filename.endswith('.mp3') or audio_file.filename.endswith('.wav')):
            # Save the uploaded file temporarily
            temp_audio_file = 'temp_audio' + os.path.splitext(audio_file.filename)[1]
            audio_file.save(temp_audio_file)

            # Analyze sentiment and transcribe text
            transcribed_text, sentiment_scores = analyze_sentiment(temp_audio_file)

            if sentiment_scores:
                sentiment_score = sentiment_scores['compound']
                if sentiment_score >= 0.05:
                    sentiment = 'positive'
                elif sentiment_score <= -0.05:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'

            # Delete the temporary audio file
            os.remove(temp_audio_file)

    return render_template('index.html', sentiment_score=sentiment, transcribed_text=transcribed_text)

if __name__ == '__main__':
    app.run(debug=True)
