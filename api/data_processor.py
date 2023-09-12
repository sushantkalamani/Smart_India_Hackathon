import numpy as np
import pandas as pd
import string
import re
import mlproject as model
from os import path
import speech_recognition as sr

AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "data/sad.wav")
r = sr.Recognizer()

def data_preprocessing(text):
    lower_text = text.lower()
    cleaned_text = lower_text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r"https\S+|www\S+https\S+", '',cleaned_text, flags=re.MULTILINE)
    text=text.strip()
    return text

# Load the audio file
with sr.AudioFile(AUDIO_FILE) as source:
    # Adjust for ambient noise, if necessary
    r.adjust_for_ambient_noise(source)
    print("Listening...")
    # Listen to the audio file
    audio = r.listen(source)

# Initialize the recognizer
try:
    text = r.recognize_google(audio)
    print("Transcription: " + text)
    final = data_preprocessing(text)
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand the audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition; {0}".format(e))

result = model.polarity_1(final)
print(f'sentiment is {result}')