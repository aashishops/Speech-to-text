import streamlit as st
import speech_recognition as sr
import os
from PIL import Image
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

# Initialize NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

# Function to recognize speech
def recognize_speech(timeout=10):
    r = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        st.write("Listening...")
        audio = r.listen(source, timeout=timeout)
        st.write("Processing...")

    try:
        text = r.recognize_google(audio)
        return text
    except sr.WaitTimeoutError:
        return "Speech recognition timed out"
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "Could not request results"

# Function to preprocess text
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha() and token.lower() not in stop_words]
    return filtered_tokens

# Function to calculate semantic similarity
def calculate_semantic_similarity(word1, word2, similarity_cache):
    if (word1, word2) in similarity_cache:
        return similarity_cache[(word1, word2)]

    synsets1 = wordnet.synsets(word1)
    synsets2 = wordnet.synsets(word2)
    max_similarity = 0
    for synset1 in synsets1:
        for synset2 in synsets2:
            similarity = synset1.wup_similarity(synset2)
            if similarity is not None and similarity > max_similarity:
                max_similarity = similarity

    similarity_cache[(word1, word2)] = max_similarity
    return max_similarity

# Function to find similar words
def find_similar_words(input_word, available_words):
    similar_words = defaultdict(float)
    similarity_cache = {}

    preprocessed_words = preprocess_text(input_word)
    if not preprocessed_words:
        return []  # Return an empty list if preprocessing results in an empty list

    input_word_preprocessed = preprocessed_words[0]  # Preprocess the input word

    for word in available_words:
        if word == input_word_preprocessed:
            similar_words[word] = 1.0

    # Limiting the number of synonyms generated per word
    synonyms = set()
    for synset in wordnet.synsets(input_word_preprocessed):
        for lemma in synset.lemmas():
            if len(synonyms) > 10:  # Limiting to 10 synonyms per word
                break
            synonyms.add(lemma.name())

    for synonym in synonyms:
        for word in available_words:
            similarity_score = calculate_semantic_similarity(synonym, word, similarity_cache)
            if similarity_score > 0.6:
                similar_words[word] = max(similar_words[word], similarity_score)

    sorted_similar_words = sorted(similar_words, key=similar_words.get, reverse=True)[:5]

    return sorted_similar_words

# Function to get filenames in the folder
def get_filenames_in_folder(folder_path):
    filenames = []
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            filenames.append(filename)
    return filenames

# Streamlit app
st.title("Voice Recognition and Synonym Finder App")

# Voice Recognition
if st.button("Start Recording"):
    st.write("Recording started.")
    text = recognize_speech()
    st.write("Spoken words: ", text)

    # Synonym Finder
    folder_path = r'C:\Users\dwija\OneDrive - SRM Institute of Science & Technology\gif\resized_gifs'  # Change this to your GIF folder path
    filenames = get_filenames_in_folder(folder_path)

    input_text = text

    if input_text:
        gif_paths = []
        words = input_text.split()
        for word in words:
            similar_words = find_similar_words(word, words)
            if similar_words:
                input_caption = ' '.join(similar_words)
                gif_paths.append(os.path.join(folder_path, f'{similar_words[0]}.gif'))
            else:
                continue
        images = [Image.open(gif_path) for gif_path in gif_paths]

        images[0].save('merged.gif', save_all=True, append_images=images[1:], loop=0)

        st.image('merged.gif', caption=input_caption, use_column_width=True)
