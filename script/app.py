import nltk
import streamlit as st

# Download necessary NLTK data
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

# Import necessary functions
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from difflib import SequenceMatcher
from collections import defaultdict

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha() and token.lower() not in stop_words]
    return filtered_tokens

def calculate_semantic_similarity(word1, word2):
    synsets1 = wordnet.synsets(word1)
    synsets2 = wordnet.synsets(word2)
    max_similarity = 0
    for synset1 in synsets1:
        for synset2 in synsets2:
            similarity = synset1.wup_similarity(synset2)
            if similarity is not None and similarity > max_similarity:
                max_similarity = similarity
    return max_similarity

def find_similar_words(input_text, available_words, max_similar_words=5):
    similar_words = defaultdict(float)
    
    
    input_tokens = preprocess_text(input_text)
    
  
    for word in available_words:
        if word in input_tokens:
            similar_words[word] = 1.0
    
    
    synonyms = set()
    for token in input_tokens:
        for synset in wordnet.synsets(token):
            for lemma in synset.lemmas():
                synonyms.add(lemma.name())
    
   
    for synonym in synonyms:
        for word in available_words:
            similarity_score = calculate_semantic_similarity(synonym, word)
            if similarity_score > 0.6:  
                similar_words[word] = max(similar_words[word], similarity_score)
    
    
    sorted_similar_words = sorted(similar_words, key=similar_words.get, reverse=True)[:max_similar_words]
    
    return sorted_similar_words

# Define available words
available_words = ['apple', 'banana', 'cherry', 'date', 'elderberry', 'fig', 'grape', 'honeydew', 'kiwi', 'lemon']

# Define input text
input_text = st.text_input('Enter a sentence:')

# Display similar words
if input_text:
    st.write('Similar words:', find_similar_words(input_text, available_words))