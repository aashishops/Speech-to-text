import nltk
import streamlit as st
import os
from PIL import Image
import imageio


# nltk.download('wordnet')
# nltk.download('stopwords')

from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import defaultdict

def get_filenames_in_folder(folder_path):
    filenames = []
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            filenames.append(filename)
    return filenames

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha() and token.lower() not in stop_words]
    return filtered_tokens

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



def resize_gif(path, save_as=None, resize_to=(200,200)):
    frames = []
    with imageio.get_reader(path) as reader:
        for frame in reader:
            frames.append(imageio.imresize(frame, resize_to))
    if not save_as:
        save_as = path
    imageio.mimsave(save_as, frames)

folder_path = r'C:\Users\dwija\OneDrive - SRM Institute of Science & Technology\gif'

filenames = get_filenames_in_folder(folder_path)

words = []
for filename in filenames:  
    word = filename[:-4]
    words.append(word)
    
input_text = st.text_input('Enter a sentence:')

if input_text:
    gif_paths = []
    words = input_text.split()
    for word in words:
        similar_words = find_similar_words(word, words)
        if similar_words:
            input_caption = ' '.join(similar_words)
            gif_path = os.path.join(folder_path, f'{similar_words[0]}.gif')
            resized_gif_path = os.path.join(folder_path, f'resized_{similar_words[0]}.gif')
            resize_gif(gif_path, resized_gif_path)
            gif_paths.append(resized_gif_path)
        else :
            continue
    images = [Image.open(gif_path) for gif_path in gif_paths]
    
    images[0].save('merged.gif', save_all=True, append_images=images[1:], loop=0)
    
    st.image('merged.gif', caption=input_caption, use_column_width=True)

    