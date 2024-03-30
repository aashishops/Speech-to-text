import streamlit as st
import speech_recognition as sr

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

st.title("Voice Recognition App")

if st.button("Start Recording"):
    st.write("Recording started.")
    text = recognize_speech()
    st.write("Spoken words: ", text)


