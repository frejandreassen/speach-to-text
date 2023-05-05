import streamlit as st
import openai
import io

# Set your OpenAI API key
openai.api_key = st.secrets["open_api_key"]

st.title("Speech to Text with OpenAI Whisper")

uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"])

def transcribe_audio(audio_file):
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript['text']

if uploaded_file is not None:
    st.write("Transcribing audio...")
    text = transcribe_audio(uploaded_file)
    st.write("Transcription:")
    st.write(text)