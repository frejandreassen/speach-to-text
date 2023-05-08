import streamlit as st
import openai
import io

# Set your OpenAI API key
openai.api_key = st.secrets["open_api_key"]

st.title("Speech to Text and Translation with OpenAI Whisper and ChatGPT")

uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"])

def transcribe_audio(audio_file):
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript['text']

def translate_text(prompt, target_language):
    message = {
        'role': 'system',
        'content': f"Translate the following text to {target_language}: {prompt}"
    }

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    st.session_state.messages.append(message)

    completions = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=st.session_state.messages
    )
    return completions.choices[0].message.content.strip()

language_options = {
    "Arabic": "arabic",
    "Chinese": "chinese",
    "Czech": "czech",
    "Danish": "danish",
    "Dutch": "dutch",
    "English": "english",
    "Finnish": "finnish",
    "French": "french",
    "German": "german",
    "Greek": "greek",
    "Hebrew": "hebrew",
    "Hindi": "hindi",
    "Hungarian": "hungarian",
    "Indonesian": "indonesian",
    "Italian": "italian",
    "Japanese": "japanese",
    "Korean": "korean",
    "Kurdish": "kurdish",
    "Norwegian": "norwegian",
    "Polish": "polish",
    "Portuguese": "portuguese",
    "Romanian": "romanian",
    "Russian": "russian",
    "Spanish": "spanish",
    "Swedish": "swedish",
    "Thai": "thai",
    "Turkish": "turkish",
    "Ukrainian": "ukrainian",
    "Vietnamese": "vietnamese",
}

selected_language = st.selectbox("Select the target language for translation", options=list(language_options.keys()))

if uploaded_file is not None:
    st.write("Transcribing audio...")
    text = transcribe_audio(uploaded_file)
    st.write("Transcription:")
    st.write(text)
    st.write("Translating...")
    translated_text = translate_text(text, selected_language)
    st.write("Translation:")
    st.write(translated_text)
