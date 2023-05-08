import streamlit as st
import openai
import io
import tempfile
from google.cloud import texttospeech
from google.oauth2 import service_account
from audio_recorder_streamlit import audio_recorder


# Set your OpenAI API key
openai.api_key = st.secrets["open_api_key"]

# Create a credentials object using the service account info from the secrets
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"],
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

# Initialize the Text-to-Speech client with the credentials object
client = texttospeech.TextToSpeechClient(credentials=credentials)

def synthesize_speech(text, language_code='en-US', voice_name='en-US-Wavenet-A'):
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=input_text,
        voice=voice,
        audio_config=audio_config
    )

    # Save the synthesized speech to a file
    with open("output.mp3", "wb") as out:
        out.write(response.audio_content)



st.title("Speech to Text, Translation, and Text-to-Speech with OpenAI and Google Text-to-Speech")
st.subheader("Upload an audio file or record audio in the browser")

language_options = {
    "Swedish": ("swedish", "sv-SE"),
    "Arabic": ("arabic", "ar-XA"),
    "Chinese": ("chinese", "zh-CN"),
    "Czech": ("czech", "cs-CZ"),
    "Danish": ("danish", "da-DK"),
    "Dutch": ("dutch", "nl-NL"),
    "English": ("english", "en-US"),
    "Finnish": ("finnish", "fi-FI"),
    "French": ("french", "fr-FR"),
    "German": ("german", "de-DE"),
    "Greek": ("greek", "el-GR"),
    "Hebrew": ("hebrew", "he-IL"),
    "Hindi": ("hindi", "hi-IN"),
    "Hungarian": ("hungarian", "hu-HU"),
    "Indonesian": ("indonesian", "id-ID"),
    "Italian": ("italian", "it-IT"),
    "Japanese": ("japanese", "ja-JP"),
    "Korean": ("korean", "ko-KR"),
    "Kurdish": ("kurdish", "ku-TR"),
    "Norwegian": ("norwegian", "no-NO"),
    "Polish": ("polish", "pl-PL"),
    "Portuguese": ("portuguese", "pt-PT"),
    "Romanian": ("romanian", "ro-RO"),
    "Russian": ("russian", "ru-RU"),
    "Spanish": ("spanish", "es-ES"),
    "Swedish": ("swedish", "sv-SE"),
    "Thai": ("thai", "th-TH"),
    "Turkish": ("turkish", "tr-TR"),
    "Ukrainian": ("ukrainian", "uk-UA"),
    "Vietnamese": ("vietnamese", "vi-VN"),
}

selected_language = st.selectbox("Select the target language for translation", options=list(language_options.keys()))
selected_language_code = language_options[selected_language][1]
voice_name = f"{selected_language_code}-Standard-A"


uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"])

audio_bytes = audio_recorder()
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

def transcribe_audio(audio_data):
    if isinstance(audio_data, bytes):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_data)
            f.seek(0)
            transcript = openai.Audio.transcribe("whisper-1", f)
    else:
        transcript = openai.Audio.transcribe("whisper-1", audio_data)

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

if uploaded_file is not None or audio_bytes is not None:
    # Use uploaded_file or audio_bytes as input, depending on which is available
    audio_input = uploaded_file if uploaded_file is not None else audio_bytes
    st.write("Transcribing audio...")
    text = transcribe_audio(audio_input)
    st.write("Transcription:")
    st.text(text)
    st.write("Translating...")
    translated_text = translate_text(text, selected_language)
    st.write("Translation:")
    st.text(translated_text)    
    st.write("Synthesizing speech")
    synthesize_speech(translated_text, selected_language_code, voice_name)
    st.write(f"Speech synthesized in {selected_language} and saved as output.mp3")
    st.audio("output.mp3", format="audio/mp3")


