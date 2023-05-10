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
        'content': f"Translate the following text to {target_language}. Only translation, no excuses. Text: {prompt}."
    }

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    st.session_state.messages.append(message)

    completions = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=st.session_state.messages
    )
    return completions.choices[0].message.content.strip()

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

    return response

language_options = {
    "Swedish": ("swedish", "sv-SE"),
    "Afrikaans": ("afrikaans", "af-ZA"),
    "Arabic": ("arabic", "ar-XA"),
    "Basque": ("basque", "eu-ES"),
    "Bengali": ("bengali", "bn-IN"),
    "Bulgarian": ("bulgarian", "bg-BG"),
    "Catalan": ("catalan", "ca-ES"),
    "Chinese (Hong Kong)": ("chinese", "yue-HK"),
    "Chinese": ("chinese", "cmn-CN"),
    "Czech": ("czech", "cs-CZ"),
    "Danish": ("danish", "da-DK"),
    "Dutch (Belgium)": ("dutch", "nl-BE"),
    "Dutch": ("dutch", "nl-NL"),
    "English (Australia)": ("english", "en-AU"),
    "English (India)": ("english", "en-IN"),
    "English (UK)": ("english", "en-GB"),
    "English (US)": ("english", "en-US"),
    "Filipino": ("filipino", "fil-PH"),
    "Finnish": ("finnish", "fi-FI"),
    "French (Canada)": ("french", "fr-CA"),
    "French": ("french", "fr-FR"),
    "Galician": ("galician", "gl-ES"),
    "German": ("german", "de-DE"),
    "Greek": ("greek", "el-GR"),
    "Gujarati": ("gujarati", "gu-IN"),
    "Hebrew": ("hebrew", "he-IL"),
    "Hindi": ("hindi", "hi-IN"),
    "Hungarian": ("hungarian", "hu-HU"),
    "Icelandic": ("icelandic", "is-IS"),
    "Indonesian": ("indonesian", "id-ID"),
    "Italian": ("italian", "it-IT"),
    "Japanese": ("japanese", "ja-JP"),
    "Kannada": ("kannada", "kn-IN"),
    "Korean": ("korean", "ko-KR"),
    "Latvian": ("latvian", "lv-LV"),
    "Lithuanian": ("lithuanian", "lt-LT"),
    "Malay": ("malay", "ms-MY"),
    "Malayalam": ("malayalam", "ml-IN"),
    "Mandarin Chinese": ("chinese", "cmn-CN"),
    "Mandarin Chinese (Taiwan)": ("chinese", "cmn-TW"),
    "Marathi": ("marathi", "mr-IN"),
    "Norwegian": ("norwegian", "nb-NO"),
    "Polish": ("polish", "pl-PL"),
    "Portuguese (Brazil)": ("portuguese", "pt-BR"),
    "Portuguese": ("portuguese", "pt-PT"),
    "Punjabi": ("punjabi", "pa-IN"),
    "Romanian": ("romanian", "ro-RO"),
    "Russian": ("russian", "ru-RU"),
    "Serbian": ("serbian", "sr-RS"),
    "Slovak": ("slovak", "sk-SK"),
    "Spanish (Spain)": ("spanish", "es-ES"),
    "Spanish (US)": ("spanish", "es-US"),
    "Swedish": ("swedish", "sv-SE"),
    "Tamil": ("tamil", "ta-IN"),
    "Telugu": ("telugu", "te-IN"),
    "Thai": ("thai", "th-TH"),
    "Turkish": ("turkish", "tr-TR"),
    "Ukrainian": ("ukrainian", "uk-UA"),
    "Vietnamese": ("vietnamese", "vi-VN")
}


## Streamlit app
st.title("Speech to Text, Translation, and Text-to-Speech with OpenAI and Google Text-to-Speech")
st.subheader("Upload an audio file or record audio in the browser")

selected_language = st.selectbox("Select the target language for translation", options=list(language_options.keys()))
selected_language_code = language_options[selected_language][1]
voice_name = f"{selected_language_code}-Standard-A"


uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"])

audio_bytes = audio_recorder(pause_threshold=4.0, neutral_color="#888888")
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

if uploaded_file is not None or audio_bytes is not None:
    # Use uploaded_file or audio_bytes as input, depending on which is available
    audio_input = uploaded_file if uploaded_file is not None else audio_bytes
    st.write("Transcribing audio...")
    text = transcribe_audio(audio_input)
    st.write("Transcription:")
    st.write(text)
    st.write("Translating...")
    translated_text = translate_text(text, selected_language)
    st.write("Translation:")
    st.write(translated_text)    
    st.write("Synthesizing speech")
    response = synthesize_speech(translated_text, selected_language_code, voice_name)
    st.write(f"Speech synthesized in {selected_language}")
    if response.audio_content:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(response.audio_content)
            f.seek(0)
            audio_file = open(f.name, "rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mpeg")