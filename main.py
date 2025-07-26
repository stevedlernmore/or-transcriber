import streamlit as st
from openai import OpenAI

AI_client = OpenAI(api_key=st.secrets["OPEN_AI_KEY"])

st.title("🎙️🩺⛑️ Operating Room Transcriber")

@st.cache_data
def load_prompts(filename):
    prompts = {}
    with open(filename, "r") as file:
        lines = file.read().split('\n\n')
        for block in lines:
            if block.strip():
                procedure, details = block.split(":", 1)
                prompts[procedure.strip()] = details.strip()
    return prompts

prompts = load_prompts("obgyn_prompts.txt")

if 'transcription_text' not in st.session_state:
    st.session_state.transcription_text = ""

if 'dictated_text' not in st.session_state:
    st.session_state.dictated_text = ""

uploaded_file = st.file_uploader(
    "Upload audio/video file for transcription",
    type=["mp3", "mp4", "wav", "mov", "m4a"]
)

if uploaded_file:
    st.audio(uploaded_file)

    if st.button("Transcribe"):
        with st.spinner("Transcribing via OpenAI Whisper API..."):
            transcription_response = AI_client.audio.transcriptions.create(
                model="whisper-1",
                file=(uploaded_file.name, uploaded_file, uploaded_file.type)
            )
            st.session_state.transcription_text = transcription_response.text

if st.session_state.transcription_text:
    st.subheader("Transcription")
    st.text_area("Transcript:", st.session_state.transcription_text, height=200)

selected_procedure = st.selectbox(
    "Select Procedure or Type to Search:", 
    options=list(prompts.keys()),
    index=0
)

st.session_state.dictated_text = st.text_area("Dictated operative details (type here):", st.session_state.dictated_text, height=150)

if selected_procedure:
    surgery_details = prompts[selected_procedure]

    prompt = f"""
Procedure: {selected_procedure}

{surgery_details}

Dictated Operative Details: {st.session_state.dictated_text}

Operative Transcription:
{st.session_state.transcription_text}

Generate a concise operative summary incorporating all relevant details above.
"""

    summary_prompt = st.text_area("Review and modify summary prompt if necessary:", prompt, height=400)

    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            completion = AI_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": summary_prompt}]
            )

            summary_text = completion.choices[0].message.content

            st.success("✅ Summary Generated!")
            st.text_area("Operative Summary:", summary_text, height=300)
