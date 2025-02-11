import streamlit as st
import os, tempfile
from google.cloud import speech, texttospeech
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from audio_recorder_streamlit import audio_recorder
import time

load_dotenv()

def convert_audio_to_text(audio_bytes, client):
    audio = speech.RecognitionAudio(content=audio_bytes)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="en-IN",
        model="command_and_search",
        audio_channel_count=2,
    )
    operation = client.long_running_recognize(config=config, audio=audio)
    conversion = operation.result(timeout=90)
    return conversion.results[0].alternatives[0].transcript if conversion.results else ""

def load_and_create_vector_db(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    vectorstore = DocArrayInMemorySearch.from_documents(docs, embeddings)
    return vectorstore

def perform_sentiment_analysis(llm, text):
    sentiment_prompt = f"""
    Analyze the sentiment of the following text and provide a structured output.
    Identify the overall sentiment (positive, negative, neutral), the dominant emotions,
    the tone, and the urgency level. Only return this structured output and nothing else.
    Text: {text}
    Output format:
    {{ "sentiment": "<positive/negative/neutral>", "emotions": ["<emotion1>", "<emotion2>"], "tone": "<tone>", "urgency": "<low/medium/high>", "explanation": "<reasoning>" }}
    """
    return llm.invoke(sentiment_prompt)

def search_vector_db(vectorstore, query):
    retriever = vectorstore.as_retriever()
    relevant_docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in relevant_docs])

def generate_response(llm, context, query, sentiment):
    response_prompt = f"""
    Generate an empathetic response based on the given user query, retrieved context, and sentiment analysis.
    Maintain a tone that aligns with the user's emotions while providing a helpful and comprehensive answer in at least 300 words. 
    Only generate the final answer and not any other text:
    Context: {context}
    Question: {query}
    Sentiment Analysis: {sentiment}
    """
    return llm.invoke(response_prompt).content

def convert_text_to_speech(client, text, output_path="output.wav"):
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code="en-US", name="en-US-Studio-M")
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)
    response = client.synthesize_speech(request={"input": input_text, "voice": voice, "audio_config": audio_config})
    with open(output_path, "wb") as out:
        out.write(response.audio_content)

def main():
    st.header("Audio to Audio RAG Application")
    uploaded_file = st.file_uploader("Choose your PDF file")
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_pdf_path = temp_file.name
        
        vectorstore = load_and_create_vector_db(temp_pdf_path)
        llm = ChatOllama(model="llama3.2:1b")
        
        audio_bytes = audio_recorder(pause_threshold=10, recording_color="#6aa36f", neutral_color="#e82c58", icon_size="2x")
        if audio_bytes:
            start_time = time.perf_counter()
            user_prompt = convert_audio_to_text(audio_bytes, speech_client)
            st.write("User Input:", user_prompt)
            
            context = search_vector_db(vectorstore, user_prompt)
            st.write("Context:", context)
            sentiment = perform_sentiment_analysis(llm, user_prompt)
            st.write("Sentiments:", sentiment.content)
            final_response = generate_response(llm, context, user_prompt, sentiment.content)
            st.write("Generated Response:", final_response)
            
            convert_text_to_speech(tts_client, final_response)
            st.success("Audio saved successfully as output.wav!")
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            st.success(f"Latency : {execution_time}")


if __name__ == '__main__':
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        speech_client = speech.SpeechClient()
        tts_client = texttospeech.TextToSpeechClient()
        main()
    else:
        st.error("Google credentials not set.")
