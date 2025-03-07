import gradio as gr
import openai
from PIL import Image
import numpy as np
from gtts import gTTS
import os

# Initialize OpenAI API (replace 'your-api-key' with your actual API key)
def get_env():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Please enter your OpenAI API key: ")
        if not api_key:
            raise ValueError("API key not provided.")
    openai.api_key = api_key

# Function to process image and get caption
def get_caption(image):
    # Convert image to bytes
    image_bytes = np.array(image).tobytes()
    
    # Send image to OpenAI for captioning
    response = openai.Image.create(
        file=image_bytes,
        model="image-alpha-001"  # Use the correct model for image captioning
    )
    
    caption = response['choices'][0]['text']
    return caption

# Function to convert text to speech
def text_to_speech(text):
    tts = gTTS(text)
    audio_file = 'caption.mp3'
    tts.save(audio_file)
    return audio_file

# Gradio app
def image_captioning_app(image):
    get_env()
    image = Image.fromarray(image)
    caption = get_caption(image)
    audio_file = text_to_speech(caption)
    return caption, audio_file

iface = gr.Interface(fn=image_captioning_app, inputs=gr.Image(type="numpy"), outputs=["text", "audio"], live=True)
iface.launch()