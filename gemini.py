# gemini.py
import os
from dotenv import load_dotenv
from google import genai
from state import chat_history  # Import the shared chat history

load_dotenv()

# Configure Gemini API key
api_key = os.getenv("API_KEY")
client = genai.Client(api_key=api_key)

def generate_gemini_response(input_text):
    # Add user input to history
    chat_history.append({"role": "user", "content": input_text})

    # Build prompt messages (Gemini doesn't use role messages exactly like OpenAI)
    # But we can approximate with text-based context
    prompt_parts = [
        (
            "You are Eva, a compassionate and intelligent virtual assistant designed to support users emotionally. "
            "Speak in a calm, empathetic tone. Respond with warmth, using phrases like 'You’ve got this' or 'I'm here for you.'"
            "Shorter responses are best, and no formatting is allowed."
        )
    ]

    # Add each history message in a flat string format (since Gemini expects plain strings in most basic usage)
    for message in chat_history:
        role = message["role"].capitalize()
        content = message["content"]
        prompt_parts.append(f"{role}: {content}")

    # Generate response
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt_parts)


    # Add response to chat history
    response_text = response.text.strip()
    chat_history.append({"role": "assistant", "content": response_text})

    return response_text
