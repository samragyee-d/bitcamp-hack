from google import genai
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Function to get the response from the Gemini model based on input text
def generate_gemini_response(input_text):
    # Retrieve the API key from the environment variable
    api_key = os.getenv("API_KEY")

    # Initialize the Gemini client using the API key
    client = genai.Client(api_key=api_key)

    # Generate content using the Gemini model
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=input_text
    )

    # Return the generated response text
    return response.text

# Example usage:
input_text = "Hi! My name is Alvia."
response = generate_gemini_response(input_text)
print(response)
