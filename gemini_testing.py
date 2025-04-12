from google import genai
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Retrieve the API key from the environment variable
api_key = os.getenv("API_KEY")

# Initialize the client using the API key
client = genai.Client(api_key=api_key)

# Generate content using the Gemini model
response = client.models.generate_content(
    model="gemini-2.0-flash", contents="What is AI?"
)

print(response.text)
