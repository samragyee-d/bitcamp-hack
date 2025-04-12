#prompting gemini code goes in this file, pretty much logic goes here

from google.generativeai import GenerativeModel

model = GenerativeModel("gemini-pro")

#generating messages to send back
# if angry for more than 10 minutes, what is the response

# generate weekly summary email recap (as time permits)