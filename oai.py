import openai
import requests
import json
import os
import shutil

# Set up the OpenAI API

def get_response(prompt):
    openai.api_key = 'passhere'
    response = openai.ChatCompletion.create(
        model="gpt-4-0314",
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
    )
    return response.choices[0]['message']['content']



