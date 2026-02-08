from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_structured_data(prompt: str):
    """Generate structured data from a given prompt using OpenAI's GPT model."""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": """You are a helpful assistant that converts
              text into structured JSON data. Return only the json object structured as below:
              {
                  "name": str,
                  "title": str,
                  "summary": str,
                  "skills": [str],
                  "experience": [{
                      "role": str,
                      "company": str,
                      "description": str
                  }],
                  "education": [str]
              }
              Ensure the JSON is properly formatted without any additional text."""},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        response_format = {"type": "json_object"}
    )
    
    return response.choices[0].message.content