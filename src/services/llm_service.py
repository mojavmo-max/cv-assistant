from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path
import numpy as np
import json

DATA_DIR = Path("data")
CHUNKS_PATH = DATA_DIR / "chunks.txt"

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

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100):
    """Split the input text into smaller chunks of a specified size."""
    
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def load_chunks(path: Path) -> list:
    """Load existing text chunks from a file."""
    
    if path.exists():
        with open(path, 'r') as f:
            return f.read().splitlines()
    return []


def load_embeddings(text: str):
    """Generate embeddings for the given text using OpenAI's embedding model."""
    
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    
    return response.data[0].embedding


def compare_embedding_set(embedding_set_1: list, embedding_set_2: list):
    
    score = cosine_similarity(embedding_set_1, embedding_set_2)[0][0]
    return score

def process_files(cv_text: str, job_desc_text: str):
    
    cv_json = json.loads(cv_text)
    job_desc_json = json.loads(job_desc_text)

    job_summary = job_desc_json.get("role_summary", "")
    cv_summary = cv_json.get("summary", "")

    cv_chunks = chunk_text(cv_summary)
    job_desc_chunks = chunk_text(job_summary)

    cv_embeddings = [load_embeddings(chunk) for chunk in cv_chunks]
    job_desc_embeddings = [load_embeddings(chunk) for chunk in job_desc_chunks]

    score1 = compare_embedding_set(cv_embeddings, job_desc_embeddings)

    job_responsibilities = " ".join(job_desc_json.get("responsibilities", []))
    cv_skills = " ".join(cv_json.get("skills", []))

    cv_chunks = chunk_text(cv_skills)
    job_desc_chunks = chunk_text(job_responsibilities)

    cv_embeddings = [load_embeddings(chunk) for chunk in cv_chunks]
    job_desc_embeddings = [load_embeddings(chunk) for chunk in job_desc_chunks]

    score2 = compare_embedding_set(cv_embeddings, job_desc_embeddings)

    job_requirements = " ".join(job_desc_json.get("requirements", []))
    cv_experience = " ".join(
    exp.get("description", "")
    for exp in cv_json.get("experience", [])
)


    cv_chunks = chunk_text(cv_experience)
    job_desc_chunks = chunk_text(job_requirements)

    cv_embeddings = [load_embeddings(chunk) for chunk in cv_chunks]
    job_desc_embeddings = [load_embeddings(chunk) for chunk in job_desc_chunks]

    score3 = compare_embedding_set(cv_embeddings, job_desc_embeddings)

    total_score = (score1*0.4 + score2*0.4 + score3*0.2)

    return {
        "score": float(round(total_score, 4))
    }