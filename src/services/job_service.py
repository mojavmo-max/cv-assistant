from dotenv import load_dotenv
from openai import OpenAI
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

load_dotenv()
client = OpenAI()

def process_input(user_input):
    
    """TODO: Add extra checks to make sue url is valid and is a job description page. Also add error handling for failed requests or parsing errors."""

    result = urlparse(user_input)
    if all([result.scheme, result.netloc]):
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(user_input)
            content = page.content()
            soup = BeautifulSoup(content, 'html.parser')
            description = soup.get_text()
            browser.close()
    else:
        description = user_input

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "From the following  link provided, extract the job description information, summarize, and return it in a json structured format described as follows: {job_title: str, company: str, location: str, description: str, role_summary: str, requirements: [str], responsibilities: [str], benefits: [str]}"},
            {"role": "user", "content": description}
        ],
        temperature=0.2,
        response_format={"type": "json_object"}
    )

    return response.choices[0].message.content