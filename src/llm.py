import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model=os.environ.get('MODEL', 'deepseek-v4-flash'),
    base_url=os.environ.get('API_BASE_URL'),
    api_key=os.environ.get('API_KEY'),
    temperature=0.0,
    timeout=120,
    extra_body={'thinking': {'type': 'disabled'}},
)
