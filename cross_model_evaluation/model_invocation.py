from google import genai
from openai import OpenAI
from xai_sdk import Client as XAIClient
from xai_sdk.chat import user, system

def gemini_llm_call(prompt: str, api_key: str, model="gemini-2.5-flash"):
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )
    return response.text.strip()

def gpt_llm_call(prompt: str, api_key: str, model="gpt-4o") -> str:
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def grok_llm_call(prompt: str, api_key: str, model="grok-3") -> str:
    client = XAIClient(api_host="api.x.ai", api_key=api_key)
    chat = client.chat.create(model=model)
    chat.append(system("You are a medieval Irish scholar."))
    chat.append(user(prompt))
    response = chat.sample()
    return response.content.strip()