from google import genai
from openai import OpenAI
from xai_sdk import Client as XAIClient
from xai_sdk.chat import user, system
from google.genai import types  

def gemini_llm_call(prompt: str, api_key: str, model="gemini-2.5-flash", top_p=1.0, temperature=0.7, top_k=0):
    client = genai.Client(api_key=api_key)
    
    # Build the config object with optional topK if > 0
    gen_config = types.GenerateContentConfig(
        temperature=temperature,
        topP=top_p,
        topK=top_k if top_k > 0 else None
    )
    
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=gen_config
    )
    return response.text.strip()

def gpt_llm_call(prompt: str, api_key: str, model="gpt-4o", top_p=1.0, temperature=0.7):
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
    )
    return response.choices[0].message.content.strip()

def grok_llm_call(prompt: str, api_key: str, model="grok-3", top_p=1.0, temperature=0.7):
    client = XAIClient(api_host="api.x.ai", api_key=api_key)
    chat = client.chat.create(model=model, temperature=temperature, top_p=top_p)
    chat.append(system("You are a medieval Irish scholar."))
    chat.append(user(prompt))
    response = chat.sample()
    return response.content.strip()
