import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

response = client.chat.completions.create(

    model="deepseek/deepseek-v4.1-flash",
        messages=[
        {"role": "system", "content": "You are a funny teacher who explains things to 10-year-old kids."},
        {"role": "user", "content": "Explain artificial intelligence in one simple paragraph."}
    ],

    temperature=0,
    max_tokens=1000,
)

print("Answer:", response.choices[0].message.content)
print("Stop reason:", response.choices[0].finish_reason)
print("Tokens used:", response.usage)