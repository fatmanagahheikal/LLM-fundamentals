import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MAX_MESSAGES = 4

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

messages = [
    {"role": "system", "content": "You are a friendly assistant. Keep answers short."}
]

print("Chatbot ready! Type 'exit' to quit.")

while True:
    user_input = input("You: ")

    if user_input == "exit":
        break

    messages.append({"role": "user", "content": user_input})


    if len(messages) > MAX_MESSAGES + 1:
        messages = [messages[0]] + messages[-MAX_MESSAGES:]

    response = client.chat.completions.create(
        model="deepseek/deepseek-v4.1-flash",
        messages=messages,
        max_tokens=1000,
    )

    reply = response.choices[0].message.content
    print("Bot:", reply)
    print(f"[prompt tokens: {response.usage.prompt_tokens}]")
    print(f"[completion tokens: {response.usage.completion_tokens}]")

    messages.append({"role": "assistant", "content": reply})