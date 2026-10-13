import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

email = """
Hi there,
This is Sara. Just wanted to ask if you have the new laptop
model in blue color. No rush, I'm planning to buy next month.
Thanks!
"""

system_prompt = """You extract information from customer emails.
Return ONLY valid JSON with exactly these keys:
- "name": the customer's name
- "issue": a short summary of the problem
- "urgency": one of "low", "medium", "high"

Urgency rules:
- "high": the customer has a deadline within 7 days, is angry, or wants to cancel or get a refund.
- "medium": the customer has a real problem but no deadline and is calm.
- "low": the customer is only asking a question or has no problem.

Do not add any text before or after the JSON."""


response = client.chat.completions.create(
    model="deepseek/deepseek-v4.1-flash",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": email},
    ],
    temperature=0,
    max_tokens=1000,
)

result = response.choices[0].message.content


try:
    data = json.loads(result)

except json.JSONDecodeError:
    print("The model did not return valid JSON:")
    print(result)
    exit()

print("Name:", data["name"])
print("Issue:", data["issue"])
print("Urgency:", data["urgency"])

if data["urgency"] == "high":
    print("⚠️ ALERT: Send this to the manager now!")