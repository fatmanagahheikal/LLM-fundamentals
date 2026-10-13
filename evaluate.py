import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

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


def extract(email):
    response = client.chat.completions.create(
        model="deepseek/deepseek-v4.1-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": email},
        ],
        temperature=0,
        max_tokens=1000,
    )
    return response.choices[0].message.content


test_emails = [
    {"email": "Dear team, I hope you are all doing well! I truly appreciate your service. However, I would kindly like to return the keyboard and get a refund, if that is not too much trouble. Warm regards, Dina", "expected": "high"},
    {"email": "WHY IS YOUR WEBSITE SO SLOW?!?! I just wanted to check your prices!!! - Tamer", "expected": "high"},
    {"email": "Hi, I'm Rania. My laptop battery dies after 1 hour. I am traveling for a conference in two weeks and need it fixed before then.", "expected": "medium"},
    {"email": "Hi, I'm Laila. My headphones stopped working on the left side. Can you help when you have time?", "expected": "medium"},
    {"email": "Dear team, the invoice I received has the wrong address. Could you send a corrected one? Youssef", "expected": "medium"},
    {"email": "Do you deliver to Alexandria? Thanks, Nour", "expected": "low"},
    {"email": "What are your opening hours on Friday? - Hany", "expected": "low"},
    {"email": "Just wanted to say thanks, the laptop works great! Regards, Salma", "expected": "low"},
    {"email": "hii my mouse no work since yesterday pls fix i need for work tmrw!!! ali", "expected": "high"},
    {"email": "السلام عليكم، أنا محمد. الشاشة اللي اشتريتها فيها خط في النص. ممكن أعرف أعمل إيه؟", "expected": "medium"},
]

valid_json = 0
correct_urgency = 0

for i, test in enumerate(test_emails, start=1):
    result = extract(test["email"])

    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        print(f"Email {i}: ❌ invalid JSON")
        continue

    valid_json += 1
    got = data.get("urgency")

    if got == test["expected"]:
        correct_urgency += 1
        print(f"Email {i}: ✅ expected {test['expected']}, got {got}")
    else:
        print(f"Email {i}: ⚠️ expected {test['expected']}, got {got}")

print()
print(f"Valid JSON: {valid_json}/{len(test_emails)}")
print(f"Correct urgency: {correct_urgency}/{len(test_emails)}")