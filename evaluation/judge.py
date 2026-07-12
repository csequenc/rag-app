import json
import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=API_KEY)


def build_judge_prompt(result, record):
    context = ""

    for chunk in result["retrieved_chunks"]:
        context += (
            f"Source: {chunk['source']}\n"
            f"{chunk['text']}\n\n"
        )

    prompt = f"""
You are an impartial AI evaluator.

Evaluate the generated answer using ONLY:
1. The user question.
2. The expected answer.
3. The retrieved context.

Do not use outside knowledge.

Return ONLY valid JSON in the following format:

{{
    "correct": true,
    "grounded": true,
    "hallucination": false,
    "completeness_score": 0,
    "overall_score": 0,
    "reason": ""
}}

Question:
{record["question"]}

Expected Answer:
{record["expected_answer"]}

Generated Answer:
{result["answer"]}

Retrieved Context:
{context}
"""

    return prompt


def judge_generation(result, record):
    prompt = build_judge_prompt(result, record)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    judge_output = response.choices[0].message.content

    try:
        judge_result = json.loads(judge_output)

    except json.JSONDecodeError:
        judge_result = {
            "correct": False,
            "grounded": False,
            "hallucination": True,
            "completeness_score": 0,
            "overall_score": 0,
            "reason": "Judge returned invalid JSON."
        }

    return judge_result