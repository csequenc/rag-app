from groq import Groq


class Planner:

    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)

    def decide(self, query):

        prompt = f"""
You are an AI agent.

Available tools:
1. calculate(expression) -> Performs mathematical calculations.

If the user needs calculation, respond ONLY in this format:

TOOL: calculate
INPUT: <expression>

Otherwise respond ONLY:

TOOL: none
INPUT: none

User: {query}
"""

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content

    def respond(self, query, observation):

        prompt = f"""
User asked:

{query}

The tool returned:

{observation}

Write the final answer for the user.
"""

    response = self.client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content
