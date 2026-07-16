from groq import Groq



class Generator:

    def __init__(self, api_key, model_name="llama-3.3-70b-versatile"):
      
        self.client = Groq(api_key=api_key)
        self.model_name = model_name


    def build_prompt(self, query, results):

        context = ""

        for chunk in results:
            context += f"Source: {chunk['source']}\n"
            context += f"{chunk['text']}\n\n"

        prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the provided context.

If the answer cannot be found in the context, reply exactly:
"I don't know based on the provided documents."

Context:
{context}

Question:
{query}

Answer:
"""

        return prompt


    def generate(self, query, results):

        prompt = self.build_prompt(query, results)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content
