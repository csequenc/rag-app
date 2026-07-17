from multiprocessing import context
import os
import requests

from dotenv import load_dotenv
from langchain.tools import tool
from langchain.tools import tool
from framework_rag import retriever,reranker

load_dotenv()

@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Use only for arithmetic calculations.
    
    """
    return str(eval(expression))


@tool
def weather(city: str) -> str:
    """
    Get the current weather for a city.
    """

    api_key = os.getenv("OPENWEATHER_API_KEY")

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    response = requests.get(url, params=params, timeout=10)

    data = response.json()
    
    print(os.getenv("OPENWEATHER_API_KEY"))
    print(response.status_code)
    print(data)

    temperature = data["main"]["temp"]
    condition = data["weather"][0]["description"]

    return f"""
            Current weather in {city}

            Temperature: {temperature}°C
            
            Condition: {condition}
            
            """

@tool
def rag_search(question: str) -> str:
    """
    Search the uploaded knowledge base for information relevant to the user's question.

    Use this tool whenever the answer depends on the uploaded documents.
    Input should be the user's question.
    The tool returns relevant document excerpts as plain text that should be used to answer the user.
    """

    results = retriever.search(
        question,
        top_k=5
    )

    # results = reranker.rerank(
    #     question,
    #     results
    # )

    context = "\n\n".join(
    chunk["text"]
    for chunk in results
    )

    return context

