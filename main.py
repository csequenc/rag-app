from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq

from framework_tools import calculate, rag_search, weather


load_dotenv()


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1
)


tools = [
    calculate,
    rag_search,
    weather
]


agent = create_agent(
    model=llm,
    tools=tools
)


def run(query: str):

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]
        }
    )

    return response["messages"][-1].content


if __name__ == "__main__":

    while True:

        query = input("\nAsk a question: ")

        if query.lower() == "exit":
            break

        answer = run(query)

        print("\nAnswer:\n")
        print(answer)