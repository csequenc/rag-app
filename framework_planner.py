from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from schemas import PlannerDecision

load_dotenv()  # Load environment variables from .env file

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

structured_llm = llm.with_structured_output(PlannerDecision)

planner_prompt = PromptTemplate.from_template(

"""
You are an AI assistant that helps choose tools from the following list based on the user's request.

Available tools:

{tools}

User Query:

{query}

"""
)


def plan(query: str, tools: str) -> PlannerDecision:

    formatted_prompt = planner_prompt.format(
        query=query,
        tools=tools
    )

    decision = structured_llm.invoke(formatted_prompt)

    return decision