from dotenv import load_dotenv
from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq

load_dotenv()

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:

    """Search the web for information"""

    return tavily_client.search(query)

system_prompt = """

You are a personal chef. The user will give you a list of ingredients they have left over in their house.

Using the web search tool, search the web for recipes that can be made with the ingredients they have.

Return recipe suggestions and eventually the recipe instructions to the user, if requested.

"""



model = init_chat_model(
    "llama-3.3-70b-versatile",
    model_provider="groq",
    temperature=1.0,
)

agent = create_agent(model=model,
                     tools=[web_search],
                system_prompt=system_prompt)

# agent = create_agent(
#     model="gpt-5-nano",
#     tools=[web_search],
#     system_prompt=system_prompt
# )

