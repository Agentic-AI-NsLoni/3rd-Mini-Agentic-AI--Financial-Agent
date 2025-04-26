import os
from dotenv import load_dotenv
import requests
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

# Load environment variables from .env file
load_dotenv()

# Set up API keys
gemini_api_key = os.getenv("GOOGLE_GEMINI_KEY")  # Ensure the correct environment variable is set
if gemini_api_key is None:
    raise ValueError("Google Gemini API key is not set. Please provide a valid key in the .env file.")

# If you want to use Google Gemini via HTTP requests (example)
def call_gemini_api(query):
    url = "https://gemini-api-endpoint.com/query"  # Example, use the actual Gemini API URL
    headers = {
        "Authorization": f"Bearer {gemini_api_key}",
        "Content-Type": "application/json"
    }
    data = {"query": query}  # Modify according to Gemini API's requirements

    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code == 200:
        return response.json()  # Modify as per the Gemini API's response format
    else:
        return {"error": response.text}

# Web search agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tools_calls=True,
    markdown=True,
)

# Financial agent
finance_agent = Agent(
    name="Finance AI Agent",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True),
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

# Multi-agent system (combining web search and finance agent)
multi_ai_agent = Agent(
    team=[web_search_agent, finance_agent],
    instructions=["Always include sources", "Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

# Example response using the multi-agent system
multi_ai_agent.print_response("Summarize analyst recommendation and share the latest news for NVDA", stream=True)

# Example Gemini API call (replace with your actual use case)
gemini_response = call_gemini_api("What is the latest news about NVDA?")
print("Gemini Response:", gemini_response)
