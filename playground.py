from dotenv import load_dotenv
import os
from phi.agent import Agent
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.groq import Groq
from phi.playground import Playground, serve_playground_app

# Load environment variables from .env file
load_dotenv()

# Set PHI API key and Gemini API key (ensure these are loaded correctly)
phi.api = os.getenv("PHI_API_KEY")  # PHI API key
gemini_api_key = os.getenv("GEMINI_API_KEY")  # Gemini API key (ensure this is used)

# Web search agent (using DuckDuckGo for web search)
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),  # Adjust model if necessary
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],  # Fixed typo here
    show_tools_calls=True,
    markdown=True,
)

# Financial agent (using Yahoo Finance tools)
finance_agent = Agent(
    name="Finance AI Agent",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),  # Adjust model if necessary
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True),
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

# Playground app to interact with both agents
app = Playground(agents=[finance_agent, web_search_agent]).get_app()

# Run the Playground app (ensure it starts correctly)
if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)

