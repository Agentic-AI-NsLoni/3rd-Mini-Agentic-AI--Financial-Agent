import phi  # This import is necessary for phi.api
import phi.api
from phi.agent import Agent
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
from phi.model.groq import Groq
import os
from phi.playground import Playground, serve_playground_app

# Load environment variables from .env file
load_dotenv()

# Set PHI API key
phi.api = os.getenv("PHI_API_KEY")

# Ensure PHI_API_KEY is loaded correctly
if not phi.api:
    raise ValueError("PHI_API_KEY not set. Please ensure it is in the .env file.")

# Web search agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),  # You can replace this with a Gemini model if necessary
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

# Financial agent
finance_agent = Agent(
    name="Finance AI Agent",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),  # Replace with Gemini model if necessary
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True
        ),
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
)

# Create Playground App
app = Playground(agents=[finance_agent, web_search_agent]).get_app()

# Run Playground App
if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)
