import os
import time
import logging
import requests
import re
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# ============================================================
# **🚀 Load Environment Variables**
# ============================================================
load_dotenv()
FINDWORK_API_KEY = os.getenv("FINDWORK_API_KEY")
FINDWORK_API_URL = "https://findwork.dev/api/jobs/"

# Validate API Key
if not FINDWORK_API_KEY:
    logging.error("❌ Findwork API key is missing! Please check your .env file.")
    raise ValueError("Findwork API key is required to use the job search agent.")

# Configure logging
logging.basicConfig(level=logging.INFO)

# ============================================================
# **🔍 Job Search Class**
# ============================================================
class FindJobs:
    """Class to interact with the Findwork job search API."""

    def __init__(self):
        self.api_url = FINDWORK_API_URL
        self.api_key = FINDWORK_API_KEY
        self.headers = {"Authorization": f"Token {self.api_key}"}

    def search_jobs(self, search="", location=None, sort_by="date", page=1):
        """
        Fetch job listings from Findwork API based on criteria.
        """
        params = {
            "search": search,
            "location": location,
            "sort_by": sort_by,
            "page": page
        }
        retries = 3
        for attempt in range(retries):
            try:
                logging.info(f"🔍 Sending Findwork API request: {params}")
                response = requests.get(self.api_url, headers=self.headers, params=params, timeout=5)
                response.raise_for_status()
                job_results = response.json()

                # Debugging: Log response
                logging.info(f"✅ Findwork API Response: {job_results}")

                return job_results
            except requests.exceptions.RequestException as e:
                logging.error(f"❌ Findwork API request failed (Attempt {attempt+1}): {e}")
                time.sleep(2)  # Backoff before retrying
        return {"error": "Failed to fetch job listings after multiple attempts."}


# ============================================================
# **🧠 Natural Language Processing (NLP) for User Queries**
# ============================================================
def parse_user_input(user_input):
    """
    Converts natural language job queries into structured API parameters.
    """
    parsed_query = {"search": "", "location": None, "sort_by": "date", "page": 1}

    # Extract job role & location
    match_location = re.search(r'in (\w+)', user_input, re.IGNORECASE)
    match_role = re.search(r'for a (\w+)', user_input, re.IGNORECASE)

    if match_location:
        parsed_query["location"] = match_location.group(1)

    if match_role:
        parsed_query["search"] = match_role.group(1)

    if "remote" in user_input.lower():
        parsed_query["search"] = "remote"

    if re.search(r'\b(\d+) open jobs\b', user_input):
        parsed_query["page"] = 1  # Ensure at least one page is retrieved

    logging.info(f"📊 Parsed Query: {parsed_query}")
    return parsed_query


# ============================================================
# **🔧 LangChain Tool for Job Search**
# ============================================================
def search_jobs_tool(input_text):
    """
    Parses user input, converts it to API parameters, and fetches job results.
    """
    find_jobs = FindJobs()
    parsed_input = parse_user_input(input_text) if isinstance(input_text, str) else input_text

    # ✅ Call API with structured query
    job_results = find_jobs.search_jobs(**parsed_input)

    # ✅ Handle API Response
    if "results" in job_results and job_results["results"]:
        formatted_jobs = []
        for job in job_results["results"][:5]:  # Limit to 5 jobs
            formatted_jobs.append(
                f"🏢 **{job['company_name']}** - {job['role']}\n"
                f"📍 **Location:** {job['location']}\n"
                f"🌍 **Remote:** {'Yes' if job['remote'] else 'No'}\n"
                f"📅 **Posted:** {job['date_posted']}\n"
                f"🔗 [View Job]({job['url']})\n"
            )
        return "\n\n".join(formatted_jobs)

    return "❌ No job listings found. Try different keywords or location."


find_jobs_tool = Tool(
    name="find_jobs_tool",
    description="Search for job listings using the Findwork API. Provide keywords, location, sorting preference, and page number.",
    func=search_jobs_tool
)

# ============================================================
# **🤖 Define the AI Agent**
# ============================================================
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.1)

tools = [find_jobs_tool]

tool_names = ", ".join([tool.name for tool in tools])

tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

# Define Prompt Template
agent_prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template=f"""
    You are an AI-powered job search assistant. Your goal is to help users find jobs based on their preferences.

    **TOOLS AVAILABLE:**
    {find_jobs_tool.description}
    
    **FORMAT:**  
    Thought: [Your reasoning]  
    Action: [Choose a tool]  
    Action Input: [Provide the necessary parameters]  
    Observation: [Result]  
    Final Answer: [Answer to the User]  

    **Example Queries:**
    - Find software engineering jobs in Toronto.
      Thought: I need to search for jobs based on the given criteria.
      Action: find_jobs_tool
      Action Input: {{"search": "software engineer", "location": "Toronto", "sort_by": "date", "page": 1}}
      Observation: [List of jobs]
      Final Answer: Here are the available software engineering jobs in Toronto.

    **Begin!**
    
    Question: {{input}}
    
    {{agent_scratchpad}}
    """
)

# Create the ReAct Agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=agent_prompt.partial(tool_names=tool_names, tools=tool_descriptions)
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,
    verbose=True,
    max_iterations=5,
    max_execution_time=30
)

# Log initialization
logging.info("🚀 Weather Agent initialized.")

# ============================================================
# **📌 Example Execution for Testing**
# ============================================================
if __name__ == "__main__":
    queries = [
        "Find Python developer jobs in Toronto.",
        "Find remote data scientist jobs.",
        "Find React developer jobs in San Francisco sorted by relevance."
    ]
    for query in queries:
        response = agent_executor.invoke({"input": query})
        print(f"Agent Response for '{query}':", response)
