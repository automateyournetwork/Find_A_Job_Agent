import os
import logging
import streamlit as st
import urllib3
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI

# ✅ IMPORT FINDWORK AGENT
from findwork_agent import find_jobs_tool

# ============================================================
# **🚀 Load Environment Variables**
# ============================================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ============================================================
# **🔧 Configure Logging & Security**
# ============================================================
logging.basicConfig(level=logging.INFO)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============================================================
# **🤖 Define the LLM**
# ============================================================
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.1)

# ============================================================
# **🌍 Initialize Findwork Agent**
# ============================================================
findwork_agent_tool = Tool(
    name="Findwork Jobs",
    func=find_jobs_tool.func,  # Directly using the existing Findwork tool
    description=find_jobs_tool.description,
)

# ============================================================
# **🤖 Main Parent Routing Agent**
# ============================================================
parent_tools = [findwork_agent_tool] 

parent_agent = initialize_agent(
    tools=parent_tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

logging.info(f"🚀 FAJA (Find A Job Agent) Initialized with Tools: {[tool.name for tool in parent_tools]}")

# ============================================================
# **🎯 Streamlit UI - Chat with FAJA**
# ============================================================
st.title("💼 Find a Job with FAJA")
st.write("🔍 Use natural language to search for jobs across multiple APIs!")

# User input text area
user_input = st.text_area("📝 Enter your job search query:")

# Conversation History (Stored in Session)
if "conversation" not in st.session_state:
    st.session_state.conversation = []

if st.button("Search Jobs"):
    if not user_input:
        st.warning("⚠️ Please enter a job search query.")
    else:
        # 🚀 Invoke the Parent Agent
        response = parent_agent.invoke(user_input)

        # ✅ Ensure response is a dictionary
        if isinstance(response, dict) and "results" in response:  
            job_listings = response.get("results", [])
            if job_listings:
                st.write(f"### 📋 Job Listings for: **{user_input}**")
                for job in job_listings:
                    st.write(f"#### 🏢 **{job['company_name']}** - {job['role']}")
                    st.write(f"📍 **Location:** {job['location']}")
                    st.write(f"🌍 **Remote:** {'Yes' if job['remote'] else 'No'}")
                    st.write(f"📅 **Posted On:** {job['date_posted']}")
                    st.write(f"🔗 [Job Link]({job['url']})")
                    st.write("---")
            else:
                st.write("❌ No jobs found for your query.")
        elif isinstance(response, str):  
            st.write(f"### 📡 Response: {response}")
        else:
            st.write("❌ No valid response received.")

        # ✅ Save conversation history
        st.session_state.conversation.append({"role": "user", "content": user_input})
        st.session_state.conversation.append({"role": "assistant", "content": response})

# ============================================================
# **📜 Display Conversation History**
# ============================================================
st.write("### 💬 Conversation History")
for chat in st.session_state.conversation:
    role = "🧑‍💼 You" if chat["role"] == "user" else "🤖 FAJA"
    st.write(f"**{role}:** {chat['content']}")
