from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_classic.agents.tool_calling_agent.base import create_tool_calling_agent
from langchain_classic.agents import AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    google_api_key = os.getenv("GEMINI_API_KEY"),
    temperature = 0.3
)

tavily_search = TavilySearchAPIWrapper(
    tavily_api_key = os.getenv("TAVILY_API_KEY")
)

search_tool = TavilySearchResults(
    api_wrapper = tavily_search,
    max_results = 3
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant. Use tools when needed. You have to summarize the results in bullet point format. Include line breaks."),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad")
])

agent = create_tool_calling_agent(
    llm = llm,
    tools = [search_tool],
    prompt = prompt
)

executor = AgentExecutor(
    agent = agent,
    tools = [search_tool],
    verbose = False
)

def format_output(result):
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, list):
        texts = []
        for item in result:
            if isinstance(item, dict):
                if "text" in item:
                    texts.append(item["text"])
            elif isinstance(item, str):
                texts.append(item)
        return "\n".join(t.strip() for t in texts if t.strip())
    if isinstance(result, dict):
        for key in ("text", "content", "output"):
            if key in result:
                return format_output(result[key])
    return str(result).strip()

chat_history=[]

while True:
    statement = input("You: ").strip()
    if statement.lower() in ["exit", "quit"]:
        print("Bot: Exiting...")
        break
    
    response = executor.invoke({
    "input": statement,
    "chat_history": chat_history
    })
    
    formatted_output=format_output(response["output"])

    print("Bot:", formatted_output)
    
    chat_history.append(HumanMessage(content=statement))
    chat_history.append(AIMessage(content=formatted_output))