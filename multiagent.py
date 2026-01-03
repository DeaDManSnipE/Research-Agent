from crewai import Agent, Task, Crew
from crewai_tools import TavilySearchTool
from textwrap import dedent
from crewai.llm import LLM
from dotenv import load_dotenv
import os

load_dotenv()

llm = LLM(
    model = "gemini-2.5-flash",
    google_api_key = os.getenv("GEMINI_API_KEY"),
    temperature = 0.5
)

search_tool = TavilySearchTool(
    tavily_api_key = os.getenv("TAVILY_API_KEY"),
    max_results = 3
)

research_agent = Agent(
    role = "Researcher",
    goal = "Gather accurate and relevant information on assigned topics.",
    backstory = dedent("""You are an expert researcher. You focus on correctness, facts and clarity of concepts. You use the tools provided for finding the latest and most accurate information."""),
    llm = llm,
    tools = [search_tool],
    verbose = True
)

writer_agent = Agent(
    role = "Writer",
    goal = "Write clear and helpful resources based on research.",
    backstory = dedent("""You are a skilled writer who explains idea simply and naturally like a friendly chatbot. You provide the answers in paragraphs and use bullet points where it is appropriate."""),
    llm = llm,
    verbose = True
)

reviewer_agent = Agent(
    role = "Reviewer",
    goal = "Improve quality, correctness and tone of responses.",
    backstory = dedent("""You review answers to ensure they are accurate, easy to understand, and well structured. Remove any repetitions. You ensure that each point only appears once. You should not add any extra points."""),
    llm = llm,
    verbose = True
)

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting")
        break
    
    research_task = Task(
        description = f"""The user asked: "{user_input}". Research the topic and identify key points needed for a correct answer.""",
        expected_output = "Key facts or explanations relevant to the user's question.",
        agent = research_agent
    )
    
    write_task = Task(
        description = """Using the research provided earlier, write a clear and friendly chatbot response.""",
        expected_output = "A clear and helpful chatbot-style answer.",
        agent = writer_agent
    )
    
    review_task = Task(
        description = """Review the response and improve:
        -clarity
        -correctness
        -tone
        Make sure it sounds natural and helpful""",
        expected_output = "A polished final chatbot response.",
        agent = reviewer_agent
    )
    
    crew = Crew(
        agents = [research_agent, writer_agent, reviewer_agent],
        tasks = [research_task, write_task, review_task],
        verbose = False
    )
    
    final_response = crew.kickoff()
    
    print("\n"+"="*180)
    print("Bot: ",final_response)
    print("="*180+"\n")