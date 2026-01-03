from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key="AIzaSyDDoEJ4HRilb3EDLuSXhipVxuYYEaWGZkw",
    temperature=0.3
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])

statement=input()

chain = prompt | llm

response = chain.invoke({
    "question": statement
})

print(response)