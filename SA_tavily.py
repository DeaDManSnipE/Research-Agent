from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

tavily_search = TavilySearchAPIWrapper(
    tavily_api_key = "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"
)

search_tool = TavilySearchResults(
    api_wrapper = tavily_search,
    max_results = 3
)

response = search_tool.invoke("What is todays weather at Aluva?")

print(response)