import os
from dotenv import load_dotenv

from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage
from fastapi import FastAPI
import streamlit as st

load_dotenv()
brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")


# 1. Tool for search
def search(query):
    """Search Google using the Serper API

    Args:
        query (str): Search query string

    Returns:
        str: JSON response from Serper API containing search results

    Example:
        response = search("python tutorials")
        print(response)
    """
    url = "https://google.serper.dev/search"

    payload = json.dumps({"q": query})

    headers = {"X-API-KEY": serper_api_key, "Content-Type": "application/json"}

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

    return response.text


# TODO: change to Langchain default Google Serper API https://python.langchain.com/docs/integrations/tools/google_serper

# # Test tool
# search("What is meta's thread product?")


# 2. Tool for scrape
def scrape_website(objective: str, url: str):
    """Scrape a website and summarize contents if content is too large.

    Args:
        objective (str): The key objective for summarization.
        url (str): The URL of the website to scrape.

    Returns:
        str: Summarized content from the website if content is > 10k chars, otherwise the full text.

    Raises:
        HTTPError: If the API request fails.

    """

    print("Scraping website...")
    # Define the headers for the request
    headers = {
        "Cache-Control": "no-cache",
        "Content-Type": "application/json",
    }

    # Define the data to be sent in the request
    data = {"url": url}

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    post_url = f"https://chrome.browserless.io/content?token={brwoserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)

    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENT:", text)

        if len(text) > 10000:
            output = summary(objective, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")


def summary(objective: str, content: str):
    """
    Generate a summary of the given content focused on the provided objective.

    Uses a GenAI model with a MapReduce summarization chain
    from Langchain to produce the summary.

    Args:
        objective (str): The key objective to guide the summarization.
        content (str): The content to summarize.

    Returns:
        str: The generated summary string.

    """
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500
    )
    docs = text_splitter.create_documents([content])

    # Summarize the content using a MapReduce summarization chain
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """

    # Create a prompt template for the map and combine prompts
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"]
    )

    # Create a summary chain to summarize each chunk and merge together into one paragraph
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True,
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output


# # Test tool
# scrape_website("What is meta's thread product?", "https://meta.discourse.org/t/what-is-metas-thread-product/198046")


# 3. Langchain agent to use tools
class ScrapeWebsiteInput(BaseModel):
    """Define hte inputs for scrape_website"""

    objective: str = Field(
        description="The objective & task that users give to the agent"
    )
    url: str = Field(description="The url of the website to be scraped")


class ScrapeWebsiteTool(BaseTool):
    """Define the tool to scrape a website"""

    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        """If there is an error, raise an exception"""
        raise NotImplementedError("error here")


# Create a tool list that can be pass into agent
tools = [
    Tool(
        name="Search",
        func=search,
        description="useful for when you need to answer questions about current events, data. You should ask targeted questions",
    ),
    ScrapeWebsiteTool(),
]

# Create system messege that will pass into agent prior to user input
updated_system_message = SystemMessage(
    content="""You are an unbiased researcher, who deeply understands context and implications. 

            Please follow these rules to produce high-quality, factual research on any objective: 
            1. Thoroughly research the objective from multiple authoritative sources, considering historical and social contexts.
            2. Dig deeper by scraping relevant sites to extract key statistics, examples, and verbatim quotes that support the facts.
            3. Iteratively analyze information and bridge connections to related concepts for a comprehensive understanding. Ask "what additional data is needed?"
            4. Synthesize research into clear, nuanced explanations using your own words. Reference all sources and data points.
            5. Focus on factual accuracy and logic. Do not make unsupported assumptions. Challenge your own biases.
            6. Aim for completeness and concision. Include reference links and cite sources.
            7. In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research"""
)

system_message = SystemMessage(
    content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
            you do not make things up, you will try as hard as possible to gather facts & data to back up the research
            
            Please make sure you complete the objective above with the following rules:
            1/ You should do enough research to gather as much information as possible about the objective
            2/ If there are url of relevant links & articles, you will scrape it to gather more information
            3/ After scraping & search, you should think "is there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iteratins
            4/ You should not make things up, you should only write facts & data that you have gathered
            5/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
            6/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research"""
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000
)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)

# 4. Use streamlit to create a web app
# def main():
#     st.set_page_config(page_title="AI research agent", page_icon=":bird:")

#     st.header("AI research agent :bird:")
#     query = st.text_input("Research goal")

#     if query:
#         st.write("Doing research for ", query)

#         result = agent({"input": query})

#         st.info(result['output'])


# if __name__ == '__main__':
#     main()


# 4. Set this as an API endpoint via FastAPI
app = FastAPI()


class Query(BaseModel):
    query: str


@app.post("/")
def researchAgent(query: Query):
    query = query.query
    content = agent({"input": query})
    actual_content = content["output"]
    return actual_content
# uvicorn app:app --host 0.0.0.0 --port 10000