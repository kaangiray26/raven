import re
import tiktoken
from typing import List
from lib.waiter import get_config
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.callbacks import CallbackManagerForRetrieverRun

def iter_words(text):
    return re.finditer(r'[^\s]+', text)

def format_docs_with_id(docs: List[Document]) -> str:
    formatted = [
        f"Source ID: {i}\nArticle Title: {doc.metadata['title']}\nSource URL: {doc.metadata['source_url']}\n\nArticle Snippet: {doc.page_content}"
        for i, doc in enumerate(docs)
    ]
    return "\n\n" + "\n\n".join(formatted)

class Topics(BaseModel):
    topics: list[str] = Field(
        default=[],
        description="List of topics extracted from the query"
    )

class Queries(BaseModel):
    queries: list[str] = Field(
        default=[],
        description="List of search queries generated from the paper content"
    )

class TitleAbstract(BaseModel):
    title: str = Field(
        default="",
        description="Title of the paper"
    )
    abstract: str = Field(
        default="",
        description="Abstract of the paper"
    )

class CitedAnswer(BaseModel):
    # Answers is a list of answer dictionaries, each with the answer and the citations
    answer: str = Field(
        default="",
        description="The answer to the user question, which is based only on the given sources."
    )
    citation: int = Field(
        default=0,
        description="The integer ID of the SPECIFIC source which justify the answer.",
    )

class CitedAnswers(BaseModel):
    # answers is a list of CitedAnswer objects
    answers: List[CitedAnswer] = Field(
        default=[],
        description="List of answers to the user question, each with their own answer and the citation."
    )

class CustomRetriever(BaseRetriever):
    docs: List[Document]
    k: int = 5

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        return self.docs

class Keywords(BaseModel):
    keywords: list[str] = Field(
        default=[],
        description="List of keywords extracted from the query in a sorted order from general to specific."
    )

class Chat:
    def __init__(self):
        # Set the configuration
        self.config = get_config()

        # Models
        self.max_tokens = 200000
        self.model = ChatOpenAI(
            model=self.config["chat"]["openai_chat_model"],
            api_key=self.config["api_keys"]["openai"],
            model_kwargs={
                "response_format": {
                    "type":"json_object"
                }
            }
        )

    def get_chunk_size(self, text):
        encoding = tiktoken.encoding_for_model(self.config["chat"]["openai_chat_model"])
        return len(encoding.encode(text))

    def load_document(self, filepath):
        loader = PyPDFLoader(filepath)
        return loader.load()

    async def load_document_truncated(self, filepath, max_tokens):
        loader = PyPDFLoader(filepath)
        context = ""
        page_added = 0
        context_size = 0
        async for page in loader.alazy_load():
            for word in iter_words(page.page_content):
                chunk_size = self.get_chunk_size(f"{word.group()} ")
                if context_size + chunk_size > max_tokens:
                    break
                context += f"{word.group()} "
                context_size += chunk_size
            page_added += 1
        return context

    # Generate a list of topics from a research question
    async def get_topics_from_query(self, query):
        # Prompt template
        system_prompt = "You are a literature researcher in the computer science field. Extract topics from the following research question in the computer science field to generate a search query. {format_instructions}\n\n{text}"

        # Construct the parser
        parser = JsonOutputParser(pydantic_object=Topics)

        # And the prompt
        prompt = PromptTemplate(
            template=system_prompt,
            input_variables=["text"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        # Set kwargs
        self.model.model_kwargs={
            "response_format": {
                "type":"json_object"
            }
        }

        # Construct the chain
        chain = prompt | self.model | parser
        return chain.invoke({
            "text": query
        })

    # Generate a list of search queries from a paper content
    async def get_queries_from_paper(self, title, abstract, filepath):
        # Prompt template
        system_prompt = """
        You are a literature researcher in the computer science field. You have been given a paper and your goal is to generate search queries by extracting the main topics from the paper content. Try to generate search queries that are relevant to the paper content.{format_instructions}

        Title: {title}

        Abstract: {abstract}

        {context}
        """

        # Construct the parser
        parser = JsonOutputParser(pydantic_object=Queries)

        # And the prompt
        prompt = PromptTemplate(
            template=system_prompt,
            input_variables=["title", "abstract", "context"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        # Get the prompt context size,
        # so we can truncate the document to fit the prompt
        prompt_context_size = self.get_chunk_size(prompt.format(
            format_instructions=parser.get_format_instructions(),
            title=title,
            abstract=abstract,
            context=""
        ))

        # Get document context
        context = await self.load_document_truncated(filepath, self.max_tokens - prompt_context_size)

        # Set kwargs
        self.model.model_kwargs={
            "response_format": {
                "type":"json_object"
            }
        }

        # Construct the chain
        chain = prompt | self.model | parser
        return chain.invoke({
            "title": title,
            "abstract": abstract,
            "context": context
        })

    async def generate_title_abstract_from_query(self, query):
        # Prompt template
        system_prompt = "You are a literature researcher in the computer science field. Generate a title and abstract for a paper based on the following research question. {format_instructions}\n\n{text}"

        # Construct the parser
        parser = JsonOutputParser(pydantic_object=TitleAbstract)

        # And the prompt
        prompt = PromptTemplate(
            template=system_prompt,
            input_variables=["text"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        # Set kwargs
        self.model.model_kwargs={
            "response_format": {
                "type":"json_object"
            }
        }

        # Construct the chain
        chain = prompt | self.model | parser
        return chain.invoke({
            "text": query
        })

    async def rag_with_citations(self, query, papers):
        # Prepare the documents
        docs = [
            Document(
                page_content=paper["abstract"],
                metadata={
                    "title": paper["title"],
                    "source_url": paper["source_url"]
                }
            )
            for paper in papers if paper["abstract"]
        ]

        # Prepare a retriever that just returns the documents
        retriever = CustomRetriever(docs=docs)

        # Prompt template
        system_prompt = """
        You are a literature researcher in the computer science field. You have been given a list of paper titles and abstract, and a research question. Your goal is to generate responses to the research question by reading the paper abstracts. Don't use any formatting in the response, just write the responses as plain text.
        Papers:
        {context}
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{query}")
            ]
        )

        # Remove the model_kwargs from the model
        self.model.model_kwargs = {}

        # Structured llm
        structured_llm = self.model.with_structured_output(CitedAnswers)

        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs_with_id(x["context"])))
            | prompt
            | structured_llm
        )

        retrieve_docs = (lambda x: x["query"]) | retriever

        chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
            answer=rag_chain_from_docs
        )

        result = chain.invoke({
            "query": query
        })

        # Return the result as JSON, nice and formatted
        return {
            "query": result["query"],
            "context": [doc.metadata for doc in result["context"]],
            "answers": [{
                "answer": cited_answer.answer,
                "citation": cited_answer.citation
            } for cited_answer in result["answer"].answers]
        }

    async def extract_keywords_and_sort(self, query):
        # Prompt template
        system_prompt = "You are a literature researcher in the computer science field. You have been given a research question or topic. Your goal is to give a list of keywords that would be necessary for narrowing down the search results. Each keyword should be relevant to the research question or topic to help in finding the most relevant papers. {format_instructions}\n\n{text}"

        # Construct the parser
        parser = JsonOutputParser(pydantic_object=Keywords)

        # And the prompt
        prompt = PromptTemplate(
            template=system_prompt,
            input_variables=["text"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        # Set kwargs
        self.model.model_kwargs={
            "response_format": {
                "type":"json_object"
            }
        }

        # Construct the chain
        chain = prompt | self.model | parser
        return chain.invoke({
            "text": query
        })