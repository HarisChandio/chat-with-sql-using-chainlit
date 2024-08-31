import os
from typing import List

import chainlit as cl
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

# Update this with your PostgreSQL connection details
db_uri = "postgresql://postgres:haris70@localhost:5432/normazlization"

# Connect to the database without specifying tables
db = SQLDatabase.from_uri(db_uri)

# Get all table names
all_tables = db.get_usable_table_names()

db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

def retrieve_from_db(query: str) -> str:
    return db_chain.run(query)

system_message = """You are a AI assitant who has access to database containing the following tables: {tables}. You can help the user with queries about the database.
        """

human_qry_template = HumanMessagePromptTemplate.from_template(
    """User Query: {human_input}

Database Context:
{db_context}

Please provide a helpful response based on the above information:"""
)

@cl.on_chat_start
async def start():
    cl.user_session.set("llm", llm)
    cl.user_session.set("db_chain", db_chain)

    welcome_message = f"Welcome! I'm here to help you with queries about our database. The database contains the following tables: {', '.join(all_tables)}. What would you like to know?"
    await cl.Message(content=welcome_message).send()

@cl.on_message
async def main(message: str):
    llm = cl.user_session.get("llm")
    db_chain = cl.user_session.get("db_chain")

    # Retrieve context from the database
    db_context = retrieve_from_db(message)

    # Generate response
    messages = [
        SystemMessage(content=system_message.format(tables=', '.join(all_tables))),
        human_qry_template.format(human_input=message, db_context=db_context)
    ]
    response = llm(messages).content
    print(response, "response")

    await cl.Message(content=response).send()

if __name__ == "__main__":
    cl.run()