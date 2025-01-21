import re
from datetime import datetime
import dateparser
from typing import List, Dict
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings

from huggingface_hub import login
from langchain_huggingface import HuggingFaceEndpoint

from dotenv import load_dotenv
import os
load_dotenv()

# Get the Hugging Face token from the .env file
huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

login(token=huggingface_token)
# Initialize free LLM (Zephyr-7B-beta)
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.5,
    huggingfacehub_api_token=huggingface_token,
)

# Initialize free local embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Document Processing
def load_and_process_documents(pdf_path: str):
    # Load PDF using PyPDF2
    reader = PdfReader(pdf_path)
    pages = []
    
    # Extract text from each page
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:  # Only add non-empty pages
            pages.append(Document(
                page_content=text,
                metadata={"source": pdf_path, "page": page_num + 1}
            ))
    
    # Split text into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separator="\n"
    )
    return text_splitter.split_documents(pages)

file_path = "documents\charniak.pdf"
# Initialize Vector Store
# Initialize Vector Store
try:
    docs = load_and_process_documents(file_path)
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()
except Exception as e:
    print(f"Error initializing vector store: {e}")
    exit()

# Validation Functions
def validate_email(email: str) -> bool:
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return re.match(pattern, email) is not None

def validate_phone(phone: str) -> bool:
    pattern = r"^\+?[1-9]\d{1,14}$"
    return re.match(pattern, phone) is not None

def parse_date(text: str) -> str:
    try:
        date = dateparser.parse(text, settings={'PREFER_DATES_FROM': 'future'})
        return date.strftime("%Y-%m-%d") if date else None
    except:
        return None


# Conversation State
class FormState:
    def __init__(self):
        self.active = False
        self.current_form = None
        self.collected_data = {}
        self.required_fields = []

form_state = FormState()

# Tools
tools = [
    Tool(
        name="DocumentQA",
        func=lambda q: RetrievalQA.from_chain_type(llm=llm, retriever=retriever).run(q),
        description="Answers questions from PDF documents"
    ),
    Tool(
        name="ScheduleCall",
        func=lambda _: "Please provide your name, phone, and email in this format: [Name] [Phone] [Email]",
        description="Initiates call scheduling process"
    ),
    Tool(
        name="BookAppointment",
        func=lambda q: f"Suggest date: {parse_date(q) or 'Date not recognized'}",
        description="Handles appointment booking with date parsing"
    )
]

# Define the prompt template with all required variables
prompt_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}"""

# Create the PromptTemplate
prompt = PromptTemplate.from_template(prompt_template)

# Create the ReAct agent
agent = create_react_agent(llm, tools, prompt)
# Create the AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Main Interaction Loop
def main():
    print("Welcome to the Free Document Chatbot!")
    print("You can:")
    print("1. Ask questions about the PDF document")
    print("2. Say 'schedule call' to book a meeting")
    print("3. Say 'book appointment' to schedule an appointment")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue

            # Handle form state
            if form_state.active:
                # Validate input based on current field
                current_field = form_state.required_fields[0]
                
                if current_field == "email" and not validate_email(user_input):
                    print("Agent: Invalid email format. Please try again.")
                    continue
                
                if current_field == "phone" and not validate_phone(user_input):
                    print("Agent: Invalid phone format. Please use international format (+XX...)")
                    continue
                
                form_state.collected_data[current_field] = user_input
                form_state.required_fields.pop(0)
                
                if not form_state.required_fields:
                    print("Agent: Thank you! We'll contact you soon.")
                    print(f"Collected info: {form_state.collected_data}")
                    form_state.active = False
                else:
                    print(f"Agent: Please provide your {form_state.required_fields[0]}:")
                continue

            # Handle special commands
            if "schedule call" in user_input.lower():
                form_state.active = True
                form_state.required_fields = ["name", "phone", "email"]
                form_state.collected_data = {}
                print("Agent: Please provide your name:")
                continue

            if "book appointment" in user_input.lower():
                date = parse_date(user_input)
                if date:
                    print(f"Agent: Suggested date: {date}. Confirm? (yes/no)")
                    if input("You: ").lower() == "yes":
                        print("Agent: Appointment booked!")
                    else:
                        print("Agent: Let's try another date.")
                else:
                    print("Agent: Please specify a date (e.g., 'next Monday' or 'March 25th')")
                continue

            # Handle document questions
            response = agent_executor.invoke({"input": user_input})
            print(f"Agent: {response['output']}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()