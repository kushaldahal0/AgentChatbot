import re
from datetime import datetime, timedelta
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
from langchain_huggingface import HuggingFaceEndpoint , HuggingFaceEmbeddings

# from langchain_community.embeddings import 

from huggingface_hub import login

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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
try:
    docs = load_and_process_documents(file_path)
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()
except Exception as e:
    print(f"Error initializing vector store: {e}")
    exit()

# Validation Functions
def validate_name(name: str) -> bool:
    """Validate name (at least 2 characters, no special characters)"""
    return bool(re.match(r"^[A-Za-z\s]{2,}$", name))

def validate_phone(phone: str) -> bool:
    """Validate international phone number format"""
    return bool(re.match(r"^\+?[1-9]\d{1,14}$", phone))

def validate_email(email: str) -> bool:
    """Validate email format"""
    return bool(re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", email))

def parse_date(text: str) -> str:
    """Enhanced date parsing with validation and fallback."""
    try:
        # Parse with context-aware settings
        date = dateparser.parse(
            text,
            settings={
                'PREFER_DATES_FROM': 'future',
                'RELATIVE_BASE': datetime.now(),
                'RETURN_AS_TIMEZONE_AWARE': True
            }
        )

        # If dateparser fails to parse, try to manually handle relative weekdays or explicit dates.
        if not date:
            # Manually handle explicit date format like "March 25"
            try:
                # Try parsing explicit dates like "March 25"
                date = datetime.strptime(text, "%B %d")
                
                # If the date has already passed this year, update to the next year
                if date < datetime.now():
                    date = date.replace(year=datetime.now().year + 1)
            
            except ValueError:
                pass  # Continue if not an explicit date

            # If still not parsed, handle relative weekdays like "next Monday"
            if not date:
                # Mapping of weekday names to their respective weekday number
                days_of_week = {
                    'monday': 0,
                    'tuesday': 1,
                    'wednesday': 2,
                    'thursday': 3,
                    'friday': 4,
                    'saturday': 5,
                    'sunday': 6
                }
                
                # Check for the weekday in the text and calculate the next occurrence
                for day_str, weekday in days_of_week.items():
                    if day_str in text.lower():
                        days_ahead = (weekday - datetime.now().weekday()) + 7
                        if days_ahead <= 0:  # If the specified day is today, move to the next one
                            days_ahead += 7
                        date = datetime.now() + timedelta(days=days_ahead)
                        break

        if not date:
            return None

        # Validate future date (minimum 24 hours ahead)
        if date < datetime.now() + timedelta(hours=24):
            return "date_too_soon"
        
        # Return the date in the desired format
        return date.strftime("%Y-%m-%d")
        
    except Exception as e:
        return f"error_{str(e)}"
        

def book_appointment_handler(query: str) -> str:
    """Enhanced appointment booking tool"""
    parsed = parse_date(query)
    
    if parsed is None:
        return "DATE_ERROR: Couldn't recognize the date. Please specify like:\n- 'Next Tuesday at 2PM'\n- 'March 25th'\n- 'Tomorrow morning'"
    
    if isinstance(parsed, str) and parsed.startswith("error"):
        return f"DATE_ERROR: Invalid date format - {parsed[6:]}"
    
    if parsed == "date_too_soon":
        return "DATE_ERROR: Please choose a date at least 24 hours in advance"
    
    return f"SUGGESTION: {parsed}\nCONFIRM: Does this work for you? (yes/no)"

def schedule_call_handler(query: str) -> str:
    """Handle call scheduling with step-by-step validation"""
    if not form_state.active:
        form_state.active = True
        form_state.required_fields = ["name", "phone", "email"]
        form_state.collected_data = {}
        return "Please provide your name:"
    
    current_field = form_state.required_fields[0]
    
    # Validate input based on the current field
    if current_field == "name" and not validate_name(query):
        return "Invalid name. Please provide a valid name (at least 2 characters, no special characters)."
    
    if current_field == "phone" and not validate_phone(query):
        return "Invalid phone number. Please provide a valid international phone number (e.g., +1234567890)."
    
    if current_field == "email" and not validate_email(query):
        return "Invalid email. Please provide a valid email address (e.g., example@domain.com)."
    
    # Store the validated input
    form_state.collected_data[current_field] = query
    form_state.required_fields.pop(0)
    
    if form_state.required_fields:
        return f"Please provide your {form_state.required_fields[0]}:"
    else:
        form_state.active = False
        return (
            "Thank you! We'll contact you soon.\n"
            f"Collected details:\n"
            f"- Name: {form_state.collected_data['name']}\n"
            f"- Phone: {form_state.collected_data['phone']}\n"
            f"- Email: {form_state.collected_data['email']}"
        )

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
        func=lambda q: RetrievalQA.from_chain_type(llm=llm, retriever=retriever).invoke(q),
        description="Answers questions from PDF documents"
    ),
    Tool(
        name="ScheduleCall",
        func=lambda _: "Please provide your name, phone, and email in this format: [Name] [Phone] [Email]",
        description="Initiates call scheduling process"
    ),
    Tool(
        name="BookAppointment",
        func=book_appointment_handler,
        description="Handles appointment booking. Input examples: 'next Monday', 'March 25th at 2PM', 'tomorrow morning'"
    ),
]

# Define the prompt template with all required variables
prompt_template = """You are an intelligent assistant designed to answer user questions based on the provided document. 
Follow these rules strictly and do not create any false data or out of topic questions on your own:

1. **Document-Based Answers**:
   - If the question is about the document, provide only relevant information from the document.
   - Do not make up answers or include external knowledge.

2. **Tool Usage**:
   - You have access to the following tools: {tools}.
   - Use the tools only when necessary to answer the question.
   - Always follow this format:
     - Thought: Think about what to do next.
     - Action: Choose the appropriate tool from [{tool_names}].
     - Action Input: Provide the input for the tool.
     - Observation: Record the result of the tool's action.

3. **Final Answer**:
   - After gathering all necessary information, provide a concise and accurate final answer.
   - Do not repeat the question or include unnecessary details.

4. **Efficiency**:
   - Use the minimum number of steps to answer the question.
   - Avoid unnecessary tool usage or repetitive actions.

5. **Format**:
   - Always follow this structure:
     - Question: The input question.
     - Thought: Your reasoning.
     - Action: The tool to use (if needed).
     - Action Input: The input for the tool (if needed).
     - Observation: The result of the tool (if used).
     - Final Answer: The final response to the question.

Begin!

Question: {input}
{agent_scratchpad}"""

# Create the PromptTemplate
prompt = PromptTemplate.from_template(prompt_template)

# Create the ReAct agent
agent = create_react_agent(llm, tools, prompt)
# Create the AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

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

            # Handle form submission
            if "schedule call" in user_input.lower():
                form_state.active = True
                form_state.required_fields = ["name", "phone", "email"]
                form_state.collected_data = {}
                print("Agent: Please provide your name:")
                continue


            #book appointment 
            if "book appointment" in user_input.lower():
                response = agent_executor.invoke({"input": user_input})
                output = response['output']
                
                if "SUGGESTION:" in output:
                    print(f"Agent: {output}")
                    confirmation = input("You: ").lower()
                    if confirmation == "yes":
                        print("Agent: Appointment booked successfully!")
                        # Store the appointment in form_state.collected_data
                    else:
                        print("Agent: Let's try another date. Please specify:")
                elif "DATE_ERROR:" in output:
                    print(f"Agent: {output[10:]}")
                else:
                    print(f"Agent: {output}")
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