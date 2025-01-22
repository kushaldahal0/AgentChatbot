# AgentChatbot
Chatbot that can answer user queries from any documents and add a conversational form for collecting user information (Name, Phone Number, Email) when user ask chatbot to call them, You can use LangChain & Gemini/or any LLMs to complete the project.

Also, integrate conversational form (book appointment) with tool-agents. Integration of  conversational form with agent-tools, extract complete date format like (YYYY-MM-DD) from users query (eg. Next Monday, and integrate validation in user input with conversational form (like email, phone number) etc.


## Setup Instructions

### 1. Clone the Repository
If you haven't already, clone the project repository:
```bash
git clone <repository-url>
cd <repository-folder>
```
### 2. Create and Activate a Virtual Environment
It's recommended to use a virtual environment to manage dependencies:
```bash 
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

```

### 3. Install Requirements
Install the dependencies from the requirements.txt file:
```bash
pip install -r requirements.txt
```
### 5. Run the chatbot
Start the chatbot in terminal:

```bash
python chatbot.py
```
