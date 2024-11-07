from flask import Flask, render_template, request, session
import openai
import os
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import requests

app = Flask(__name__)
app.secret_key = 'supersecretkey'

######################################
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
import csv

# Global variables to store initialized components
llm = None
retrieval_chain = None
context = []

from langchain.llms import Ollama  # 使用 LangChain 的 Ollama 模型
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory


API_URL = 'https://tw-03.access.glows.ai:23926/api/chat/completions'
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6Ijk5MzJiY2FkLTU3MmYtNDAyYy05NzQ0LTYxNTg3YjdmNWQ5OCJ9.yAomfcv62s8RB8_qlWV8KWnqQg_Km57TMOwZYvWTKck"
MODEL = "llama3.1:70b"
FILE_ID = "32c3235d-2e91-45e8-80a4-939586b22cd2"

# Helper function to get a response from the custom LLM API
def get_answer(query):
    prompt = """
    以下內容分成兩個部分:規則和我的最終query，請根據我的query找出適當的規則，基於規則回答。
    以下是第一部分，規則：
    如果我的query是，生成一份歷史個案讓我練習判斷或相關內容，請用以下格式來產生新的模擬個案：
    # Clinical Case
    ## Patient Information:
    Age, Gender, Diagnosis, Medical History, Presenting Complaint, and more
    ## Important and necessary information
    ## Medical Condition:
    ## Current Situation:
    ## Ethical Considerations:
    ## Questions:
    舉例：
    # Clinical Case

    ## Patient Information:

    - **Age**: 60 years old
    - **Gender**: Male
    - **Diagnosis**: Terminal liver cancer
    - **Medical History**: Heavy alcohol use, non-compliance with medical advice
    - **Presenting Complaint**:
    Three weeks ago, the patient was admitted to the hospital with severe jaundice and abdominal pain. Imaging revealed extensive metastasis to the lungs and bones.

    ---

    ## Family Dynamics:

    - The patient is estranged from his family due to his past behavior and alcoholism.
    - His only close relative is a younger sister, who has taken on the role of caregiver.
    - The sister is very protective and insists on exploring every possible treatment option.

    ---

    ## Medical Condition:

    - The patient's condition has dramatically worsened over the past week.
    - He is now experiencing severe pain, nausea, and confusion.
    - Despite aggressive pain management, his quality of life is very poor.
    - The medical team has explained that additional treatments will likely only prolong suffering rather than improve the quality of life.

    ---

    ## Patient Wishes:

    - The patient had previously communicated to the medical team that he does not want to undergo extensive treatments that would prolong life without quality.
    - However, given his current confusion, he cannot reconfirm his wishes.

    ---

    ## Current Situation:

    - The sister is pressing for an experimental treatment that is not covered by insurance and has a very low probability of success.
    - The medical team has recommended transitioning to palliative care to focus on comfort and quality of life.

    ---

    ## Ethical Considerations:

    - **Beneficence vs. Non-maleficence**: Balancing the benefit of possible life extension against the risk of increased suffering.
    - **Autonomy**: Respecting the patient's previously expressed wishes.
    - **Justice**: Considering the fair allocation of resources, particularly when the likelihood of success is very low.
    - **Fidelity**: Maintaining trust with the patient and his sister by being honest about prognosis and treatment options.

    ---

    ## Questions:

    1. Given the patient's previous wishes and current state, what would be your next step in terms of communication with the sister and the medical team?
    2. How would you approach discussing the transition to palliative care with the sister who is insistent on pursuing all possible treatment options?
    3. Considering the patient's confusion, who should make the final decision about his care, and how should this decision be communicated to all involved?

    ---

    ## Next Steps:

    What are your decisions for the next steps regarding this case?。
    如果我的query是一串可能是行為描述(對應到這個個案)，那就幫我分析這樣子的護理行為是否正確。
    
    請用盡可能多的token詳細回答，都使用繁體中文。
    以下是我的query:
    """
    query = prompt + query
    headers = {
        'Authorization': f'Bearer {TOKEN}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': MODEL,
        'messages': [{'role': 'user', 'content': query}],
        'files': [{'type': 'file', 'id': FILE_ID}]
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']

messages = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

def ExplainMore():
    return get_answer("Please explain more in detail. Answer in English.")

def OriginalCase():
    return get_answer("Please tell me: What is the original case for the current discussion? Answer in English.")

def SimilarScenario():
    return get_answer("Please generate a similar scenario based on our current discussion for me to practice. Answer in English.")

def RelevantTheories():
    return get_answer("Please tell me: What data and theories form the basis of our current discussion, and provide the sources of this information. Answer in English.")

@app.route('/main', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        input_text = request.form['input_text']
        previous_input = session.get('previous_system_response', '')
        
        session['previous_system_response'] = input_text
        
        if 'Explain More' in input_text:
            result_text = ExplainMore()
        elif 'Original Case' in input_text:
            result_text = OriginalCase()
        elif 'Similar Scenario' in input_text:
            result_text = SimilarScenario()
        elif 'Relevant Theories' in input_text:
            result_text = RelevantTheories()
        else:
            # prompt_template = f"Answer in English. Based on my input, please determine my intent. If my input doesn't mention 'generate a new case' or similar terms, it means I am providing feedback on an existing case. You should compare my feedback with previous data handling methods, then generate suggestions and data sources. If I mention 'generate a new case,' then please generate a new case description, and after finishing the description, ask: Based on this case description, what will your next decision and action be?\n\n You don't need to explain your reasoning, just start outputting the relevant content. If it's Intent 1, this is the last system message (case description): {previous_input}. If it's Intent 2, please ignore the last system message and generate a new case description.\n\n This is my input:"
            prompt_template = ""
            result_text = get_answer(prompt_template + input_text)
        
        session['previous_system_response'] = result_text
        
        messages.append({'text': input_text, 'type': 'user-message'})
        messages.append({'text': result_text, 'type': 'bot-message'})
        return render_template('main.html', messages=messages)

    return render_template('main.html', messages=messages)

@app.route('/data')
def data():
    return render_template('data.html')

if __name__ == '__main__':
    app.run(debug=True)
