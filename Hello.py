# Resume App

# Imports
import streamlit as st 
import openai
from openai import OpenAI
import PyPDF2
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders.telegram import text_to_docs
from langchain.chains import RetrievalQA 
from langchain.prompts import PromptTemplate
from langchain. chat_models import ChatOpenAI
import os
import shutil

# Get the current working directory
current_directory = os.getcwd()

# Append '/db' to the current path
directory_path = os.path.join(current_directory, 'db')

if os.path.isdir(directory_path):
    # The directory exists and can be deleted
    shutil.rmtree(directory_path)
    print(f"Directory {directory_path} has been deleted.")
else:
    # The directory does not exist
    print(f"Directory {directory_path} does not exist.")

# Set API Key
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", 
                                   key="file_qa_api_key", 
                                   type="password", 
                                   value = "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"
                                  )
    
# Set the LLM Model
#model="gpt-3.5-turbo-0613"
model="gpt-4-1106-preview"
#model="ft:gpt-3.5-turbo-0613:personal::8Nvo6C0W"

# Define Text Extraction Function
@st.cache_data
def file_text(file):
    # Resume Text Extraction
    reader = PyPDF2.PdfReader(file)

    # Iterate over each page and extract text
    file_text = ""
    for page_num in range(len(reader.pages)):
        file_text += reader.pages[page_num].extract_text()
    
    return file_text

# Define cleaned text functon
def clean_text(text):
    # Implement your cleaning logic here
    cleaned_text = text.lower() # Example: simple lowercasing
    return cleaned_text

# Define ChromaDB Function
@st.cache_resource
def reading_resume(resume_text):
    # Set the OpenAI client
    client = OpenAI(api_key=openai_api_key)

    # Set Embeddings Model    
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)

    # Set the Request for Facts
    request = """
    Using the following resume text, please return a list of all facts and inferred facts about the user
    formatted as 'a+b+c+...' with each entry stated as an individual sentence stating a fact. 
    Collect as many facts as you can from the resume including details about time periods spent at each company.
    For experience facts, include context such as which position each experience fact is related to 
    and the relevant time when it was applicable, based on the position.
    
    In examining responsibilities, go beyond the surface-level descriptions.
    Uncover the intricacies and specific details of their duties.
    
    Craft the responses with an in-depth understanding, considering hidden skills and unique contributions.
    Break down and really understand each aspect of their duties to endorse the candidate.
    Keep the Technical skills or skills section as is. Do not  paraphrase it, 
    keep it as comma separated lit of skills.
    For each experience in each section (e.g., Education, Work Experience, Skills), 
    do not group experience in one section together, list the facts in a clear manner.
    
    Give more insights into the industry, soft skills and initiative the candidate showed. really market them back.

    Whenever relevant, for each fact include the section (EXPERIENCE, SKILLS) that the fact was found under in the format 'SECTION: fact'

    Collect no less than 200 fact statements.
    
    Do not include any other text or characters.

    """
    # Collect Messages
    messages = [
        {
            "role": "user",
            "content": request + resume_text
        }
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    fact_list = response.choices[0].message.content

    # Remove the brackets and split the string into a list
    facts = fact_list.split("+")

    # Clean the facts
    data = [clean_text(item) for item in facts]

    # Convert to Documents
    documents = text_to_docs(data)

    # Store embeddings in Chroma
    store = Chroma.from_documents(documents,
                                  embeddings,
                                  ids=[f"{index}" for index, _ in enumerate(documents)],
                                  collection_name="Resume-Embeddings",
                                  persist_directory='db'
                                 )
    #store.persist()
    return store
    
#Define the Template Function
@st.cache_resource
def resume_build():
    # Set the OpenAI client
    client = OpenAI(api_key=openai_api_key)
    
    template2 = """You are a bot that generates resumes for users based on your knowledge about the user, 
    using only the context provided and the job description provided.

    First determine if the user is qualified for the position in the job description by comparing known information
    about the user to the requirements in the job description. If the user is very clearly unqualified, 
    return the reason they are unqualified and do not generate a resume. 

    Based on the known information about the user and the provided job description, 
    please generate a resume by doing the following: 

    1. Create a resume using the relevant information known about the user.

    2. Edit the entries under skills, without making new skills, to better fit those listed in the job description.

    3. Edit the experience entries to better match the experiences from the job description. 

    4. Edit the entire resume to increase the chances of getting a job and to maximize similarity with the job description. 

    Do not return any of the contents of the job description itself.

    Do not make up any information not already known about the user.

    KNOWN INFORMATION ABOUT THE USER:
    {context}

    JOB DESCRIPTION: 
    {question}
    """

    PROMPT = PromptTemplate(
        template = template2, input_variables=["context", "question"]
    )

    # Define the LLM
    llm = ChatOpenAI(temperature=0.3, model=model, api_key=openai_api_key)
    #llm = ChatOpenAI(temperature=0.0, model="ft:gpt-3.5-turbo-0613:personal::8Nvo6C0W", api_key=openai_api_key)
    
    # Create the Resume Builder
    resume_builder = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=store.as_retriever(
            #search_type="mmr",  # Also test "similarity"
            search_kwargs={"k": 200,"score_threshold" : 0.0},
        ), 
        chain_type_kwargs={"prompt": PROMPT, },
    return_source_documents=True,
    )
    
    return resume_builder

def clear_store():

    try: 
        store.Resume-Embeddings.delete_collection()
    except NameError:
        exception = 1

# GENERATE PAGE VIEW
st.title("📝 Resume Generator") 
uploaded_file = st.file_uploader("Upload your resume (PDF).", 
                                 type=("pdf"), 
                                 on_change=clear_store
                                ) 
jd_text = st.text_area(
    "Job Description:",
    placeholder="Paste Job Description Text Here",
    disabled=not uploaded_file,
)

if st.button('Generate'):
    # Check for Errors and Missing Data
    if uploaded_file and jd_text and not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")

    # Execute if no errors found
    if uploaded_file and jd_text and openai_api_key:
        # Get text from file
        resume_text = file_text(uploaded_file)

        # Load embeddings into store
        store = reading_resume(resume_text)
        
        # Create Resume Builder
        placeholder = st.empty()
        placeholder.text("Generating New Resume... This may take a few minutes.")
        resume_builder = resume_build()

        # Generate Resume
        result = resume_builder(jd_text)
        resume_new = result['result']

        #Write Response
        placeholder.empty()
        st.write(resume_new)