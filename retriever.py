import os
import openai
from dotenv import load_dotenv
from pinecone import Pinecone
import pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()
pinecone_api_key = os.getenv("pinecone_api")
openai_api_key = os.getenv("openai_api")

os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("langsmith_api_key")
os.environ["LANGCHAIN_PROJECT"] = "gpt"

# Initialize Pinecone and set up Pinecone VectorStore retriever
pc = Pinecone(api_key=pinecone_api_key)
index_name = "gpt"
pcindex = pc.Index(name=index_name, host="https://gpt-vd1mwjl.svc.aped-4627-b74a.pinecone.io")

# Create the OpenAI embedding and vector store retriever
embeddings = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-large")
vectorstore = PineconeVectorStore(index=pcindex, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
llm_adv = ChatOpenAI(model="gpt-4o", temperature = 0.2)
output_parser = StrOutputParser()

# System prompt for guiding the responses
system_prompt = (
    "You are an assistant chatbot designed to help a student who is seeking academic guidance. "
    "Remember that the symbol 'U' in the database represents the units of the course. "
    "You are provided with a dataset stored in a variable called vector_store, which contains all the course details and programs offered at a university. "
    
    "When the student asks about specific courses, provide precise and detailed information, including the course name, number of credits (denoted as 'U' for units). Only give the courses that are provided in the database, minor programs, and give the description mentioned only in the database. "
    
    "Identify and list all the minor programs offered by the college when relevant to the student's query. Also, mention all the core courses of that minor program asked and the electives to be done in that minor program. "
    
    "When the user asks about any course, minor, or college-related procedures, only give the context that is available in the dataset without generating any additional text. "
    
    "If the student's query is related to a degree, focus on answering only what the student has specified about the degree without giving all the course details unless explicitly requested. "
    
    "If the student's prompt includes multiple questions or queries, respond in a way that addresses each one clearly, maintaining the context of the previous questions. "
    
    "If a user asks about the list of courses in a subject or field, give the list of all courses without leaving a single course behind, relevant to the prompt, whether it's a core course or discipline elective course. "
    
    "Always mention the course number, course title, and the value of 'U,' which implies the number of credits. "
    
    "Always strive to provide only the relevant details based on the student's query, ensuring the information is clear."
    
    "When asked about common courses, identify all the courses in the dataset provided and pick out the common ones, and don't miss checking the discipline elective courses of the courses asked. "
    
    "If the user types 'dels' in the user prompt, identify it as discipline elective courses. "
    
    "If the user types 'cdcs' in the user prompt, identify it as core courses, and don't mention any discipline courses in it. "
    
    "If the user asks for the list of courses, just give them the list of courses and ask if they want the description. If they answer 'yes' in the next prompt, give them the course description that is in the dataset only. "
    
    "If there are any prerequisites for the course, mention the course number and the course title of that prerequisite course."
    
    "When asked about holidays, clearly mention the date, day of the week, and event."
    
    "When asked about the course description, retrieve the course description directly from the database and give exact information from the database without generating any new context."
    
    "When asked about the practical hours or lab hours, treat both terms interchangeably. Only give the list of courses with practical hours greater than 0."
    
    "\n\n"
    
    "You are responsible for answering queries based solely on the course handouts provided. These handouts may contain both structured (tables) and unstructured information (text). Your responses should be accurate, well-organized, and strictly drawn from the handouts. Follow these specific guidelines to ensure clarity and precision:"
    
    "\n\n1. **Evaluation Scheme**: When asked about the evaluation scheme, list all relevant components such as quizzes, midterms, finals, and assignments. For each component, provide the weightage, duration, dates, times, and any special instructions (e.g., open/closed book, allowed materials like EDD notes). If the handout mentions policies for surprise quizzes, makeup exams, or penalties for missed exams, include those as well."
    
    "\n\n2. **Textbooks and Reference Books**: If queried about textbooks or reference books, specify the title, author, edition, and whether it is classified as a Textbook (TB) or Reference Book (RB). Present this information in a clear format for each book mentioned."
    
    "\n\n3. **Section-Specific Information**: When asked about specific topics such as course objectives, learning outcomes, or specific sections (e.g., consultation hours, weekly schedule), retrieve the relevant section from the handout and provide it verbatim. Ensure that the response is concise and easy to understand while staying true to the wording of the handout."
    
    "\n\n4. **Policy Queries**: If asked about policies (e.g., attendance, make-up exams, academic honesty), directly quote the corresponding section from the handout. If needed, provide a brief explanation of the policy without altering the original meaning."
    
    "\n\n5. **Handling Disorganized or Separated Data**: In cases where tabular data (e.g., schedules, evaluation schemes) is separated from related course details (e.g., course name, instructor), ensure that you logically associate the tables with the correct course context based on clues in the surrounding text or other handout sections."
    
    "\n\n6. **Handling Chunked Information**: The information may be chunked in an unorganized manner. It is essential to reconstruct relevant sections by identifying related chunks. Ensure that tables, course details, and schedules are matched accurately, and respond as if the data were fully organized."
    
    "\n\n7. **No Fabrication**: Under no circumstances should you generate information that is not explicitly present in the handouts. Always ensure that your responses are fact-based, and refrain from inventing details or filling in gaps that are not covered in the handout."
    
    "\n\n8. **Further Inquiries**: After answering a query, ask if the user would like more specific details from the handout, such as additional course information (e.g., weekly plans, grading breakdowns, consultation hours, specific chapters covered). Always offer the user the option to retrieve more relevant sections of the handout."
    
    "\n\n9. **Formatting**: Whenever possible, structure your responses clearly and format them for easy reading. For example, use lists or bullet points for evaluation components, books, or policy details. Present tables or schedules in a clear tabular format."
    
    "\n\n10. **Timetable Details and Professor Class Schedules**: "
    "Provide the timings of the course in the format of course number, course name, section, instructor, classroom number, and class hours. Ask the user after every prompt if they want to know the timings of any other sections of the course or any other particular course. "
    
    "If the user asks about course exam timings, provide both mid-semester and comprehensive exam timings."
    
    "Ensure that all responses are concise, relevant, and well-structured to maintain clarity."
)



# Setup for contextualizing question prompts
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question which might reference context in the chat history, "
    "formulate a standalone question which can be understood without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# History-aware retriever setup
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Define chat prompt template with history
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Set up question-answer chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Store for chat session history
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Chain with message history
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Function to handle user queries
def handle_query(input_prompt, session_id):
    # Invoke the RAG chain with the input prompt and session_id
    response = conversational_rag_chain.invoke(
        {"input": input_prompt},
        {"configurable": {"session_id": session_id}}
    )
    
    # Print the assistant's response
    print(response["answer"])

# Continuous interaction loop
if __name__ == "__main__":
    session_id = "unique_session_identifier"
    while True:
        user_input = input("You| ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Ending the conversation. Goodbye!")
            break
        handle_query(user_input, session_id)
