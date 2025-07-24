# BITS Chatbot

This repository contains the source code for an advanced academic guidance chatbot for BITS Pilani. The chatbot is built using a Retrieval-Augmented Generation (RAG) architecture, leveraging OpenAI's language models and Pinecone's vector database to provide accurate and context-aware responses to student queries. It can answer questions about courses, minor programs, timetables, holidays, and other academic information by drawing from a variety of university documents like bulletins, course handouts, and timetables.

-----

## How It Works

The system operates in three main stages: Data Preprocessing, Vector Database Storage, and Retrieval & Generation.

### 1\. Data Preprocessing (`preprocessing.py`)

This is the foundational stage where information is extracted from various source documents and prepared for the AI model. The key challenge is that the source documents come in different formats (e.g., two-column PDFs, PDFs with mixed text and tables, JSON files). The script uses specialized techniques to handle each type:

  * **Two-Column PDF Bulletins**: For documents with a two-column layout (like `campus_facilities.pdf`), the script uses the `PyMuPDF` (`fitz`) library. The `extract_text_from_columns` function works by treating each page as two separate halves. It defines two rectangular areas—one for the left column and one for the right—and extracts the text from each column individually before combining them. This ensures the text is read in the correct order.

  * **PDFs with Text and Tables**: For standard PDFs that contain a mix of paragraphs and tables (e.g., `list_of_courses.pdf`, `SU_constitution.pdf`), the `extract_content_pdfplumber` function is used. It leverages the `pdfplumber` library, which is excellent at distinguishing between text and tabular data. The function iterates through each page, extracts all the text, and then specifically extracts tables. The table data is cleaned up to handle cells that might contain multi-line text and then converted into a string format, which is appended to the extracted page text.

  * **Course Handouts**: Course handouts are also processed using `pdfplumber` in the `extract_content_handouts` function. This function is similar to the one for general PDFs but is designed to treat each handout as a complete document. It includes an option to `bypass_chunking`, meaning the entire content of a handout is kept together as a single piece of information, preserving its context.

  * **JSON Timetable Data**: The script also processes structured data from `timetable.json`. The `process_course_data` function reads the JSON file, which contains nested information about courses, sections, instructors, schedules, and exam dates. It then formats this structured data into human-readable text chunks. Each chunk contains all the relevant details for a single course, making it easy for the language model to understand.

  * **Chunking and Labeling**: After extracting the raw text, the `create_documents_with_labels` function orchestrates the entire process. It uses LangChain's `RecursiveCharacterTextSplitter` to break down the long extracted texts into smaller, overlapping chunks (of about 1000 characters). Each chunk is then assigned a unique, descriptive label (e.g., `bulletin_0_chunk_1`, `handout_3`, `course_5_chunk_0`). This labeling is critical for identifying the source of information during the retrieval phase.

### 2\. Vector Database Storage (`vectordb.py`)

Once the data is preprocessed and chunked, it needs to be stored in a way that allows for efficient searching.

  * **Embedding**: The `create_embeddings` function takes the labeled text chunks and uses OpenAI's powerful `text-embedding-3-large` model to convert each chunk into a high-dimensional vector (an embedding). These vectors are numerical representations of the text's meaning.
  * **Storage**: The `store_embeddings` function then takes these embeddings and stores them in a **Pinecone vector database**. Each vector is stored along with its corresponding text content and the unique label generated during preprocessing. The data is "upserted" in batches to ensure efficient processing.

### 3\. Retrieval and Generation (`retriever.py`)

This is the final stage where the chatbot interacts with the user.

  * **RAG Pipeline**: The system uses a Retrieval-Augmented Generation (RAG) pipeline. When a user asks a question, the user's query is first converted into a vector embedding using the same OpenAI model.
  * **Retrieval**: The retriever then searches the **Pinecone vector database** to find the vectors (and their corresponding text chunks) that are most semantically similar to the user's query vector. It is configured to retrieve the top 8 most relevant chunks (`"k": 8`).
  * **Generation**: These relevant chunks of text are then passed as context to an advanced language model (**GPT-4o** or **GPT-4o-mini**). The model is given the user's question along with the retrieved context and a detailed **system prompt**.
  * **System Prompt**: The `system_prompt` is a crucial component that guides the LLM's behavior. It instructs the model to act as a helpful academic assistant, to provide precise details, to correctly interpret abbreviations (like 'U' for units, 'dels' for discipline electives), and how to format its responses based on the query type (e.g., listing courses, detailing evaluation schemes, providing holiday information). It also contains strict instructions to **never invent information** and to base all answers strictly on the provided context from the handouts.
  * **Conversational History**: The chatbot is designed to handle follow-up questions. It uses LangChain's `create_history_aware_retriever` to reformulate a user's latest question by taking the chat history into account. This allows for a more natural and continuous conversation.

-----

## Setup and Usage

To run this project, follow these steps:

1.  **Clone the Repository**

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install Dependencies**
    Make sure you have Python installed, then install the required packages from `requirements.txt`.

    ```bash
    pip install -r requirements.txt
    ```

3.  **Set Up Environment Variables**
    Create a `.env` file in the root directory and add your API keys:

    ```
    pinecone_api="YOUR_PINECONE_API_KEY"
    openai_api="YOUR_OPENAI_API_KEY"
    langsmith_api_key="YOUR_LANGSMITH_API_KEY" # Optional, for tracing
    ```

4.  **Preprocess Data and Populate VectorDB**
    Run the `vectordb.py` script to extract data from the documents in the `dataset` folder and store the embeddings in your Pinecone index.

    ```bash
    python vectordb.py
    ```

5.  **Run the Chatbot**
    Start the interactive chatbot by running the `retriever.py` script.

    ```bash
    python retriever.py
    ```

    You can then start asking questions in the terminal. Type `exit`, `quit`, or `q` to end the conversation.

-----

## File Descriptions

  * `preprocessing.py`: Contains all the logic for extracting and cleaning text and data from PDF and JSON files.
  * `vectordb.py`: Handles the creation of text embeddings and their storage in the Pinecone vector database.
  * `retriever.py`: Sets up the RAG pipeline, defines the LLM's behavior via the system prompt, and manages the conversational flow.
  * `requirements.txt`: A list of all the Python libraries needed to run the project.
  * `dataset/`: This folder should contain all the source documents (PDFs, JSON, etc.) that the chatbot will use to answer questions.
