import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from typing import List

# ------------------ ANIMATION FUNCTIONS ------------------
def animated_text(text, effect="typewriter", delay=0.1, color=None):
    """Custom animated text function using HTML/CSS"""
    if effect == "typewriter":
        # Simple typewriter effect using JavaScript
        html = f"""
        <div id="typewriter" style="color: {color or 'inherit'}; font-size: 24px; font-weight: bold;"></div>
        <script>
        function typeWriter(text, i, id) {{
            if (i < text.length) {{
                document.getElementById(id).innerHTML += text.charAt(i);
                i++;
                setTimeout(() => typeWriter(text, i, id), {delay * 1000});
            }}
        }}
        typeWriter("{text}", 0, "typewriter");
        </script>
        """
        st.markdown(html, unsafe_allow_html=True)
    elif effect == "fade_in":
        # Fade-in effect using CSS animation
        html = f"""
        <style>
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        .fade-in {{
            animation: fadeIn 1s ease-in;
            font-size: 20px;
            font-weight: bold;
            color: {color or 'inherit'};
        }}
        </style>
        <div class="fade-in">{text}</div>
        """
        st.markdown(html, unsafe_allow_html=True)
    elif effect == "pulse":
        # Pulse effect using CSS animation
        html = f"""
        <style>
        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
            100% {{ transform: scale(1); }}
        }}
        .pulse {{
            animation: pulse 2s infinite;
            font-size: 20px;
            font-weight: bold;
            color: {color or 'inherit'};
        }}
        </style>
        <div class="pulse">{text}</div>
        """
        st.markdown(html, unsafe_allow_html=True)
    elif effect == "bounce":
        # Bounce effect using CSS animation
        html = f"""
        <style>
        @keyframes bounce {{
            0%, 20%, 50%, 80%, 100% {{ transform: translateY(0); }}
            40% {{ transform: translateY(-20px); }}
            60% {{ transform: translateY(-10px); }}
        }}
        .bounce {{
            animation: bounce 1s infinite;
            font-size: 18px;
            font-weight: bold;
            color: {color or 'inherit'};
        }}
        </style>
        <div class="bounce">{text}</div>
        """
        st.markdown(html, unsafe_allow_html=True)
    elif effect == "tada":
        # Celebration effect using CSS animation
        html = f"""
        <style>
        @keyframes tada {{
            0% {{ transform: scale(1); }}
            10%, 20% {{ transform: scale(0.9) rotate(-3deg); }}
            30%, 50%, 70%, 90% {{ transform: scale(1.1) rotate(3deg); }}
            40%, 60%, 80% {{ transform: scale(1.1) rotate(-3deg); }}
            100% {{ transform: scale(1) rotate(0); }}
        }}
        .tada {{
            animation: tada 1s infinite;
            font-size: 20px;
            font-weight: bold;
            color: {color or 'inherit'};
        }}
        </style>
        <div class="tada">{text}</div>
        """
        st.markdown(html, unsafe_allow_html=True)
    else:
        # Fallback to regular text
        st.markdown(f"**{text}**")

# ------------------ CONFIG ------------------
load_dotenv()  # Load from .env file
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB_DIR = "chroma_db"

# Initialize session state for config parameters
if 'chunk_size' not in st.session_state:
    st.session_state.chunk_size = 1200
if 'chunk_overlap' not in st.session_state:
    st.session_state.chunk_overlap = 250
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.3
if 'max_tokens' not in st.session_state:
    st.session_state.max_tokens = 1024
if 'top_k' not in st.session_state:
    st.session_state.top_k = 4
if 'theme' not in st.session_state:
    st.session_state.theme = "Light"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'use_next_chunk' not in st.session_state:
    st.session_state.use_next_chunk = True
if 'llm_model' not in st.session_state:
    st.session_state.llm_model = "llama3-8b-8192"
if 'selected_sources' not in st.session_state:
    st.session_state.selected_sources = []
if 'quiz_questions' not in st.session_state:
    st.session_state.quiz_questions = []
if 'quiz_answers' not in st.session_state:
    st.session_state.quiz_answers = {}
if 'quiz_score' not in st.session_state:
    st.session_state.quiz_score = 0
if 'quiz_completed' not in st.session_state:
    st.session_state.quiz_completed = False
if 'custom_color' not in st.session_state:
    st.session_state.custom_color = "#4682b4"  # Default steel blue
if 'study_date' not in st.session_state:
    st.session_state.study_date = datetime.now().date()
if 'study_time' not in st.session_state:
    st.session_state.study_time = datetime.now().time()
if 'quiz_history' not in st.session_state:
    st.session_state.quiz_history = []
if 'user_rating' not in st.session_state:
    st.session_state.user_rating = 0

# Initialize session state for document statistics
if 'num_documents' not in st.session_state:
    st.session_state.num_documents = 0
if 'num_chunks' not in st.session_state:
    st.session_state.num_chunks = 0
if 'avg_chunk_size' not in st.session_state:
    st.session_state.avg_chunk_size = 0


# ------------------ LOAD DOCUMENTS ------------------
def load_pdfs(uploaded_files, progress_callback=None):
    docs = []
    os.makedirs("temp", exist_ok=True)
    total_files = len(uploaded_files)
    
    for i, file in enumerate(uploaded_files):
        temp_path = os.path.join("temp", file.name)
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())
        loader = PyPDFLoader(temp_path)
        docs.extend(loader.load())
        
        # Update progress if callback provided
        if progress_callback:
            progress_callback(int((i + 1) / total_files * 100))
            
    return docs

# ------------------ PREVIEW DOCUMENTS ------------------
def preview_pdf(file):
    """Extract first few pages of PDF for preview"""
    try:
        # Reset file pointer to beginning
        file.seek(0)
        pdf_reader = PdfReader(file)
        text = ""
        # Extract first 3 pages or fewer
        pages_to_extract = min(3, len(pdf_reader.pages))
        for i in range(pages_to_extract):
            page = pdf_reader.pages[i]
            text += page.extract_text() + "\n\n"
        return text[:1000] + "..." if len(text) > 1000 else text
    except Exception as e:
        return f"Error previewing document: {str(e)}"

# ------------------ GET DOCUMENT SOURCES ------------------
def get_document_sources(vectorstore):
    """Get list of document sources from vectorstore"""
    try:
        all_docs = vectorstore._collection.get(include=["metadatas"])
        sources = set()
        for meta in all_docs["metadatas"]:
            if "source" in meta:
                sources.add(meta["source"])
        return list(sources)
    except:
        return []

# ------------------ PARSE QUIZ CONTENT ------------------
def parse_quiz_content(content):
    """Parse quiz content into structured format"""
    import re
    import random
    
    # Split content into questions
    questions = re.split(r'\n\d+\.', content)
    if len(questions) > 0 and not questions[0].strip():
        questions = questions[1:]  # Remove first empty element
    
    parsed_questions = []
    for i, q in enumerate(questions):
        q = q.strip()
        if not q:
            continue
            
        # Extract question text and options
        lines = q.split('\n')
        question_text = ""
        options = []
        correct_answer = None
        
        # Find where options start
        option_start_idx = -1
        for idx, line in enumerate(lines):
            line = line.strip()
            if re.match(r'^[A-Da-d][\).]', line):
                option_start_idx = idx
                break
        
        # Extract question text (everything before first option)
        if option_start_idx > 0:
            question_text = " ".join(lines[:option_start_idx]).strip()
        else:
            # Fallback: use first line as question
            question_text = lines[0].strip() if lines else ""
            option_start_idx = 1
        
        # Extract options
        for line in lines[option_start_idx:]:
            line = line.strip()
            if re.match(r'^[A-Da-d][\).]', line):
                options.append(line)
                # Check if this is the correct answer (assuming it's marked)
                if '*' in line or '(correct)' in line.lower():
                    # Extract the letter from the beginning of the line
                    match = re.match(r'^([A-Da-d])[\).]', line)
                    if match:
                        correct_answer = match.group(1).upper()
                # Limit to 5 options max
                if len(options) >= 5:
                    break
                    
        # If no explicit correct answer, try to find "Answer:" line
        if not correct_answer:
            for line in lines:
                if line.lower().startswith('answer:'):
                    match = re.search(r'[A-Da-d]', line)
                    if match:
                        correct_answer = match.group().upper()
                        break
                        
        # If still no correct answer, randomly select one of the options as correct
        # This prevents "None" from being displayed as the correct answer
        if not correct_answer and options:
            # Extract option letters from the options
            option_letters = []
            for option in options:
                match = re.match(r'^([A-Da-d])[\).]', option)
                if match:
                    option_letters.append(match.group(1).upper())
            # Randomly select one as correct if we have option letters
            if option_letters:
                correct_answer = random.choice(option_letters)
                         
        # Only add questions with 3-5 options and non-empty question text
        if len(options) >= 3 and len(options) <= 5 and question_text:
            # Limit options to 4 if more than 4
            if len(options) > 4:
                options = options[:4]
            parsed_questions.append({
                'id': len(parsed_questions),  # Use parsed_questions length for sequential IDs
                'question': question_text,
                'options': options,
                'correct_answer': correct_answer
            })
    
    # Limit to exactly 5 questions
    return parsed_questions[:5]

# ------------------ DISPLAY QUIZ ------------------
def display_quiz(questions):
    """Display quiz with radio options and immediate feedback"""
    import re
    
    if not questions:
        st.warning("No quiz questions available.")
        return
    
    # Progress bar
    total_questions = len(questions)
    answered_questions = len([q for q in questions if f"quiz_answer_{q['id']}" in st.session_state.quiz_answers])
    progress = answered_questions / total_questions if total_questions > 0 else 0
    st.progress(progress)
    st.write(f"Progress: {answered_questions}/{total_questions} questions answered")
    
    # Display each question
    for question in questions:
        q_id = question['id']
        st.markdown(f"**Question {q_id + 1}:** {question['question']}")
        
        # Extract just the letters from options for radio buttons
        option_labels = []
        option_texts = []
        for option in question['options']:
            match = re.match(r'^([A-Da-d])[\).]\s*(.*)', option)
            if match:
                option_labels.append(match.group(1))
                option_texts.append(match.group(2))
            else:
                option_labels.append(option)
                option_texts.append(option)
        
        # Check if question has been answered
        user_answer_key = f"quiz_answer_{q_id}"
        if user_answer_key in st.session_state.quiz_answers:
            user_answer = st.session_state.quiz_answers[user_answer_key]
            correct_answer = question['correct_answer']
            
            # Display radio buttons (disabled since already answered)
            user_answer_idx = option_labels.index(user_answer) if user_answer in option_labels else 0
            st.radio(
                "Select your answer:",
                option_texts,
                index=user_answer_idx,
                key=f"quiz_radio_{q_id}",
                disabled=True
            )
            
            # Show feedback
            # Convert both answers to uppercase for comparison
            if user_answer and correct_answer and user_answer.upper() == correct_answer.upper():
                animated_text("Correct! 🎉", effect="tada", delay=0.1)
            else:
                # Only show the correct answer if it exists
                if correct_answer:
                    st.error(f"Incorrect. The correct answer is: {correct_answer}")
                    # Show correct answer text
                    if correct_answer in option_labels:
                        correct_idx = option_labels.index(correct_answer)
                        st.info(f"Explanation: {option_texts[correct_idx]}")
                else:
                    st.error("Incorrect.")
        else:
            # Display radio buttons for unanswered questions
            user_answer = st.radio(
                "Select your answer:",
                option_texts,
                key=f"quiz_radio_{q_id}"
            )
            
            # Submit button for each question
            if st.button("Submit", key=f"submit_{q_id}"):
                if user_answer:
                    # Find the corresponding letter for the selected option
                    selected_idx = option_texts.index(user_answer)
                    selected_letter = option_labels[selected_idx]
                    
                    # Store answer
                    st.session_state.quiz_answers[user_answer_key] = selected_letter
                    
                    # Check if all questions are answered
                    if len(st.session_state.quiz_answers) == total_questions:
                        # Calculate score
                        score = 0
                        for q in questions:
                            answer_key = f"quiz_answer_{q['id']}"
                            if answer_key in st.session_state.quiz_answers:
                                user_ans = st.session_state.quiz_answers[answer_key]
                                correct_ans = q['correct_answer']
                                # Convert both answers to uppercase for comparison
                                if user_ans and correct_ans and user_ans.upper() == correct_ans.upper():
                                    score += 1
                        st.session_state.quiz_score = score
                        st.session_state.quiz_completed = True
                    
                    st.rerun()
                else:
                    st.warning("Please select an answer before submitting.")
        
        st.markdown("---")
    
    # Show final score if quiz is completed
    if st.session_state.quiz_completed:
        st.markdown("---")
        st.subheader("Quiz Results")
        st.metric("Your Score", f"{st.session_state.quiz_score}/{total_questions}")
        percentage = (st.session_state.quiz_score / total_questions) * 100 if total_questions > 0 else 0
        st.write(f"Percentage: {percentage:.1f}%")
        
        if percentage >= 80:
            animated_text("Excellent work! 🎉", effect="tada", delay=0.1)
        elif percentage >= 60:
            animated_text("Good job! Keep it up! 👍", effect="bounce", delay=0.1)
        else:
            animated_text("Keep studying, you'll get there! 💪", effect="pulse", delay=0.1)
        
        # Store quiz result in history
        st.session_state.quiz_history.append({
            'date': st.session_state.study_date.strftime("%Y-%m-%d"),
            'time': st.session_state.study_time.strftime("%H:%M"),
            'score': st.session_state.quiz_score,
            'total': total_questions,
            'percentage': percentage
        })
        
        # Reset quiz button - now generates new questions
        if st.button("Retake Quiz"):
            # Generate new quiz questions
            if 'vectorstore' in st.session_state and st.session_state.vectorstore:
                animated_text("Creating new quiz questions...", effect="pulse", delay=0.1)
                with st.spinner():
                    # Retrieve all documents for quiz generation
                    all_docs = st.session_state.vectorstore.get()
                    doc_texts = all_docs["documents"]
                    # Limit to first 5 chunks for performance
                    doc_texts = doc_texts[:5]
                    # Combine documents for quiz generation (limiting size for performance)
                    combined_text = "\n\n".join(doc_texts)
                    # Truncate if too long
                    if len(combined_text) > 4000:
                        combined_text = combined_text[:4000]
                    
                    quiz_prompt = f"Based on the following educational content, generate 5 multiple-choice quiz questions with 4 options each and indicate the correct answer:\n\n{combined_text}"
                    llm = ChatGroq(model=st.session_state.llm_model, temperature=st.session_state.temperature, max_tokens=st.session_state.max_tokens)
                    quiz_content = llm.invoke(quiz_prompt)
                    
                    # Parse and store quiz questions
                    st.session_state.quiz_questions = parse_quiz_content(quiz_content.content)
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_score = 0
                    st.session_state.quiz_completed = False
                    st.rerun()
            else:
                # Fallback if vectorstore is not available
                st.session_state.quiz_answers = {}
                st.session_state.quiz_score = 0
                st.session_state.quiz_completed = False
                st.rerun()
        
        # Download results button
        if st.button("Download Quiz Results"):
            import csv
            import io
            
            # Create CSV content
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(["Question", "Your Answer", "Correct Answer", "Result"])
            
            # Write data
            for question in st.session_state.quiz_questions:
                q_id = question['id']
                user_answer_key = f"quiz_answer_{q_id}"
                user_answer = st.session_state.quiz_answers.get(user_answer_key, "Not answered")
                correct_answer = question['correct_answer']
                # Convert both answers to uppercase for comparison
                result = "Correct" if user_answer and correct_answer and user_answer.upper() == correct_answer.upper() else "Incorrect"
                
                writer.writerow([
                    question['question'],
                    user_answer,
                    correct_answer if correct_answer else "Not provided",
                    result
                ])
            
            # Write summary
            writer.writerow([])
            writer.writerow(["Summary"])
            writer.writerow(["Total Questions", len(st.session_state.quiz_questions)])
            writer.writerow(["Correct Answers", st.session_state.quiz_score])
            writer.writerow(["Score", f"{st.session_state.quiz_score}/{len(st.session_state.quiz_questions)}"])
            writer.writerow(["Percentage", f"{(st.session_state.quiz_score / len(st.session_state.quiz_questions) * 100):.1f}%"])
            
            # Create download button
            st.download_button(
                label="Download CSV",
                data=output.getvalue(),
                file_name="quiz_results.csv",
                mime="text/csv"
            )
        
        st.markdown("---")
        

# ------------------ SPLIT DOCUMENTS ------------------
def split_docs(documents, chunk_size=None, chunk_overlap=None, progress_callback=None):
    # Use session state values if not provided
    if chunk_size is None:
        chunk_size = st.session_state.chunk_size
    if chunk_overlap is None:
        chunk_overlap = st.session_state.chunk_overlap
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # If progress callback is provided, split documents one by one
    if progress_callback and documents:
        splits = []
        total_docs = len(documents)
        for i, doc in enumerate(documents):
            splits.extend(splitter.split_documents([doc]))
            # Update progress
            if progress_callback:
                progress_callback(int((i + 1) / total_docs * 100))
        return splits
    else:
        return splitter.split_documents(documents)

# ------------------ CUSTOM RETRIEVER WITH NEXT CHUNK ------------------
def custom_retrieve_with_next_chunk(vectorstore, query: str, k: int = 4) -> List:
    """Retrieve top-k chunks and also fetch the next chunk after each match."""
    # Filter by selected sources if any are selected
    if st.session_state.selected_sources:
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": k,
                "filter": {"source": {"$in": st.session_state.selected_sources}}
            }
        )
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        
    results = retriever.get_relevant_documents(query)
    
    # If next chunk feature is disabled, return results as is
    if not st.session_state.use_next_chunk:
        return results
    
    all_docs = vectorstore._collection.get(include=["metadatas", "documents"])
    doc_texts = all_docs["documents"]
    doc_metas = all_docs["metadatas"]
    
    extended_results = []
    seen_indexes = set()
    
    for doc in results:
        # Filter by selected sources
        if st.session_state.selected_sources and doc.metadata.get("source") not in st.session_state.selected_sources:
            continue
            
        if doc.page_content not in doc_texts:
            continue
        idx = doc_texts.index(doc.page_content)
        # Add current chunk
        if idx not in seen_indexes:
            extended_results.append(doc)
            seen_indexes.add(idx)
        # Add next chunk if exists
        if idx + 1 < len(doc_texts) and (idx + 1) not in seen_indexes:
            from langchain.schema import Document
            next_doc = Document(page_content=doc_texts[idx + 1], metadata=doc_metas[idx + 1])
            extended_results.append(next_doc)
            seen_indexes.add(idx + 1)
    
    return extended_results

# ------------------ CREATE QA CHAIN ------------------
def create_qa_chain(vectorstore):
    llm = ChatGroq(
        model=st.session_state.llm_model,
        temperature=st.session_state.temperature,
        max_tokens=st.session_state.max_tokens
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": st.session_state.top_k})
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

# ------------------ STREAMLIT UI ------------------
st.set_page_config(page_title="Groq RAG Chatbot (Chroma) - Multi Page Retrieval", layout="wide")

# Theme styling
if st.session_state.theme == "Dark":
    st.markdown(f"""
    <style>
    .stApp {{
        background-color: #0e1117;
        color: #fafafa;
    }}
    .stMarkdown, .stText {{
        color: #fafafa;
    }}
    .stSidebar {{
        background-color: #1e2130;
    }}
    .stButton>button {{
        background-color: #2e313e;
        color: #fafafa;
    }}
    </style>
    """, unsafe_allow_html=True)
elif st.session_state.theme == "Blue":
    st.markdown(f"""
    <style>
    .stApp {{
        background-color: #f0f8ff;
        color: #00008b;
    }}
    .stSidebar {{
        background-color: #e6f3ff;
    }}
    .stButton>button {{
        background-color: {st.session_state.custom_color};
        color: white;
    }}
    </style>
    """, unsafe_allow_html=True)
elif st.session_state.theme == "Green":
    st.markdown(f"""
    <style>
    .stApp {{
        background-color: #f5fff5;
        color: #006400;
    }}
    .stSidebar {{
        background-color: #e6ffe6;
    }}
    .stButton>button {{
        background-color: #2e8b57;
        color: white;
    }}
    </style>
    """, unsafe_allow_html=True)
else:  # Custom theme using the selected color
    st.markdown(f"""
    <style>
    .stApp {{
        background-color: #ffffff;
        color: #000000;
    }}
    .stSidebar {{
        background-color: #f0f0f0;
    }}
    .stButton>button {{
        background-color: {st.session_state.custom_color};
        color: white;
    }}
    </style>
    """, unsafe_allow_html=True)

# Centered content with CSS
st.markdown(
    """
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;">
    """,
    unsafe_allow_html=True
)

st.title("📄 STUDYMATE-AI LEARNING ASSISTANT")
st.markdown("## Learning Assistant")
st.write("Ask me anything about your studies")

# Centered illustration - use placeholder image or static asset URL
st.image("technology-7994887_1280.png", width=500)

# Welcome message with animation
animated_text(
    "Welcome to StudyMate AI",
    effect="typewriter",
    delay=0.1,
    color=st.session_state.custom_color
)

st.markdown(
    """
    Your intelligent learning companion powered by advanced AI. Upload your study materials, ask questions, and get personalized explanations tailored to your learning style.
    """
)

st.markdown(
    """
    </div>
    """,
    unsafe_allow_html=True
)


if os.path.exists(CHROMA_DB_DIR):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    qa_chain = create_qa_chain(vectorstore)
    # Store vectorstore in session state for quiz regeneration
    st.session_state.vectorstore = vectorstore
else:
    vectorstore = None
    qa_chain = None
    st.session_state.vectorstore = None

animated_text("StudyMate AI", effect="pulse", delay=0.1, color=st.session_state.custom_color)
st.sidebar.write("Your intelligent learning companion")

# Configuration options
with st.sidebar.expander("⚙️ Configuration"):
    st.session_state.chunk_size = st.slider("Chunk Size", 500, 2000, st.session_state.chunk_size)
    st.session_state.chunk_overlap = st.slider("Chunk Overlap", 0, 500, st.session_state.chunk_overlap)
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, 0.1)
    st.session_state.max_tokens = st.slider("Max Tokens", 512, 2048, st.session_state.max_tokens)
    st.session_state.top_k = st.slider("Top K Documents", 1, 10, st.session_state.top_k)
    st.session_state.theme = st.selectbox("Theme", ["Light", "Dark", "Blue", "Green", "Custom"], index=["Light", "Dark", "Blue", "Green", "Custom"].index(st.session_state.theme) if st.session_state.theme in ["Light", "Dark", "Blue", "Green", "Custom"] else 0)
    st.session_state.use_next_chunk = st.checkbox("Use Next Chunk Feature", value=st.session_state.use_next_chunk)
    st.session_state.llm_model = st.radio("LLM Model", ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"], index=["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"].index(st.session_state.llm_model))
    st.session_state.chunk_size = st.number_input("Custom Chunk Size", min_value=500, max_value=2000, value=st.session_state.chunk_size)
    st.session_state.custom_color = st.color_picker("Custom Theme Color", st.session_state.custom_color)

with st.sidebar:
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    # Study tracking
    st.markdown("---")
    st.subheader("📅 Study Tracking")
    st.session_state.study_date = st.date_input("Study Date", st.session_state.study_date)
    st.session_state.study_time = st.time_input("Study Time", st.session_state.study_time)
    
    # Progress tracking
    if st.session_state.quiz_history:
        st.markdown("---")
        st.subheader("📈 Progress Tracking")
        import pandas as pd
        import altair as alt
        
        # Create DataFrame from quiz history
        df = pd.DataFrame(st.session_state.quiz_history)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Create chart
        chart = alt.Chart(df).mark_line(point=True).encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('percentage:Q', title='Score (%)'),
            tooltip=['date:T', 'percentage:Q']
        ).properties(
            title='Quiz Performance Over Time'
        )
        
        st.altair_chart(chart, use_container_width=True)
        
        # Show recent history
        st.write("Recent Results:")
        for result in st.session_state.quiz_history[-5:]:  # Show last 5
            st.write(f"{result['date']}: {result['score']}/{result['total']} ({result['percentage']:.1f}%)")
    
    # Document sources multiselect
    if vectorstore:
        sources = get_document_sources(vectorstore)
        if sources:
            st.markdown("---")
            st.subheader("Filter Sources")
            st.session_state.selected_sources = st.multiselect(
                "Select document sources to include in search:",
                sources,
                default=st.session_state.selected_sources if st.session_state.selected_sources else sources
            )
    
    # Document preview
    if uploaded_files:
        st.markdown("---")
        st.subheader("Document Preview")
        for file in uploaded_files:
            with st.expander(f"Preview: {file.name}"):
                preview_text = preview_pdf(file)
                st.text_area("", value=preview_text, height=200, key=f"preview_{file.name}")
    
    # Chat history
    if st.session_state.chat_history:
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        col1.subheader("Chat History")
        if col2.button("Clear"):
            st.session_state.chat_history = []
            st.rerun()
            
        for chat in reversed(st.session_state.chat_history[-5:]):  # Show last 5 conversations
            with st.expander(f"Q: {chat['question'][:30]}{'...' if len(chat['question']) > 30 else ''}"):
                st.write(f"**Question:** {chat['question']}")
                st.write(f"**Answer:** {chat['answer']}")
                st.caption(f"Asked at: {chat['timestamp']}")
    
    # User feedback
    st.markdown("---")
    st.subheader("Feedback")
    st.session_state.user_rating = st.slider("Rate your experience (1-5)", 1, 5, st.session_state.user_rating)
    feedback_text = st.text_area("Additional feedback:")
    if st.button("Submit Feedback"):
        animated_text("Thank you for your feedback! 🙏", effect="tada", delay=0.1)

if uploaded_files:
    # Show progress for document processing with animation
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Load PDFs with progress callback
    animated_text("Loading PDFs...", effect="bounce", delay=0.1)
    def update_loading_progress(value):
        progress_bar.progress(int(value * 0.33))  # 0-33% for loading
    docs = load_pdfs(uploaded_files, update_loading_progress)
    
    # Split documents with progress callback
    animated_text("Splitting documents...", effect="bounce", delay=0.1)
    def update_splitting_progress(value):
        progress_bar.progress(33 + int(value * 0.33))  # 33-66% for splitting
    splits = split_docs(docs, progress_callback=update_splitting_progress)
    
    # Create vector store
    animated_text("Creating vector store... This may take a moment.", effect="bounce", delay=0.1)
    progress_bar.progress(66)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma.from_documents(splits, embeddings, persist_directory=CHROMA_DB_DIR)
    vectorstore.persist()
    qa_chain = create_qa_chain(vectorstore)
    # Store vectorstore in session state for quiz regeneration
    st.session_state.vectorstore = vectorstore
    
    # Complete progress
    progress_bar.progress(100)
    animated_text("Processing complete!", effect="tada", delay=0.1)
    
    # Clear progress indicators after a short delay
    st.success("Documents processed successfully!")
    st.balloons()  # Celebration effect
    
    # Store document statistics in session state
    st.session_state.num_documents = len(docs)
    st.session_state.num_chunks = len(splits)
    st.session_state.avg_chunk_size = int(sum(len(chunk.page_content) for chunk in splits) / len(splits)) if splits else 0

# ------------------ METRICS DASHBOARD ------------------
if 'num_documents' in st.session_state and st.session_state.num_documents > 0:
    st.markdown("---")
    st.subheader("📊 Document Statistics")
    
    # Basic metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Documents", st.session_state.num_documents)
    col2.metric("Chunks", st.session_state.num_chunks)
    col3.metric("Avg. Chunk Size", f"{st.session_state.avg_chunk_size} chars")
    
    # Additional metrics
    if vectorstore:
        # Get total number of vectors in the database
        try:
            collection_count = vectorstore._collection.count()
            st.metric("Vectors in Database", collection_count)
        except:
            pass
            
    # Configuration metrics
    col4, col5, col6 = st.columns(3)
    col4.metric("Chunk Size", st.session_state.chunk_size)
    col5.metric("Chunk Overlap", st.session_state.chunk_overlap)
    col6.metric("Top K", st.session_state.top_k)

if qa_chain:
    # Create tabs for different features
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📚 Summarization", "❓ Quiz", "🔍 Search"])
    
    with tab1:
        animated_text("Ask Questions", effect="fade_in", delay=0.05)
        query = st.text_input("Ask a question:", key="chat_question")
        if query:
            animated_text("Retrieving multiple pages...", effect="pulse", delay=0.1)
            with st.spinner():
                # Increase the number of documents retrieved for more comprehensive answers
                related_docs = custom_retrieve_with_next_chunk(vectorstore, query, k=max(6, st.session_state.top_k))
                context_text = "\n\n".join([doc.page_content for doc in related_docs])
                # Improve the prompt to encourage comprehensive answers
                prompt = f"Provide a comprehensive and detailed answer to the question using the context below. Include all relevant information and explain concepts thoroughly.\n\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
                # Increase max_tokens for more detailed responses
                llm = ChatGroq(model=st.session_state.llm_model, temperature=st.session_state.temperature, max_tokens=min(2048, st.session_state.max_tokens * 2))
                answer = llm.invoke(prompt)
                st.write("### Answer")
                st.write(answer.content)
                st.write("### Sources")
                for doc in related_docs:
                    st.write(f"- {doc.metadata.get('source', 'Unknown')}")
                
                # Save to chat history
                st.session_state.chat_history.append({
                    "question": query,
                    "answer": answer.content,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
    
    with tab2:
        animated_text("Chapter Summarization", effect="fade_in", delay=0.05)
        if st.button("Summarize Document Content"):
            animated_text("Generating summary...", effect="pulse", delay=0.1)
            with st.spinner():
                # Retrieve all documents for summarization
                all_docs = vectorstore.get()
                doc_texts = all_docs["documents"]
                # Limit to first 10 chunks for performance
                doc_texts = doc_texts[:10]
                # Combine documents for summarization (limiting size for performance)
                combined_text = "\n\n".join(doc_texts)
                # Truncate if too long
                if len(combined_text) > 8000:
                    combined_text = combined_text[:8000]
                
                summary_prompt = f"Please provide a comprehensive summary of the following educational content:\n\n{combined_text}"
                llm = ChatGroq(model=st.session_state.llm_model, temperature=st.session_state.temperature, max_tokens=st.session_state.max_tokens)
                summary = llm.invoke(summary_prompt)
                st.write("### Document Summary")
                st.write(summary.content)
    
    with tab3:
        animated_text("Quiz Mode", effect="fade_in", delay=0.05)
        if st.button("Generate Quiz"):
            animated_text("Creating quiz questions...", effect="pulse", delay=0.1)
            with st.spinner():
                # Retrieve all documents for quiz generation
                all_docs = vectorstore.get()
                doc_texts = all_docs["documents"]
                # Limit to first 5 chunks for performance
                doc_texts = doc_texts[:5]
                # Combine documents for quiz generation (limiting size for performance)
                combined_text = "\n\n".join(doc_texts)
                # Truncate if too long
                if len(combined_text) > 4000:
                    combined_text = combined_text[:4000]
                
                quiz_prompt = f"Based on the following educational content, generate 5 multiple-choice quiz questions with 4 options each and indicate the correct answer:\n\n{combined_text}"
                llm = ChatGroq(model=st.session_state.llm_model, temperature=st.session_state.temperature, max_tokens=st.session_state.max_tokens)
                quiz_content = llm.invoke(quiz_prompt)
                
                # Parse and store quiz questions
                st.session_state.quiz_questions = parse_quiz_content(quiz_content.content)
                st.session_state.quiz_answers = {}
                st.session_state.quiz_score = 0
                st.session_state.quiz_completed = False
                st.rerun()
        
        # Display quiz if available
        if st.session_state.quiz_questions:
            display_quiz(st.session_state.quiz_questions)
        else:
            st.info("Click 'Generate Quiz' to create a new quiz.")
    
    with tab4:
        animated_text("Content Search", effect="fade_in", delay=0.05)
        search_query = st.text_input("Enter search terms:", key="search_query")
        if search_query:
            animated_text("Searching documents...", effect="pulse", delay=0.1)
            with st.spinner():
                # Use the retriever to find relevant documents
                search_results = custom_retrieve_with_next_chunk(vectorstore, search_query, k=st.session_state.top_k)
                
                if search_results:
                    st.write(f"Found {len(search_results)} relevant sections:")
                    for i, doc in enumerate(search_results):
                        with st.expander(f"Result {i+1} - {doc.metadata.get('source', 'Unknown')}"):
                            st.write(doc.page_content)
                            st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")
                else:
                    st.info("No results found for your search query.")
