import time
import streamlit as st
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langgraph.types import Command

# Import our structured modules
from src.models import Concept, GraphState
from src.services import (
    PDFService, LLMService, GraphService, 
    IntentService, PlannerService, DiagnoserService, TutorWorkflow
)
from src.database import StudentDB, Neo4jConn

load_dotenv()

# Page Config
st.set_page_config(page_title="Structured Brain Tutor", page_icon="📚", layout="wide")

# Initialize Services in session state to avoid re-init
if "pdf_service" not in st.session_state:
    st.session_state.pdf_service = PDFService()
if "llm_service" not in st.session_state:
    st.session_state.llm_service = LLMService()
if "neo_conn" not in st.session_state:
    st.session_state.neo_conn = Neo4jConn()
if "graph_service" not in st.session_state:
    st.session_state.graph_service = GraphService(st.session_state.neo_conn)
if "tutor_workflow" not in st.session_state:
    st.session_state.tutor_workflow = TutorWorkflow(st.session_state.neo_conn)

# UI Header
st.title("📚 AI Learning Brain")

# Sidebar - User Session
st.sidebar.header("User Session")
student_id = st.sidebar.text_input("Enter Student ID", value="STUDENT001")

if student_id:
    student_db = StudentDB(student_id)
    st.sidebar.divider()
    st.sidebar.subheader("System Status")
    if st.session_state.neo_conn.is_connected():
        st.sidebar.success("Knowledge Base: Online ✅")
    else:
        st.sidebar.error("Knowledge Base: Offline ❌")
        st.sidebar.info("Please Contact Support or check Configuration.")

    last_json_path = student_db.get_last_json_path()
    
    if last_json_path and os.path.exists(last_json_path):
        st.sidebar.success(f"Active Session: {student_id}")
        st.sidebar.info(f"Loaded: `{os.path.basename(last_json_path)}`")
        
        if st.sidebar.button("Process New PDF"):
            student_db.update_last_json_path(None)
            st.session_state.messages = []
            st.session_state.graph_state = None
            st.session_state.concepts = []
            st.rerun()
            
        # Load concepts if not present or if list is empty
        if not st.session_state.get("concepts"):
            st.session_state.concepts = st.session_state.graph_service.load_concepts_from_json(last_json_path)
    else:
        st.sidebar.warning("Ready for new content.")
        st.session_state.concepts = []

# Main UI Tabs
tab1, tab2 = st.tabs(["📖 Learning Session", "🛠 Content Processing"])

with tab1:
    if not st.session_state.get("concepts"):
        st.info("To start, please go to the 'Content Processing' tab and upload your learning material.")
    else:
        st.subheader("Chat with your Tutor")
        
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Welcome! I've loaded your knowledge brain. What would you like to learn today?"}]
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_input := st.chat_input("I want to learn about..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.spinner("Tutor is thinking..."):
                tutor = st.session_state.tutor_workflow.app
                thread_id = f"thread_{student_id}"
                config = {"configurable": {"thread_id": thread_id}}
                
                state = tutor.get_state(config)
                # Don't send the literal placeholder as a quiz answer while interrupted.
                _ui = user_input.strip()
                if _ui.lower() in ("i want to learn about...", "i want to learn about"):
                    user_input = ""

                if not state.values and not (user_input or "").strip():
                    st.warning("Please type a topic, e.g. **I want to learn about transistor biasing**.")
                    st.stop()

                # Check if the state exists and has a next step (not at END)
                if state.values and state.next: # Already in an active session
                    result = tutor.invoke(Command(resume=user_input), config=config)
                else: # Start fresh
                    initial_state = GraphState(
                        student_id=student_id,
                        messages=[],
                        current_input=user_input,
                        target_topics=[],
                        known_topics=[],
                        current_concept="",
                        current_question="",
                        student_answer="",
                        answer_score=0.0,
                        diagnosis_report={},
                        planned_paths=[],
                        current_path_index=0,
                        current_concept_index=0,
                        final_response="",
                        is_transition=False,
                        phase="quiz",
                    )
                    result = tutor.invoke(initial_state, config=config)
                
                bot_response = result.get("final_response", "I'm ready to help.")
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                with st.chat_message("assistant"):
                    st.markdown(bot_response)
                    if result.get("current_question"):
                        st.markdown(f"**Recall Question:** {result['current_question']}")

with tab2:
    st.subheader("Process Learning Content")
    uploaded_file = st.file_uploader("Upload a PDF Textbook", type=["pdf"])
    
    if uploaded_file and st.button("Generate Learning Brain"):
        with st.status("Processing Content...", expanded=True) as status:
            # 1. Save temp
            temp_path = os.path.join("data", f"temp_{uploaded_file.name}")
            with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())
            
            # 2. Extract MD
            status.write("Preparing Document...")
            md_text = st.session_state.pdf_service.to_markdown(temp_path)
            
            # 3. Process Chunks
            status.write("Analyzing Structure...")
            chunks = st.session_state.pdf_service.split_and_clean(md_text)
            
            # 4. Analyze Concepts
            status.write("Identifying Key Concepts...")
            concepts = []
            errors_count = 0
            pbar = st.progress(0)
            process_limit = 50
            actual_chunks = chunks[:process_limit]
            
            for i, c in enumerate(actual_chunks):
                try:
                    res = st.session_state.llm_service.extract_concept(c.page_content, section=c.metadata.get("section",""))
                    if res and res.topic != "SKIP":
                        concepts.append(res)
                except Exception:
                    errors_count += 1
                pbar.progress((i+1)/len(actual_chunks))
                time.sleep(2)
            
            if errors_count > 0:
                st.toast(f"Notice: Some sections were skipped to maintain processing quality.", icon="ℹ️")
            
            if not concepts:
                status.update(label="Processing Failed", state="error")
                st.error("No relevant concepts identified. Please try a more detailed document.")
                st.stop()
            
            # 5. Save persistent data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = os.path.join("data", f"concepts_{student_id}_{timestamp}.json")
            with open(json_path, "w") as f:
                json.dump([c.model_dump() for c in concepts], f, indent=2)
            
            # 6. Finalize Knowledge Base
            status.write("Finalizing Learning Brain...")
            st.session_state.graph_service.build_graph_from_json(json_path)
            
            # 7. Update User Session
            student_db.update_last_json_path(json_path)
            st.session_state.concepts = concepts
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            status.update(label="Brain Ready!", state="complete")
            st.success("Your learning brain is ready. Head over to the 'Learning Session' tab.")
            st.rerun()

    if st.session_state.get("concepts"):
        st.divider()
        st.subheader("Your Knowledge Inventory")
        st.write(f"Total topics: {len(st.session_state.concepts)}")
        st.table([{"Topic": c.topic, "Complexity": c.difficulty_score} for c in st.session_state.concepts[:10]])
