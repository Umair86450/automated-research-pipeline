import streamlit as st
from research_assistance import graph_executor, State, vector_db
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Custom CSS for attractive styling
st.markdown("""
    <style>
    .main { background-color: #f0f4f8; }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .stTextInput>div>input {
        border: 2px solid #4CAF50;
        border-radius: 8px;
        padding: 10px;
    }
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-right: 2px solid #4CAF50;
    }
    .stMarkdown h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Arial', sans-serif;
    }
    .status-box {
        background-color: #e8f4f0;
        padding: 10px;
        border-radius: 8px;
        border-left: 5px solid #4CAF50;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit app configuration
st.set_page_config(page_title="Research Assistant", page_icon="🔬", layout="wide")

# Initialize session state
if "current_step" not in st.session_state:
    st.session_state.current_step = 0

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/research.png", use_container_width=True)
    st.title("🔍 Research Assistant")
    st.markdown("Generate structured research briefs on any topic with AI-powered insights.")
    st.markdown("---")
    st.markdown("**Settings**")
    theme = st.selectbox("Theme", ["Light", "Dark"])
    if theme == "Dark":
        st.markdown("""
            <style>
            .main { background-color: #2c3e50; color: #ecf0f1; }
            .stMarkdown h1, h2, h3 { color: #ecf0f1; }
            .status-box { background-color: #34495e; border-left-color: #1abc9c; }
            </style>
        """, unsafe_allow_html=True)

# Main content
st.title("🌟 AI-Powered Research Assistant")
st.markdown("Enter a topic below to generate a comprehensive research brief, powered by advanced AI and web search.")

# Input form
with st.form(key="research_form"):
    topic = st.text_input("📝 Research Topic", placeholder="e.g., Machine learning in healthcare", help="Enter a topic or use the default.")
    submit_button = st.form_submit_button(label="🚀 Generate Research Brief")

if submit_button:
    if not topic.strip():
        topic = "Quantum sensors in agriculture"
        st.info(f"ℹ️ No topic provided. Using default: **{topic}**")
    
    st.markdown(f"### Starting Research on: **{topic}**")
    progress_bar = st.progress(0)
    status_container = st.container()

    # Initialize state
    initial_state = {
        "topic": topic,
        "subtopics": [],
        "current_index": 0,
        "summaries": [],
        "scores": [],
        "raw": "",
        "summary": "",
        "retry_count": 0,
        "status": "Initializing..."
    }

    # Estimate total steps (3 subtopics * (search + summarize + evaluate + up to 3 retries) + planner + final)
    total_steps = 3 * (1 + 1 + 1 + 3) + 2
    st.session_state.current_step = 0

    # Custom progress update function
    def update_progress(state, step_increment=1):
        st.session_state.current_step += step_increment
        progress = min(st.session_state.current_step / total_steps, 1.0)
        progress_bar.progress(progress)
        with status_container:
            st.markdown(f'<div class="status-box">📌 {state.get("status", "Processing...")}</div>', unsafe_allow_html=True)

    # Run graph with progress updates
    with st.spinner("🔄 Processing research..."):
        state = initial_state
        try:
            for chunk in graph_executor.stream(state):
                for node, state_update in chunk.items():
                    state.update(state_update)
                    if node == "planner":
                        update_progress(state)
                    elif node == "search":
                        update_progress(state)
                    elif node == "summarizer":
                        update_progress(state)
                    elif node == "evaluator":
                        update_progress(state)
                    elif node == "refiner":
                        update_progress(state)
                    elif node == "memory":
                        update_progress(state)
                    elif node == "final":
                        update_progress(state, 2)  # Final step is significant
                        break
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.stop()

    # Display final research brief
    if state.get("status") == "Final research brief generated":
        st.markdown("## 📚 Final Research Brief")
        with open("research_brief.md", "r", encoding="utf-8") as f:
            brief_content = f.read()
        st.markdown(brief_content, unsafe_allow_html=True)
        st.success("✅ Research brief generated and saved to 'research_brief.md'")
        
        # Provide download button
        with open("research_brief.md", "rb") as f:
            st.download_button(
                label="📥 Download Research Brief",
                data=f,
                file_name="research_brief.md",
                mime="text/markdown"
            )

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and xAI's Grok | © 2025")