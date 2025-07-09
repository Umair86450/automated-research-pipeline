# import os
# from dotenv import load_dotenv
# load_dotenv()

# from langgraph.graph import START, StateGraph
# from typing_extensions import TypedDict
# from langchain_groq import ChatGroq
# from langchain_core.tools import tool
# from langchain_community.utilities import SerpAPIWrapper
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.schema import Document

# # Load environment variables
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# # Initialize LLM and tools
# llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.3, api_key=GROQ_API_KEY)
# search_tool = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# vector_db = None

# # Define State
# class State(TypedDict):
#     topic: str
#     subtopics: list[str]
#     current_index: int
#     summaries: list[str]
#     scores: list[float]
#     raw: str
#     summary: str
#     retry_count: int

# # Tools and Nodes
# @tool
# def web_search(query: str) -> str:
#     """Perform a web search using SerpAPI and return the results as a string."""
#     try:
#         return search_tool.run(query)
#     except Exception as e:
#         return f"[ERROR in search]: {e}"

# def planner(state: State) -> State:
#     print("📌 Planning subtopics...")
#     prompt = (
#         f"Split the topic '{state['topic']}' into exactly 3 concise subtopics. "
#         "Return each subtopic as a short phrase (not a full sentence) on a new line, "
#         "without bullet points or numbering."
#     )
#     resp = llm.invoke(prompt)
#     subtopics = [s.strip() for s in resp.content.split("\n") if s.strip()]
#     # Ensure exactly 3 subtopics
#     state["subtopics"] = subtopics[:3] if len(subtopics) >= 3 else subtopics + [""] * (3 - len(subtopics))
#     state["current_index"] = 0
#     state["retry_count"] = 0
#     print(f"Generated subtopics: {state['subtopics']}")
#     return state

# def search_agent(state: State) -> State:
#     current_subtopic = state["subtopics"][state["current_index"]]
#     if not current_subtopic:
#         print(f"⚠️ Warning: Empty subtopic at index {state['current_index']}")
#         state["raw"] = "No valid subtopic provided."
#         return state
#     print(f"🔍 Searching: {current_subtopic}")
#     query = current_subtopic
#     content = web_search.invoke(query)
#     state["raw"] = content
#     return state

# def summarizer(state: State) -> State:
#     print("📝 Summarizing results...")
#     prompt = f"Summarize this text into 5 lines:\n{state['raw']}"
#     resp = llm.invoke(prompt)
#     state["summary"] = resp.content
#     return state

# def evaluator(state: State) -> State:
#     print("📊 Evaluating summary...")
#     prompt = f"Evaluate the following summary and give it a score from 1 to 10:\n{state['summary']}"
#     resp = llm.invoke(prompt)
#     try:
#         score = float(resp.content.strip().split()[0])
#     except:
#         score = 0.0
#     state["scores"].append(score)
#     print(f"Summary score: {score}")
#     return state

# def refiner(state: State) -> State:
#     print("🔁 Refining summary...")
#     prompt = f"Improve the following summary for clarity and completeness:\n{state['summary']}"
#     resp = llm.invoke(prompt)
#     state["summary"] = resp.content
#     state["retry_count"] += 1
#     return state

# def memory_agent(state: State) -> State:
#     global vector_db
#     print("💾 Saving summary to memory...")
#     summary_text = state["summary"]
#     state["summaries"].append(summary_text)
#     state["current_index"] += 1
#     state["retry_count"] = 0

#     doc = Document(page_content=summary_text)
#     if vector_db is None:
#         vector_db = FAISS.from_documents([doc], embeddings)
#     else:
#         vector_db.add_documents([doc])
#     print(f"Current index: {state['current_index']}, Total subtopics: {len(state['subtopics'])}")
#     return state

# def final_compiler(state: State) -> State:
#     print("📚 Compiling final research brief...")
#     prompt = (
#         f"Create a structured research brief for the topic '{state['topic']}' using the following sections. "
#         "Format the output in Markdown with clear headings and concise content:\n\n" +
#         "\n\n".join([f"### Subtopic {i+1}: {state['subtopics'][i]}\n{summary}" for i, summary in enumerate(state["summaries"])])
#     )
#     resp = llm.invoke(prompt)
#     # Save the final brief to a file
#     output = f"# Research Brief: {state['topic']}\n\n{resp.content}"
#     with open("research_brief.md", "w", encoding="utf-8") as f:
#         f.write(output)
#     print("✅ Final Research Brief:\n")
#     print(output)
#     print("\n📄 Saved to 'research_brief.md'")
#     return state

# # Build LangGraph
# graph = StateGraph(State)
# graph.add_node("planner", planner)
# graph.add_node("search", search_agent)
# graph.add_node("summarizer", summarizer)
# graph.add_node("evaluator", evaluator)
# graph.add_node("refiner", refiner)
# graph.add_node("memory", memory_agent)
# graph.add_node("final", final_compiler)

# graph.set_entry_point("planner")
# graph.add_edge("planner", "search")
# graph.add_edge("search", "summarizer")
# graph.add_edge("summarizer", "evaluator")

# # Conditional edge: refine if score < 7 and retries < 3, else save to memory
# graph.add_conditional_edges(
#     "evaluator",
#     lambda s: "memory" if (s["scores"] and s["scores"][-1] >= 7) or s["retry_count"] >= 3 else "refiner",
#     {"refiner": "refiner", "memory": "memory"}
# )
# graph.add_edge("refiner", "memory")

# # Conditional edge: loop to search if more subtopics, else finish
# graph.add_conditional_edges(
#     "memory",
#     lambda s: "search" if s["current_index"] < len(s["subtopics"]) and s["current_index"] < 3 else "final",
#     {"search": "search", "final": "final"}
# )

# graph.set_finish_point("final")
# graph_executor = graph.compile()

# # Entry point
# if __name__ == "__main__":
#     # Prompt user for input
#     topic = input("Please enter the research topic: ").strip()
#     if not topic:
#         topic = "Quantum sensors in agriculture"  # Fallback topic
#         print(f"No topic provided. Using default: {topic}")
    
#     initial_state = {
#         "topic": topic,
#         "subtopics": [],
#         "current_index": 0,
#         "summaries": [],
#         "scores": [],
#         "raw": "",
#         "summary": "",
#         "retry_count": 0
#     }
#     print(f"\n🔬 Starting research on: {topic}\n")
#     graph_executor.invoke(initial_state)




#=================================== =========================================
                # streamlit setting gui in backendcode 
#==================================== =========================================


import os
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# Load environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# Initialize LLM and tools
llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.3, api_key=GROQ_API_KEY)
search_tool = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_db = None

# Define State
class State(TypedDict):
    topic: str
    subtopics: list[str]
    current_index: int
    summaries: list[str]
    scores: list[float]
    raw: str
    summary: str
    retry_count: int
    status: str  # Added for GUI status updates

# Tools and Nodes
@tool
def web_search(query: str) -> str:
    """Perform a web search using SerpAPI and return the results as a string."""
    try:
        return search_tool.run(query)
    except Exception as e:
        return f"[ERROR in search]: {e}"

def planner(state: State) -> State:
    print("📌 Planning subtopics...")
    prompt = (
        f"Split the topic '{state['topic']}' into exactly 3 concise subtopics. "
        "Return each subtopic as a short phrase (not a full sentence) on a new line, "
        "without bullet points or numbering."
    )
    resp = llm.invoke(prompt)
    subtopics = [s.strip() for s in resp.content.split("\n") if s.strip()]
    state["subtopics"] = subtopics[:3] if len(subtopics) >= 3 else subtopics + [""] * (3 - len(subtopics))
    state["current_index"] = 0
    state["retry_count"] = 0
    state["status"] = f"Generated subtopics: {', '.join(state['subtopics'])}"
    return state

def search_agent(state: State) -> State:
    current_subtopic = state["subtopics"][state["current_index"]]
    if not current_subtopic:
        print(f"⚠️ Warning: Empty subtopic at index {state['current_index']}")
        state["raw"] = "No valid subtopic provided."
        state["status"] = "Warning: Empty subtopic encountered"
        return state
    print(f"🔍 Searching: {current_subtopic}")
    query = current_subtopic
    content = web_search.invoke(query)
    state["raw"] = content
    state["status"] = f"Search completed for: {current_subtopic}"
    return state

def summarizer(state: State) -> State:
    print("📝 Summarizing results...")
    prompt = f"Summarize this text into 5 lines:\n{state['raw']}"
    resp = llm.invoke(prompt)
    state["summary"] = resp.content
    state["status"] = "Summary generated"
    return state

def evaluator(state: State) -> State:
    print("📊 Evaluating summary...")
    prompt = f"Evaluate the following summary and give it a score from 1 to 10:\n{state['summary']}"
    resp = llm.invoke(prompt)
    try:
        score = float(resp.content.strip().split()[0])
    except:
        score = 0.0
    state["scores"].append(score)
    state["status"] = f"Summary score: {score}"
    return state

def refiner(state: State) -> State:
    print("🔁 Refining summary...")
    prompt = f"Improve the following summary for clarity and completeness:\n{state['summary']}"
    resp = llm.invoke(prompt)
    state["summary"] = resp.content
    state["retry_count"] += 1
    state["status"] = f"Summary refined (attempt {state['retry_count']})"
    return state

def memory_agent(state: State) -> State:
    global vector_db
    print("💾 Saving summary to memory...")
    summary_text = state["summary"]
    state["summaries"].append(summary_text)
    state["current_index"] += 1
    state["retry_count"] = 0
    doc = Document(page_content=summary_text)
    if vector_db is None:
        vector_db = FAISS.from_documents([doc], embeddings)
    else:
        vector_db.add_documents([doc])
    state["status"] = f"Saved summary {state['current_index']}/{len(state['subtopics'])}"
    return state

def final_compiler(state: State) -> State:
    print("📚 Compiling final research brief...")
    prompt = (
        f"Create a structured research brief for the topic '{state['topic']}' using the following sections. "
        "Format the output in Markdown with clear headings and concise content:\n\n" +
        "\n\n".join([f"### Subtopic {i+1}: {state['subtopics'][i]}\n{summary}" for i, summary in enumerate(state["summaries"])])
    )
    resp = llm.invoke(prompt)
    output = f"# Research Brief: {state['topic']}\n\n{resp.content}"
    with open("research_brief.md", "w", encoding="utf-8") as f:
        f.write(output)
    state["status"] = "Final research brief generated"
    return state

# Build LangGraph
graph = StateGraph(State)
graph.add_node("planner", planner)
graph.add_node("search", search_agent)
graph.add_node("summarizer", summarizer)
graph.add_node("evaluator", evaluator)
graph.add_node("refiner", refiner)
graph.add_node("memory", memory_agent)
graph.add_node("final", final_compiler)

graph.set_entry_point("planner")
graph.add_edge("planner", "search")
graph.add_edge("search", "summarizer")
graph.add_edge("summarizer", "evaluator")
graph.add_conditional_edges(
    "evaluator",
    lambda s: "memory" if (s["scores"] and s["scores"][-1] >= 7) or s["retry_count"] >= 3 else "refiner",
    {"refiner": "refiner", "memory": "memory"}
)
graph.add_edge("refiner", "memory")
graph.add_conditional_edges(
    "memory",
    lambda s: "search" if s["current_index"] < len(s["subtopics"]) and s["current_index"] < 3 else "final",
    {"search": "search", "final": "final"}
)
graph.set_finish_point("final")
graph_executor = graph.compile()