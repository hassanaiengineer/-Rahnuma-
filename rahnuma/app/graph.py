from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from .model import qalb
from .prompts import SYSTEM_PROMPT, SAFETY_REFUSAL, INTENT_CLASSIFICATION_PROMPT
from .safety import check_safety_heuristics

class AgentState(TypedDict):
    user_input: str
    intent: str
    safety_passed: bool
    response: str
    history: List[str]

def classify_intent(state: AgentState):
    # For a production-grade 8B model, we can use the model itself to classify
    # but for speed and strictness in this specific logic, 
    # we'll combine heuristics with a model check if needed.
    
    # Heuristic first
    if not check_safety_heuristics(state["user_input"]):
        return {"intent": "blocked", "safety_passed": False}
    
    classification_prompt = INTENT_CLASSIFICATION_PROMPT.format(user_input=state["user_input"])
    result = qalb.generate(classification_prompt, system_prompt="آپ ایک معاون کلاسیفائر ہیں۔", max_new_tokens=10, temperature=0.1)
    
    intent = result.lower().strip()
    # Cleanup possible model chatter
    if "blocked" in intent: intent = "blocked"
    elif "rewrite" in intent: intent = "rewrite"
    elif "summarize" in intent: intent = "summarize"
    else: intent = "chat"
    
    return {"intent": intent, "safety_passed": intent != "blocked"}

def safety_check(state: AgentState):
    if not state["safety_passed"]:
        return {"response": SAFETY_REFUSAL}
    return state

def generate_response(state: AgentState):
    if state.get("response"): # Already set by safety_check
        return state
        
    full_user_input = state["user_input"]
    if state.get("history"):
        history_str = "\n".join(state["history"])
        full_user_input = f"پچھلی گفتگو:\n{history_str}\n\nصارف کا نیا سوال: {state['user_input']}"
        
    response = qalb.generate(
        prompt=full_user_input,
        system_prompt=SYSTEM_PROMPT,
        max_new_tokens=250,
        temperature=0.4
    )
    return {"response": response}

def post_process(state: AgentState):
    # Trim and clean up output
    response = state["response"].strip()
    return {"response": response}

# Build Graph
builder = StateGraph(AgentState)

builder.add_node("classify", classify_intent)
builder.add_node("safety", safety_check)
builder.add_node("generate", generate_response)
builder.add_node("format", post_process)

builder.set_entry_point("classify")
builder.add_edge("classify", "safety")
builder.add_edge("safety", "generate")
builder.add_edge("generate", "format")
builder.add_edge("format", END)

workflow = builder.compile()
