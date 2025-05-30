from typing import TypedDict, List
from langgraph.graph import StateGraph, END
import gradio as gr
from langchain_core.messages import HumanMessage

# Attempt to import ChatOpenAI and handle potential errors
try:
    from langchain_openai import ChatOpenAI
    # Initialize the LLM (globally for efficiency)
    # IMPORTANT: Ensure you have OPENAI_API_KEY set in your environment
    # and the langchain-openai package is installed (pip install langchain-openai)
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    LLM_INITIALIZED = True
    LLM_ERROR_MESSAGE = ""
except ImportError:
    llm = None
    LLM_INITIALIZED = False
    LLM_ERROR_MESSAGE = "Error: langchain-openai package not found. Please install it (`pip install langchain-openai`) to use the chatbot."
    print(LLM_ERROR_MESSAGE)
except Exception as e:
    llm = None
    LLM_INITIALIZED = False
    LLM_ERROR_MESSAGE = f"Error initializing ChatOpenAI: {e}. Please ensure OPENAI_API_KEY is set correctly."
    print(LLM_ERROR_MESSAGE)


# 1. Define the State for the LangGraph application
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        input_text: The initial text provided by the user.
        processed_text: The text after processing by the graph (chatbot's response).
        steps: A list to keep track of operations.
    """
    input_text: str
    processed_text: str
    steps: List[str]

# 2. Define LangGraph Nodes
def entry_node(state: GraphState) -> GraphState:
    """
    Node that handles the initial input.
    """
    state['steps'] = [f"Received input: {state['input_text']}"] # Initialize steps
    if 'processed_text' not in state:
        state['processed_text'] = ""
    return state

def chatbot_node(state: GraphState) -> GraphState:
    """
    Node that uses an LLM to respond to the input_text.
    """
    if not LLM_INITIALIZED or llm is None:
        state['processed_text'] = f"LLM not available. {LLM_ERROR_MESSAGE}"
        state['steps'].append("LLM invocation skipped due to initialization issues.")
        return state

    user_input = state['input_text']
    state['steps'].append(f"Sending to LLM: '{user_input}'")
    
    try:
        messages = [HumanMessage(content=user_input)]
        response = llm.invoke(messages)
        processed_text = response.content
    except Exception as e:
        error_message = f"Error invoking LLM: {e}"
        processed_text = error_message
        state['steps'].append(error_message)
        # You might want to log the full error e for more detailed debugging
    
    state['processed_text'] = processed_text
    state['steps'].append(f"LLM responded: '{processed_text}'")
    return state

# 3. Build the Graph
workflow = StateGraph(GraphState)

# Add the nodes to the graph
workflow.add_node("entry", entry_node)
workflow.add_node("chatbot", chatbot_node) # Changed from "processor" to "chatbot"

# Set the entry point for the graph
workflow.set_entry_point("entry")

# Add edges to define the flow
workflow.add_edge("entry", "chatbot")     # Changed edge to point to "chatbot"
workflow.add_edge("chatbot", END)       # Changed edge from "chatbot" to END

# Compile the graph into a runnable application
app_runnable = workflow.compile()

# 4. Define the Gradio interaction function
def run_langgraph_app(user_input: str) -> str:
    """
    This function will be called by Gradio.
    It takes user input, runs it through the LangGraph, and returns the output.
    """
    if not user_input:
        return "Please enter some text."

    # Initial state for the graph invocation
    initial_state: GraphState = {
        "input_text": user_input,
        "processed_text": "",
        "steps": []  # steps will be populated by the nodes
    }
    
    # Invoke the LangGraph runnable
    final_state = app_runnable.invoke(initial_state)
    
    # For debugging or more detailed output, you can inspect final_state['steps']
    # print(f"Graph execution steps: {final_state['steps']}")
    
    return final_state["processed_text"]

# 5. Create and Launch the Gradio Interface
if __name__ == "__main__":
    gradio_description = "Enter some text, and the LangGraph with an LLM will provide a response."
    if not LLM_INITIALIZED:
        gradio_description += f"\n\n**Warning: {LLM_ERROR_MESSAGE} The chatbot functionality will be impaired.**"
    
    iface = gr.Interface(
        fn=run_langgraph_app,
        inputs=gr.Textbox(lines=3, placeholder="Enter your message for the chatbot here..."),
        outputs=gr.Textbox(label="Chatbot Response"),
        title="LangGraph Chatbot with Gradio", # Updated title
        description=gradio_description # Updated description
    )
    
    print("Launching Gradio interface...")
    iface.launch()
