from typing import TypedDict, List
from langgraph.graph import StateGraph, END
import gradio as gr

# 1. Define the State for the LangGraph application
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        input_text: The initial text provided by the user.
        processed_text: The text after processing by the graph.
        steps: A list to keep track of operations (for demonstration).
    """
    input_text: str
    processed_text: str
    steps: List[str]

# 2. Define LangGraph Nodes
def entry_node(state: GraphState) -> GraphState:
    """
    Node that handles the initial input.
    For this example, it just records the input.
    The 'input_text' is expected to be set when invoking the graph.
    """
    state['steps'].append(f"Received input: {state['input_text']}")
    # 'processed_text' is not set here, but initialized if needed
    if 'processed_text' not in state:
        state['processed_text'] = ""
    return state

def processing_node(state: GraphState) -> GraphState:
    """
    Node that performs some processing on the input_text.
    For this example, it converts the text to uppercase.
    """
    original_text = state['input_text']
    processed = original_text.upper()
    state['processed_text'] = processed
    state['steps'].append(f"Processed '{original_text}' to '{processed}'")
    return state

# 3. Build the Graph
workflow = StateGraph(GraphState)

# Add the nodes to the graph
workflow.add_node("entry", entry_node)
workflow.add_node("processor", processing_node)

# Set the entry point for the graph
workflow.set_entry_point("entry")

# Add edges to define the flow
workflow.add_edge("entry", "processor")
workflow.add_edge("processor", END) # END signifies the completion of the graph flow

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
        "steps": []
    }
    
    # Invoke the LangGraph runnable
    final_state = app_runnable.invoke(initial_state)
    
    # For debugging or more detailed output, you can inspect final_state['steps']
    # print(f"Graph execution steps: {final_state['steps']}")
    
    return final_state["processed_text"]

# 5. Create and Launch the Gradio Interface
if __name__ == "__main__":
    iface = gr.Interface(
        fn=run_langgraph_app,
        inputs=gr.Textbox(lines=2, placeholder="Enter text here..."),
        outputs=gr.Textbox(label="Processed Text"),
        title="Simple LangGraph App with Gradio",
        description="Enter some text, and the LangGraph will process it (convert to uppercase)."
    )
    
    print("Launching Gradio interface...")
    iface.launch()
