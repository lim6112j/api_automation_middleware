from typing import TypedDict, List
from langgraph.graph import StateGraph, END
import gradio as gr
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
import os # Added for file operations, though not strictly necessary for the tool itself

# --- Tool Definition ---
@tool
def magic_number_tool(input_value: int) -> str:
    """
    Use this tool when you need to calculate a magic number based on an input integer.
    The magic number is the input_value multiplied by 7.
    For example, if the input_value is 3, the result is 'The magic number for 3 is 21.'.
    Only use this tool if the user explicitly asks for a magic number or a calculation that fits this pattern.
    """
    print(f"--- Tool: magic_number_tool called with input: {input_value} ---")
    if not isinstance(input_value, int):
        return "Error: Input value must be an integer."
    return f"The magic number for {input_value} is {input_value * 7}."

@tool
def create_or_edit_file_tool(filename: str, content: str) -> str:
    """
    Use this tool to create a new file or overwrite an existing file with the given content.
    Provide the filename (e.g., 'my_file.txt', 'output/data.json') and the full content for the file.
    Be cautious with file paths; by default, files will be created in the current working directory.
    Example: To create a file named 'example.txt' with 'Hello, World!' as content,
    call with filename='example.txt' and content='Hello, World!'.
    """
    print(f"--- Tool: create_or_edit_file_tool called with filename: {filename} ---")
    # SECURITY WARNING: This tool writes to the filesystem.
    # In a real application, ensure proper sandboxing and path validation.
    print(f"WARNING: Attempting to write to filesystem: {os.path.abspath(filename)}")
    try:
        # Ensure directory exists if a path is specified (simple case for one level)
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"Created directory: {directory}")
            except Exception as e:
                return f"Error creating directory for file {filename}: {e}"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully created or edited file: {filename}"
    except IOError as e:
        return f"Error creating or editing file {filename}: {e}"
    except Exception as e: # Catch any other unexpected errors
        return f"An unexpected error occurred while handling file {filename}: {e}"

defined_tools = [magic_number_tool, create_or_edit_file_tool] # Added the new tool

# --- LLM Initialization ---
llm = None
llm_with_tools = None
LLM_INITIALIZED = False
LLM_ERROR_MESSAGE = ""

try:
    from langchain_openai import ChatOpenAI
    # IMPORTANT: Ensure you have OPENAI_API_KEY set in your environment
    # and the langchain-openai package is installed (pip install langchain-openai)
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    llm_with_tools = llm.bind_tools(defined_tools)
    LLM_INITIALIZED = True
    print("ChatOpenAI initialized successfully and tools are bound.")
except ImportError:
    LLM_ERROR_MESSAGE = "Error: langchain-openai package not found. Please install it (`pip install langchain-openai`) to use the chatbot."
    print(LLM_ERROR_MESSAGE)
except Exception as e:
    LLM_ERROR_MESSAGE = f"Error initializing ChatOpenAI: {e}. Please ensure OPENAI_API_KEY is set correctly."
    print(LLM_ERROR_MESSAGE)


# 1. Define the State for the LangGraph application
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        messages: A list of messages (Human, AI, Tool) that form the conversation.
        steps: A list to keep track of graph operations for logging/debugging.
    """
    messages: List[BaseMessage]
    steps: List[str]

# 2. Define LangGraph Nodes
def entry_node(state: GraphState) -> GraphState:
    """
    Node that handles the initial setup.
    Messages list is expected to be pre-populated with the initial HumanMessage.
    """
    if 'steps' not in state or not state['steps']: # Initialize steps if not already
        state['steps'] = []
    state['steps'].append(f"Entry node: Initial messages received ({len(state['messages'])}).")
    return state

def chatbot_node(state: GraphState) -> GraphState:
    """
    Node that uses an LLM to respond to the current conversation history (messages).
    The LLM may respond directly or decide to call a tool.
    """
    state['steps'].append("Chatbot node entered.")
    if not LLM_INITIALIZED or llm_with_tools is None:
        error_content = f"LLM not available. {LLM_ERROR_MESSAGE}"
        error_ai_message = AIMessage(content=error_content)
        state['messages'].append(error_ai_message)
        state['steps'].append(f"LLM invocation skipped: {error_content}")
        return state

    current_messages = state['messages']
    state['steps'].append(f"Sending {len(current_messages)} messages to LLM.")
    
    try:
        # The LLM is already bound with tools
        response_message = llm_with_tools.invoke(current_messages)
        state['messages'].append(response_message)
        
        if response_message.tool_calls:
            state['steps'].append(f"LLM responded with tool calls: {response_message.tool_calls}")
        else:
            state['steps'].append(f"LLM responded with content: '{response_message.content[:100]}...'")
            
    except Exception as e:
        error_message_content = f"Error invoking LLM: {e}"
        state['steps'].append(error_message_content)
        # Append an AIMessage with the error to ensure the messages list always ends with an AI response type
        state['messages'].append(AIMessage(content=f"An error occurred: {error_message_content}"))
    
    return state

def tool_node(state: GraphState) -> GraphState:
    """
    Node that executes tools called by the LLM.
    """
    state['steps'].append("Tool node entered.")
    last_message = state['messages'][-1]
    
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        state['steps'].append("No tool calls found in the last AIMessage or last message is not AIMessage.")
        # This path should ideally not be hit if routing is correct.
        return state

    tool_call_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call['name'] # tool_call is a dict-like object from AIMessage
        tool_args = tool_call['args']
        tool_call_id = tool_call['id']
        
        state['steps'].append(f"Executing tool: {tool_name} with args: {tool_args} (Call ID: {tool_call_id})")
        
        selected_tool = None
        for t in defined_tools:
            if t.name == tool_name:
                selected_tool = t
                break
        
        if selected_tool:
            try:
                # The @tool decorator and .invoke handle argument passing.
                # tool_args is a dict, e.g., {'input_value': 5} or {'filename': 'f.txt', 'content': 'c'}
                result = selected_tool.invoke(tool_args)
                tool_call_messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_call_id)
                )
                state['steps'].append(f"Tool {tool_name} executed. Result: {result}")
            except Exception as e:
                error_msg = f"Error executing tool {tool_name}: {e}"
                state['steps'].append(error_msg)
                tool_call_messages.append(
                    ToolMessage(content=error_msg, tool_call_id=tool_call_id)
                )
        else:
            error_msg = f"Tool '{tool_name}' not found."
            state['steps'].append(error_msg)
            tool_call_messages.append(
                ToolMessage(content=error_msg, tool_call_id=tool_call_id)
            )
            
    state['messages'].extend(tool_call_messages)
    state['steps'].append(f"Added {len(tool_call_messages)} tool messages to state.")
    return state

# 3. Define the Router for Conditional Edges
def route_messages(state: GraphState) -> str:
    """
    Router function to decide the next step based on the last message.
    - If the last AIMessage has tool_calls, route to 'tool_node'.
    - Otherwise (AIMessage with no tool_calls), route to END.
    """
    last_message = state['messages'][-1]
    state['steps'].append(f"Router: Last message type is {type(last_message).__name__}.")

    if isinstance(last_message, AIMessage):
        if last_message.tool_calls:
            state['steps'].append("Router: AIMessage has tool calls. Routing to tool_node.")
            return "tool_node"
        else:
            state['steps'].append("Router: AIMessage has no tool calls. Routing to END.")
            return END
    
    state['steps'].append("Router: Last message not AIMessage or no specific route. Routing to END (fallback).")
    return END

# 4. Build the Graph
workflow = StateGraph(GraphState)

# Add the nodes to the graph
workflow.add_node("entry", entry_node)
workflow.add_node("chatbot", chatbot_node)
workflow.add_node("tool", tool_node)

# Set the entry point for the graph
workflow.set_entry_point("entry")

# Add edges to define the flow
workflow.add_edge("entry", "chatbot")
workflow.add_conditional_edges(
    "chatbot",          # Source node
    route_messages,     # Function to decide the route
    {                   # Mapping of router output to destination nodes
        "tool_node": "tool",
        END: END
    }
)
workflow.add_edge("tool", "chatbot") # After tools are run, go back to chatbot to process results

# Compile the graph into a runnable application
app_runnable = workflow.compile()
print("LangGraph workflow compiled.")

# 5. Define the Gradio interaction function
def run_langgraph_app(user_input: str) -> str:
    """
    This function will be called by Gradio.
    It takes user input, runs it through the LangGraph, and returns the final AI response.
    """
    if not user_input:
        return "Please enter some text."

    initial_graph_state: GraphState = {
        "messages": [HumanMessage(content=user_input)],
        "steps": [] 
    }
    
    final_state = app_runnable.invoke(initial_graph_state)
    
    # For debugging, print the steps:
    # print("\n--- Graph Execution Steps ---")
    # for step in final_state.get('steps', []):
    #     print(step)
    # print("--- End Graph Execution Steps ---\n")

    if final_state['messages']:
        final_ai_message = None
        for msg in reversed(final_state['messages']):
            if isinstance(msg, AIMessage):
                final_ai_message = msg
                break
        
        if final_ai_message:
            return final_ai_message.content
        else: 
            return "Chatbot did not provide a final AI response."

    return "Could not get a response from the chatbot. The message list was empty."

# 6. Create and Launch the Gradio Interface
if __name__ == "__main__":
    gradio_description = (
        "Enter some text. The chatbot may use tools (like a magic number calculator or file creator) to help answer.\n"
        "Try: 'what is the magic number for 10?' or 'create a file named hello.txt with the content Hello world from the chatbot!'"
    )
    if not LLM_INITIALIZED:
        gradio_description += f"\n\n**Warning: {LLM_ERROR_MESSAGE} The chatbot functionality will be impaired or non-functional.**"
    
    iface = gr.Interface(
        fn=run_langgraph_app,
        inputs=gr.Textbox(lines=1, placeholder="Ask a question, try 'magic number for 5?', or 'create file test.txt with content hello'"),
        outputs=gr.Textbox(label="Chatbot Response", lines=5),
        title="LangGraph Chatbot with Tools",
        description=gradio_description
    )
    
    print("Launching Gradio interface...")
    iface.launch()
