import logging
from datetime import datetime
from typing import Dict, Any, TypedDict, Optional, List
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
# Add these imports near the top of the file, with your other LangChain imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("workflow")

# Define the state structure using TypedDict
class WorkflowState(TypedDict):
    prompt: str
    project_id: str
    project_name: Optional[str]
    original_prompt: str
    answer: Optional[str]
    clarification_needed: bool
    clarification_message: Optional[str]
    suggestions: list
    error: Optional[str]
    messages: List[Dict]  # To store conversation history

class RAGWorkflow:
    """Manages the RAG workflow with proper tool integration and dependency management"""
    
    def __init__(self, project_loader, vector_store_manager, rag_chain_factory, tools):
        self.project_loader = project_loader
        self.vector_store_manager = vector_store_manager
        self.rag_chain_factory = rag_chain_factory
        self.tools = tools
        self.tool_node = ToolNode(tools)
        self.builder = StateGraph(WorkflowState)
        self._build_graph()
        logger.info("RAGWorkflow initialized with tools")
    
    def _build_graph(self):
        """Build the workflow graph with tool support"""
        logger.debug("Building workflow graph with tool support")
        
        # Add nodes
        self.builder.add_node("identify_project", self._identify_project)
        self.builder.add_node("clarify_project", self._clarify_project)
        self.builder.add_node("generate_answer", self._generate_answer)
        self.builder.add_node("tools", self.tool_node)
        
        # Define edges
        self.builder.add_edge(START, "identify_project")
        
        # Conditional edges from identify_project
        self.builder.add_conditional_edges(
            "identify_project",
            self._project_identification_check,
            {
                "identified": "generate_answer",
                "ambiguous": "clarify_project",
                "not_found": "clarify_project"
            }
        )
        
        # Edges from clarify_project
        self.builder.add_edge("clarify_project", "generate_answer")
        
        # Tool handling in answer generation
        self.builder.add_conditional_edges(
            "generate_answer",
            tools_condition,
            {
                "tools": "tools",
                END: END
            }
        )
        
        # Loop back to generate_answer after tool execution
        self.builder.add_edge("tools", "generate_answer")
        
        # Compile the graph with memory saver
        self.graph = self.builder.compile(checkpointer=MemorySaver())
        logger.debug("Workflow graph built with tool support")
    
    def _project_identification_check(self, state: WorkflowState) -> str:
        """Determine which path to take after project identification"""
        logger.debug(f"Project identification check. State keys: {list(state.keys())}")
        
        if state.get("project_id") and state["project_id"] != "-1":
            logger.debug(f"Project identified: {state['project_id']}")
            return "identified"
        elif state.get("suggestions"):
            logger.debug(f"Project ambiguous. Suggestions: {state['suggestions']}")
            return "ambiguous"
        else:
            logger.debug("Project not found")
            return "not_found"
    
    def _identify_project(self, state: WorkflowState) -> WorkflowState:
        """Identify the project based on the query"""
        logger.debug(f"Entering _identify_project. State keys: {list(state.keys())}")
        
        # Initialize messages if not present
        if "messages" not in state:
            state["messages"] = []
        
        # Check if prompt exists
        if "prompt" not in state:
            logger.error("State is missing 'prompt' key!")
            return {
                "prompt": "Unknown prompt",
                "project_id": "-1",
                "original_prompt": "Unknown prompt",
                "clarification_needed": True,
                "clarification_message": "I couldn't understand your question. Please rephrase it.",
                "suggestions": [],
                "messages": state["messages"]
            }
        
        # If project_id is already provided and valid, use it
        if state.get("project_id") and state["project_id"] != "-1":
            logger.debug(f"Using provided project_id: {state['project_id']}")
            state["project_name"] = self.project_loader.get_project_name(state["project_id"])
            return state
        
        # Otherwise, identify project from query
        logger.debug(f"Identifying project for prompt: {state['prompt']}")
        result = self.project_loader.identify_project(state["prompt"])
        
        # Create a new state object
        new_state = {
            "prompt": state["prompt"],
            "original_prompt": state["original_prompt"],
            "clarification_needed": False,
            "suggestions": [],
            "project_id": "-1",
            "messages": state["messages"]
        }
        
        if result["status"] == "selected":
            logger.info(f"Project identified: {result['project_id']} - {result['project_name']}")
            new_state["project_id"] = result["project_id"]
            new_state["project_name"] = result["project_name"]
        else:
            logger.info(f"Project identification result: {result['status']}")
            new_state["suggestions"] = result["suggestions"]
            new_state["clarification_needed"] = True
            
            if result["status"] == "ambiguous":
                # Create a clarification message for ambiguous projects
                suggestions = [f"{s['name']} (ID: {s['id']})" for s in result["suggestions"]]
                new_state["clarification_message"] = (
                    f"I found multiple possible projects related to your question: "
                    f"{', '.join(suggestions[:3])}. "
                    f"Which one did you mean?"
                )
            else:
                # Create a clarification message when no projects match
                new_state["clarification_message"] = (
                    "I couldn't match your question to any specific project. "
                    "Please select a project from the dropdown or rephrase your question."
                )
        
        logger.debug(f"Exiting _identify_project. State keys: {list(new_state.keys())}")
        return new_state
    
    def _clarify_project(self, state: WorkflowState) -> WorkflowState:
        """Handle project clarification"""
        logger.debug(f"Entering _clarify_project. State keys: {list(state.keys())}")
        return state
    
    def _generate_answer(self, state: WorkflowState) -> WorkflowState:
        """Generate an answer using the identified project with tool support"""
        logger.debug(f"Entering _generate_answer. State keys: {list(state.keys())}")
        
        # Initialize messages if not present
        if "messages" not in state:
            state["messages"] = []
        
        # Add user message if not already in history
        if not any(isinstance(m, dict) and m.get("role") == "user" and m.get("content") == state["prompt"] 
                  for m in state["messages"]):
            state["messages"].append({"role": "user", "content": state["prompt"]})
        
        if not state.get("project_id") or state["project_id"] == "-1":
            logger.error("No valid project_id in state for answer generation")
            state["answer"] = "I couldn't determine which project you're asking about. Please select a project from the dropdown."
            state["messages"].append({"role": "assistant", "content": state["answer"]})
            return state
        
        # Get vector store
        logger.debug(f"Getting vector store for project {state['project_id']}")
        vector_store = self.vector_store_manager.get_vector_store(state["project_id"])
        if not vector_store:
            logger.error(f"No vector store for project {state['project_id']}")
            state["answer"] = f"No knowledge base available for project {state.get('project_name', state['project_id'])}."
            state["messages"].append({"role": "assistant", "content": state["answer"]})
            return state
        
        # Create retriever
        logger.debug("Creating retriever")
        from rag.retriever import TwoStageRetriever
        retriever = TwoStageRetriever(vector_store, state["project_id"], self.project_loader.settings)
        
        # Retrieve relevant documents
        logger.debug(f"Retrieving documents for prompt: {state['prompt']}")
        context_docs = retriever.get_relevant_documents(state["prompt"])
        
        if not context_docs:
            logger.warning("No relevant documents found")
            state["answer"] = "I couldn't find relevant information to answer your question."
            state["messages"].append({"role": "assistant", "content": state["answer"]})
            return state
        
        # Create tool-enabled RAG chain
        logger.debug(f"Generating answer with RAG chain for project {state['project_name']}")
        return self._generate_tool_enabled_answer(state, context_docs)
    
    def _generate_tool_enabled_answer(self, state: WorkflowState, context_docs: list) -> WorkflowState:
        """Generate an answer using tool-enabled agent"""
        from langchain_openai import ChatOpenAI
        
        # Format context
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Create prompt template that instructs the LLM to use tools for date information
        system_message = f"""You are a helpful assistant for {state['project_name']}.
When answering questions about dates, events, or time-sensitive information:
1. ALWAYS use the GetCurrentDate tool to determine today's date
2. Use the IsDatePassed tool to check if dates have passed
3. NEVER assume or make up dates

Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}"""

        # Create tool-enabled agent
        llm = ChatOpenAI(
            model_name=self.project_loader.settings.LLM_MODEL,
            temperature=self.project_loader.settings.LLM_TEMPERATURE,
            openai_api_key=self.project_loader.settings.OPENAI_API_KEY
        )
        
        # Convert messages to LangChain format
        messages = []
        for msg in state["messages"]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        
        # If no messages or last message is user message, create new prompt
        if not messages or isinstance(messages[-1], HumanMessage):
            # Add system message
            messages.insert(0, HumanMessage(content=system_message))
            
            # Create agent with proper tool-calling capabilities
            from langchain.agents import create_tool_calling_agent
            agent = create_tool_calling_agent(
                llm,
                self.tools,
                ChatPromptTemplate.from_messages([
                    ("system", system_message),
                    MessagesPlaceholder("chat_history", optional=True),
                    ("human", "{input}"),
                    MessagesPlaceholder("agent_scratchpad")
                ])
            )
            
            # Create agent executor
            from langchain.agents import AgentExecutor
            agent_executor = AgentExecutor(
                agent=agent, 
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5  # Allow multiple tool calls if needed
            )
            
            # Run agent
            try:
                response = agent_executor.invoke({
                    "input": state["prompt"],
                    "chat_history": messages[1:-1]  # Exclude system message
                })

                # Format answer with project identification and sources
                rag_chain = self.rag_chain_factory(state["project_name"])
                formatted = rag_chain.format_response_with_sources(
                    response["output"],
                    context_docs,
                    state["project_id"],
                    state["project_name"],
                )

                # Update state with formatted answer
                state["answer"] = formatted
                state["messages"].append({"role": "assistant", "content": state["answer"]})
                
            except Exception as e:
                logger.error(f"Error in tool-enabled agent: {str(e)}")
                # Fallback to simple RAG if tool execution fails
                rag_chain = self.rag_chain_factory(state["project_name"])
                answer = rag_chain.generate_answer(state["prompt"], context_docs)
                state["answer"] = rag_chain.format_response_with_sources(
                    answer, context_docs, state["project_id"], state["project_name"]
                )
                state["messages"].append({"role": "assistant", "content": state["answer"]})
        
        return state
    
    def run(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the workflow with the given initial state"""
        logger.info("Starting workflow run")
        logger.debug(f"Initial state keys: {list(initial_state.keys())}")
        
        # Ensure we have all required fields
        state = {
            "prompt": initial_state.get("prompt", ""),
            "project_id": initial_state.get("project_id", "-1"),
            "original_prompt": initial_state.get("original_prompt", initial_state.get("prompt", "")),
            "clarification_needed": False,
            "suggestions": [],
            "clarification_message": "",
            "messages": [{"role": "user", "content": initial_state.get("prompt", "")}]
        }
        
        logger.debug(f"Prepared state for workflow: {state}")
        
        try:
            # Run the graph
            logger.debug("Invoking graph with state")
            final_state = self.graph.invoke(state, {"configurable": {"thread_id": "1"}})
            logger.info("Workflow completed successfully")
            
            # Convert to regular dict for return
            return dict(final_state)
        except Exception as e:
            logger.exception("Error during workflow execution")
            # Return error state
            return {
                "error": "workflow_error",
                "message": str(e),
                "details": str(e.__class__.__name__)
            }