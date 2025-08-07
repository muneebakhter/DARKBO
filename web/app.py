import os
import numpy as np
import json
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, make_response

# Import our modules
from config.settings import Settings
from data.project_loader import ProjectLoader
from data.vector_store import VectorStoreManager
from rag.rag_chain import RAGChain
from graph.workflow import RAGWorkflow
import logging
from tools.datetools import get_date_tools

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("app")

# Initialize settings
settings = Settings()
settings.validate()

# Initialize components
project_loader = ProjectLoader(settings)
vector_store_manager = VectorStoreManager(settings, project_loader)

# Create RAG chain factory
def rag_chain_factory(project_name):
    return RAGChain(settings, project_name)

# Initialize workflow
date_tools = get_date_tools()
workflow = RAGWorkflow(
    project_loader,
    vector_store_manager,
    rag_chain_factory,
    date_tools
)

# Create Flask app
app = Flask(__name__)

@app.route("/")
def index():
    """Render the main page with project list"""
    return render_template("index.html", projects=project_loader.get_all_projects())

@app.route("/query", methods=["POST"])
def query():
    logger.debug("Received query request")
    try:
        # Get JSON data from request
        data = request.get_json(force=True)
        logger.debug(f"Request data: {data}")
        
        if not data:
            logger.error("No JSON data provided in request")
            return jsonify({"error": "Invalid request", "message": "No JSON data provided"}), 400
        
        # Extract prompt and project_id
        prompt = data.get("prompt", "").strip()
        project_id = str(data.get("project_id", "-1")).strip()
        
        logger.debug(f"Processing prompt: '{prompt}' with project_id: '{project_id}'")
        
        if not prompt:
            logger.error("Prompt is empty")
            return jsonify({"error": "Invalid request", "message": "Prompt cannot be empty"}), 400
        
        # Prepare initial state
        initial_state = {
            "prompt": prompt,
            "project_id": project_id if project_id != "-1" else "-1",
            "original_prompt": prompt
        }
        logger.debug(f"Initial state for workflow: {initial_state}")
        
        # Run the workflow
        result = workflow.run(initial_state)
        logger.debug(f"Workflow result: {result}")
        
        # Format response based on workflow result
        if result.get("error"):
            logger.error(f"Workflow error: {result['error']} - {result.get('message', '')}")
            return jsonify({
                "error": "Workflow error",
                "message": result.get("message", "An error occurred in the workflow")
            }), 500
        
        if result.get("clarification_needed"):
            logger.info("Request needs project clarification")
            return jsonify({
                "clarification": result.get("clarification_message", "Please select a project."),
                "suggestions": result.get("suggestions", []),
                "originalPrompt": prompt
            })
        elif result.get("answer"):
            logger.info("Generated answer for query")
            return jsonify({"answer": result["answer"]})
        else:
            logger.error("No response generated from workflow")
            return jsonify({"error": "No response", "message": "The system couldn't generate a response."}), 500
            
    except Exception as e:
        # Log the error for debugging
        logger.exception(f"Error in /query: {str(e)}")
        
        # Return a proper JSON error response
        return jsonify({
            "error": "Server error", 
            "message": str(e),
            "details": str(e.__class__.__name__)
        }), 500

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors with JSON response"""
    return jsonify({"error": "Not found", "message": "The requested resource was not found"}), 404

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors with JSON response"""
    app.logger.error(f"Server error: {str(error)}", exc_info=True)
    return jsonify({"error": "Server error", "message": "An internal server error occurred"}), 500

if __name__ == "__main__":
    # Print startup message
    print("\n" + "="*50)
    print("KB-AI System Starting...")
    print(f"Available projects: {len(project_loader.projects)}")
    print(f"Vector store directory: {settings.VECTOR_STORE_DIR}")
    print("="*50 + "\n")
    
    # Run the app
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
        debug=settings.DEBUG
    )