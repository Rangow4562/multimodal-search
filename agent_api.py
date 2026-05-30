"""
Agent Workflow API Server - Multimodal RAG System
Replaces Chainlit interface with FastAPI REST endpoints
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import asyncio
import logging
import base64
import mimetypes
from datetime import datetime
from contextlib import asynccontextmanager
from io import BytesIO
import tempfile

# Agent system imports (reusing the agent framework)
from agent_ather_multimodel import (
    AgentRole,
    AgentMessage,
    AgentContext,
    BaseAgent,
    QueryProcessingAgent,
    ImageProcessingAgent,
    ContentAnalysisAgent,
    ResponseOrchestrationAgent,
    AgentWorkflowManager,
    initialize_services,
    setup_agent_workflow,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Lifespan event handler (modern approach replacing deprecated on_event)
@asynccontextmanager
async def lifespan(app):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Starting Agent Workflow API Server...")
    os.makedirs("./documents", exist_ok=True)
    await initialize_agent_system()
    yield
    # Shutdown
    logger.info("Shutting down Agent Workflow API Server...")


# Initialize FastAPI app with lifespan handler
app = FastAPI(
    title="Agent Workflow Multimodal RAG API",
    description="Advanced agent workflow system for multimodal document search and Q&A",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for agent system
workflow_manager = None
services_initialized = False


# Pydantic models for API requests/responses
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    max_images: Optional[int] = 10
    text_only: Optional[bool] = False
    external_search: Optional[bool] = False


class QueryResponse(BaseModel):
    response_id: str
    session_id: str
    segments: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    processing_time: float


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str
    services: Dict[str, str]
    timestamp: datetime


class SessionInfo(BaseModel):
    session_id: str
    created_at: datetime
    query_count: int
    last_activity: datetime


# Session management
sessions: Dict[str, Dict[str, Any]] = {}


async def initialize_agent_system():
    """Initialize the agent workflow system."""
    global workflow_manager, services_initialized

    if services_initialized:
        return True

    try:
        logger.info("Initializing agent workflow system...")

        # Initialize services
        index, multi_modal_llm, search_client = initialize_services()

        # Set up agent workflow
        workflow_manager = setup_agent_workflow(index, multi_modal_llm, search_client)

        services_initialized = True
        logger.info("Agent workflow system initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize agent system: {str(e)}")
        return False


def create_session(session_id: Optional[str] = None) -> str:
    """Create or retrieve a session."""
    if not session_id:
        session_id = (
            f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(sessions)}"
        )

    if session_id not in sessions:
        sessions[session_id] = {
            "created_at": datetime.now(),
            "query_count": 0,
            "last_activity": datetime.now(),
            "context": AgentContext(user_query=""),
        }

    return session_id


def update_session_activity(session_id: str):
    """Update session activity timestamp."""
    if session_id in sessions:
        sessions[session_id]["last_activity"] = datetime.now()
        sessions[session_id]["query_count"] += 1


# API Endpoints


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        services = {
            "agent_system": "ready" if services_initialized else "initializing",
            "storage": "ok" if os.path.exists("./documents") else "warning",
        }

        return HealthResponse(
            status="healthy", services=services, timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/initialize")
async def initialize_system():
    """Initialize the agent system."""
    success = await initialize_agent_system()
    if success:
        return {"status": "success", "message": "Agent system initialized"}
    else:
        raise HTTPException(status_code=500, detail="Failed to initialize agent system")


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a user query through the agent workflow."""
    if not services_initialized:
        await initialize_agent_system()
        if not services_initialized:
            raise HTTPException(status_code=503, detail="Agent system not available")

    try:
        start_time = datetime.now()

        # Create or retrieve session
        session_id = create_session(request.session_id)
        update_session_activity(session_id)

        # Create context for the query
        metadata = {"max_images": request.max_images}
        if request.text_only:
            metadata["text_only"] = True
            metadata["max_images"] = 0  # Override max_images when text_only is True
        if request.external_search:
            metadata["external_search"] = True

        context = AgentContext(user_query=request.query, metadata=metadata)

        # Execute agent workflow
        search_mode = "text-only" if request.text_only else "multimodal"
        if request.external_search:
            search_mode += " + external"
        logger.info(f"Processing query ({search_mode}): {request.query[:100]}...")
        processed_context = await workflow_manager.execute_workflow(context)

        # Prepare response segments
        segments = []
        for i, segment_data in enumerate(processed_context.response_segments):
            if isinstance(segment_data, dict):
                # Convert images to base64 for JSON response
                segment = {
                    "id": f"segment_{i}",
                    "text": segment_data.get("text", ""),
                    "images": [],
                }

                # Process images
                for img_element in segment_data.get("images", []):
                    # Handle both dict-based and object-based image elements
                    if isinstance(img_element, dict):
                        img_data = {
                            "name": img_element.get("name", f"Image_{i}"),
                            "display": img_element.get("display", "inline"),
                        }
                        # Handle image content (URL or base64) for dict elements
                        if img_element.get("url"):
                            img_data["url"] = img_element["url"]
                        elif img_element.get("path"):
                            img_data["url"] = img_element["path"]
                    else:
                        # Legacy support for object-based elements
                        img_data = {
                            "name": getattr(img_element, "name", f"Image_{i}"),
                            "display": getattr(img_element, "display", "inline"),
                        }
                        # Handle image content (URL or base64) for objects
                        if hasattr(img_element, "url") and img_element.url:
                            img_data["url"] = img_element.url
                        elif hasattr(img_element, "path") and img_element.path:
                            # Convert local image to base64
                            try:
                                with open(img_element.path, "rb") as img_file:
                                    img_content = base64.b64encode(
                                        img_file.read()
                                    ).decode("utf-8")
                                    mime_type, _ = mimetypes.guess_type(
                                        img_element.path
                                    )
                                    img_data["base64"] = (
                                        f"data:{mime_type};base64,{img_content}"
                                    )
                                    img_data["filename"] = os.path.basename(
                                        img_element.path
                                    )
                            except Exception as e:
                                logger.error(f"Failed to encode image: {str(e)}")
                                continue

                    segment["images"].append(img_data)

                segments.append(segment)
            else:
                # Legacy format
                segments.append(
                    {"id": f"segment_{i}", "text": str(segment_data), "images": []}
                )

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()

        response = QueryResponse(
            response_id=f"resp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            session_id=session_id,
            segments=segments,
            metadata={
                "query_processed": processed_context.metadata.get(
                    "query_processed", False
                ),
                "retrieval_count": processed_context.metadata.get("retrieval_count", 0),
                "image_count": processed_context.metadata.get("image_count", 0),
                "segment_count": len(segments),
                "processing_time": processing_time,
                "text_only": request.text_only,
                "external_search": request.external_search,
                "search_mode": search_mode,
            },
            processing_time=processing_time,
        )

        # Store response in session context
        sessions[session_id]["context"] = processed_context

        return response

    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Query processing failed: {str(e)}"
        )


@app.get("/session/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str):
    """Get session information."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = sessions[session_id]
    return SessionInfo(
        session_id=session_id,
        created_at=session_data["created_at"],
        query_count=session_data["query_count"],
        last_activity=session_data["last_activity"],
    )


@app.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    return {
        "sessions": [
            {
                "session_id": session_id,
                "created_at": data["created_at"].isoformat(),
                "query_count": data["query_count"],
                "last_activity": data["last_activity"].isoformat(),
            }
            for session_id, data in sessions.items()
        ],
        "total_sessions": len(sessions),
    }


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    del sessions[session_id]
    return {"message": f"Session {session_id} deleted successfully"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document for processing."""
    try:
        # Create documents directory if it doesn't exist
        os.makedirs("./documents", exist_ok=True)

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join("./documents", filename)

        # Save uploaded file
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"Document uploaded: {filename}")

        return DocumentUploadResponse(
            document_id=f"doc_{timestamp}",
            filename=filename,
            status="uploaded",
            message=f"Document {file.filename} uploaded successfully",
        )

    except Exception as e:
        logger.error(f"Document upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/documents")
async def list_documents():
    """List uploaded documents."""
    try:
        documents_dir = "./documents"
        if not os.path.exists(documents_dir):
            return {"documents": [], "total_documents": 0}

        documents = []
        for filename in os.listdir(documents_dir):
            file_path = os.path.join(documents_dir, filename)
            if os.path.isfile(file_path):
                stat = os.stat(file_path)
                documents.append(
                    {
                        "filename": filename,
                        "size": stat.st_size,
                        "uploaded_at": datetime.fromtimestamp(
                            stat.st_ctime
                        ).isoformat(),
                        "type": mimetypes.guess_type(filename)[0] or "unknown",
                    }
                )

        return {"documents": documents, "total_documents": len(documents)}

    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list documents: {str(e)}"
        )


@app.get("/agent/status")
async def get_agent_status():
    """Get detailed agent system status."""
    try:
        if not services_initialized:
            return {
                "status": "not_initialized",
                "agents": [],
                "message": "Agent system not initialized",
            }

        agent_status = []
        for agent_name, agent in workflow_manager.agents.items():
            agent_status.append(
                {
                    "name": agent_name,
                    "role": agent.role.value,
                    "type": type(agent).__name__,
                }
            )

        return {
            "status": "initialized",
            "agents": agent_status,
            "workflow_steps": len(workflow_manager.workflow_steps),
            "active_sessions": len(sessions),
        }

    except Exception as e:
        logger.error(f"Failed to get agent status: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get agent status: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Agent Workflow Multimodal RAG API",
        "version": "1.0.0",
        "description": "Advanced agent workflow system for multimodal document search and Q&A",
        "endpoints": {
            "health": "/health",
            "initialize": "/initialize",
            "query": "/query",
            "sessions": "/sessions",
            "upload": "/upload",
            "documents": "/documents",
            "agent_status": "/agent/status",
        },
        "query_options": {
            "text_only": "Set to true to search text content only (no images)",
            "external_search": "Set to true to search outside the knowledge base using LLM general knowledge",
            "max_images": "Maximum number of images to include in response (ignored if text_only=true)",
        },
        "documentation": "/docs",
    }


@app.get("/ws/{path:path}")
async def websocket_catchall(path: str):
    """Handle websocket requests that were used by the old Chainlit interface."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "WebSocket endpoint not available",
            "message": "This API uses HTTP REST endpoints instead of WebSockets",
            "alternative": "Use the /query endpoint for processing queries",
            "documentation": "/docs",
        },
    )


@app.api_route("/socket.io/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def socketio_catchall(path: str):
    """Handle socket.io requests from the old Chainlit interface."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Socket.IO endpoint not available",
            "message": "This API uses HTTP REST endpoints instead of Socket.IO",
            "alternative": "Use the /query endpoint for processing queries",
            "documentation": "/docs",
        },
    )


# Run the server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "agent_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
