import warnings
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Union
import os

from config import config
from rag_system import RAGSystem

# Initialize FastAPI app
app = FastAPI(title="Course Materials RAG System", root_path="")

# Add trusted host middleware for proxy
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# Enable CORS with proper settings for proxy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize RAG system
rag_system = RAGSystem(config)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for course queries"""
    query: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    """Response model for course queries"""
    answer: str
    sources: List[Dict[str, Optional[str]]]
    session_id: str

class CourseStats(BaseModel):
    """Response model for course statistics"""
    total_courses: int
    course_titles: List[str]

# API Endpoints

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Process a user query using the RAG system and return response with sources.
    
    Args:
        request: QueryRequest containing the user's query and optional session_id.
        
    Returns:
        QueryResponse: Contains the AI-generated answer, source references,
            and session_id for conversation continuity.
            
    Raises:
        HTTPException: If query processing fails, with appropriate error message.
            Status 500 for internal errors, with user-friendly messages for
            common issues like API credit problems or authentication failures.
    """
    try:
        # Create session if not provided
        session_id = request.session_id
        if not session_id:
            session_id = rag_system.session_manager.create_session()
        
        # Process query using RAG system
        answer, sources = rag_system.query(request.query, session_id)
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            session_id=session_id
        )
    except Exception as e:
        print(f"ERROR in query_documents: {e}")
        import traceback
        traceback.print_exc()
        
        # Handle specific API errors with better user messages
        error_message = str(e)
        if "credit balance is too low" in error_message:
            error_message = "API credit balance is too low. Please add credits to your Anthropic account."
        elif "authentication" in error_message.lower():
            error_message = "Invalid API key. Please check your ANTHROPIC_API_KEY in the .env file."
        
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/api/courses", response_model=CourseStats)
async def get_course_stats():
    """Retrieve course catalog statistics and metadata.
    
    Returns:
        CourseStats: Contains total number of courses and list of all course titles.
        
    Raises:
        HTTPException: If statistics retrieval fails (status 500).
    """
    try:
        analytics = rag_system.get_course_analytics()
        return CourseStats(
            total_courses=analytics["total_courses"],
            course_titles=analytics["course_titles"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Load course documents from the docs folder during application startup.
    
    Automatically processes and indexes all course documents found in the
    ../docs directory. Skips files that have already been processed to
    avoid duplicate indexing.
    
    Side Effects:
        - Populates the vector store with course content
        - Prints loading status to console
        - Handles errors gracefully without stopping app startup
    """
    docs_path = "../docs"
    if os.path.exists(docs_path):
        print("Loading initial documents...")
        try:
            courses, chunks = rag_system.add_course_folder(docs_path, clear_existing=False)
            print(f"Loaded {courses} courses with {chunks} chunks")
        except Exception as e:
            print(f"Error loading documents: {e}")

# Custom static file handler with no-cache headers for development
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from pathlib import Path


class DevStaticFiles(StaticFiles):
    """Custom static file handler that adds no-cache headers for development.
    
    Prevents browser caching of static files during development to ensure
    changes are immediately visible.
    """
    
    async def get_response(self, path: str, scope):
        """Override response handling to add no-cache headers.
        
        Args:
            path: Requested file path.
            scope: ASGI scope dictionary.
            
        Returns:
            Response object with no-cache headers added for FileResponse types.
        """
        response = await super().get_response(path, scope)
        if isinstance(response, FileResponse):
            # Add no-cache headers for development
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response
    
    
# Serve static files for the frontend
app.mount("/", StaticFiles(directory="../frontend", html=True), name="static")