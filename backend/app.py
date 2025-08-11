import os
import warnings
from pathlib import Path
from typing import List, Optional

from config import config
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rag_system import RAGSystem

warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")

# Initialize logging
from logging_config import get_logger, setup_logging

setup_logging(log_level="INFO", log_file="logs/rag_system.log")
logger = get_logger(__name__)


# Initialize FastAPI app
app = FastAPI(title="Course Materials RAG System", root_path="")

# Add trusted host middleware for proxy
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

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
logger.info("Starting Course Materials RAG System")
rag_system = RAGSystem(config)

# Pydantic models for request/response


class QueryRequest(BaseModel):
    """Request model for course queries"""

    query: str
    session_id: Optional[str] = None


class Source(BaseModel):
    """Source information with optional link"""

    text: str
    link: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for course queries"""

    answer: str
    sources: List[Source]
    session_id: str


class CourseStats(BaseModel):
    """Response model for course statistics"""

    total_courses: int
    course_titles: List[str]


# API Endpoints


# New endpoint to create a new chat session


@app.post("/api/session/new")
async def create_new_session(prev_session_id: Optional[str] = None):
    """Create a new session and clear previous session if provided"""
    try:
        if prev_session_id:
            rag_system.session_manager.clear_session(prev_session_id)
        session_id = rag_system.session_manager.create_session()
        return JSONResponse(content={"session_id": session_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Process a query and return response with sources"""
    try:
        # Create session if not provided
        session_id = request.session_id
        if not session_id:
            session_id = rag_system.session_manager.create_session()

        # Process query using RAG system
        answer, sources = rag_system.query(request.query, session_id)

        # Convert sources to Source objects
        source_objects = []
        for source in sources:
            if isinstance(source, dict) and "text" in source:
                # New format with embedded links
                source_objects.append(
                    Source(text=source["text"], link=source.get("link"))
                )
            else:
                # Legacy string format
                source_objects.append(Source(text=str(source), link=None))

        return QueryResponse(
            answer=answer, sources=source_objects, session_id=session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/courses", response_model=CourseStats)
async def get_course_stats():
    """Get course analytics and statistics"""
    try:
        analytics = rag_system.get_course_analytics()
        return CourseStats(
            total_courses=analytics["total_courses"],
            course_titles=analytics["course_titles"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """Load initial documents on startup"""
    docs_path = "../docs"
    if os.path.exists(docs_path):
        logger.info("Loading initial documents...")
        try:
            courses, chunks = rag_system.add_course_folder(
                docs_path, clear_existing=False
            )
            logger.info(f"Loaded {courses} courses with {chunks} chunks")
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
    else:
        logger.warning(f"Documents directory does not exist: {docs_path}")


# Custom static file handler with no-cache headers for development


class DevStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if isinstance(response, FileResponse):
            # Add no-cache headers for development
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response


# Serve static files for the frontend
app.mount("/", StaticFiles(directory="../frontend", html=True), name="static")
