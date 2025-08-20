import os
from typing import Dict, List, Optional, Tuple

from ai_generator import AIGenerator
from document_processor import DocumentProcessor
from logging_config import get_logger
from models import Document, DocumentChunk, Section
from search_tools import DocumentListTool, DocumentOutlineTool, DocumentSearchTool, ToolManager
from session_manager import SessionManager
from vector_store import VectorStore

logger = get_logger(__name__)


class RAGSystem:
    """Main orchestrator for the Retrieval-Augmented Generation system"""

    def __init__(self, config):
        logger.info("=== Initializing RAG System ===")
        self.config = config

        # Initialize core components
        logger.debug("Creating DocumentProcessor")
        self.document_processor = DocumentProcessor(
            config.CHUNK_SIZE, config.CHUNK_OVERLAP
        )

        logger.debug("Creating VectorStore")
        self.vector_store = VectorStore(
            config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
        )

        logger.debug("Creating AIGenerator")
        self.ai_generator = AIGenerator(
            config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL
        )

        logger.debug("Creating SessionManager")
        self.session_manager = SessionManager(config.MAX_HISTORY)

        # Initialize search tools
        logger.debug("Setting up search tools")
        self.tool_manager = ToolManager()
        self.search_tool = DocumentSearchTool(self.vector_store)
        self.outline_tool = DocumentOutlineTool(self.vector_store)
        self.list_tool = DocumentListTool(self.vector_store)
        self.tool_manager.register_tool(self.search_tool)
        self.tool_manager.register_tool(self.outline_tool)
        self.tool_manager.register_tool(self.list_tool)

        logger.info("RAG System initialization complete")

    def add_document(self, file_path: str) -> Tuple[Document, int]:
        """
        Add a single document to the knowledge base.

        Args:
            file_path: Path to the document

        Returns:
            Tuple of (Document object, number of chunks created)
        """
        logger.info(f"Adding document: {file_path}")
        try:
            # Process the document
            document, document_chunks = self.document_processor.process_document(
                file_path
            )

            if document:
                # Add document metadata to vector store for semantic search
                self.vector_store.add_document_metadata(document)

                # Add document content chunks to vector store
                self.vector_store.add_document_content(document_chunks)

                logger.info(
                    f"Successfully added document: '{document.title}' with {len(document_chunks)} chunks"
                )
                return document, len(document_chunks)
            else:
                logger.warning(f"No document object created from document: {file_path}")
                return None, 0

        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            return None, 0

    def add_document_folder(
        self, folder_path: str, clear_existing: bool = False
    ) -> Tuple[int, int]:
        """
        Add all documents from a folder.

        Args:
            folder_path: Path to folder containing documents
            clear_existing: Whether to clear existing data first

        Returns:
            Tuple of (total documents added, total chunks created)
        """
        logger.info(
            f"=== Processing document folder: {folder_path} (clear_existing={clear_existing}) ==="
        )
        total_documents = 0
        total_chunks = 0

        # Clear existing data if requested
        if clear_existing:
            logger.info("Clearing existing data for fresh rebuild...")
            self.vector_store.clear_all_data()

        if not os.path.exists(folder_path):
            logger.error(f"Folder {folder_path} does not exist")
            return 0, 0

        # Get existing document titles to avoid re-processing
        existing_document_titles = set(self.vector_store.get_existing_document_titles())
        logger.info(
            f"Found {len(existing_document_titles)} existing documents in vector store"
        )

        # Get all files to process
        files = [
            f
            for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
            and f.lower().endswith((".pdf", ".docx", ".txt"))
        ]
        logger.info(f"Found {len(files)} document files to process")

        # Process each file in the folder
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            logger.debug(f"Processing file: {file_name}")
            try:
                # Check if this document might already exist
                # We'll process the document to get the document ID, but only add if new
                document, document_chunks = self.document_processor.process_document(
                    file_path
                )

                if document and document.title not in existing_document_titles:
                    # This is a new document - add it to the vector store
                    logger.info(f"Adding new document to vector store: '{document.title}'")
                    self.vector_store.add_document_metadata(document)
                    self.vector_store.add_document_content(document_chunks)
                    total_documents += 1
                    total_chunks += len(document_chunks)
                    logger.info(
                        f"Added new document: {document.title} ({len(document_chunks)} chunks)"
                    )
                    existing_document_titles.add(document.title)
                elif document:
                    logger.info(f"Document already exists: {document.title} - skipping")
                else:
                    logger.warning(
                        f"Failed to create document object from file: {file_name}"
                    )
            except Exception as e:
                logger.error(f"Error processing {file_name}: {e}")

        logger.info(
            f"=== Folder processing complete: {total_documents} new documents, {total_chunks} total chunks ==="
        )
        return total_documents, total_chunks

    def query(
        self, query: str, session_id: Optional[str] = None
    ) -> Tuple[str, List[str]]:
        """
        Process a user query using the RAG system with tool-based search.

        Args:
            query: User's question
            session_id: Optional session ID for conversation context

        Returns:
            Tuple of (response, sources list - empty for tool-based approach)
        """
        # Create prompt for the AI with clear instructions
        prompt = f"""Answer this question about document materials: {query}"""

        # Get conversation history if session exists
        history = None
        if session_id:
            history = self.session_manager.get_conversation_history(session_id)

        # Generate response using AI with tools
        response = self.ai_generator.generate_response(
            query=prompt,
            conversation_history=history,
            tools=self.tool_manager.get_tool_definitions(),
            tool_manager=self.tool_manager,
        )

        # Get sources from the search tool
        sources = self.tool_manager.get_last_sources()

        # Reset sources after retrieving them
        self.tool_manager.reset_sources()

        # Update conversation history
        if session_id:
            self.session_manager.add_exchange(session_id, query, response)

        # Return response with sources from tool searches
        return response, sources

    def get_document_analytics(self) -> Dict:
        """Get analytics about the document catalog"""
        return {
            "total_documents": self.vector_store.get_document_count(),
            "document_titles": self.vector_store.get_existing_document_titles(),
        }
