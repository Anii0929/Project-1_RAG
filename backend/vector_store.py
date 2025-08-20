from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from logging_config import get_logger
from models import Document, DocumentChunk
from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)


@dataclass
class SearchResults:
    """Container for search results with metadata"""

    documents: List[str]
    metadata: List[Dict[str, Any]]
    distances: List[float]
    error: Optional[str] = None

    @classmethod
    def from_chroma(cls, chroma_results: Dict) -> "SearchResults":
        """Create SearchResults from ChromaDB query results"""
        return cls(
            documents=(
                chroma_results["documents"][0] if chroma_results["documents"] else []
            ),
            metadata=(
                chroma_results["metadatas"][0] if chroma_results["metadatas"] else []
            ),
            distances=(
                chroma_results["distances"][0] if chroma_results["distances"] else []
            ),
        )

    @classmethod
    def empty(cls, error_msg: str) -> "SearchResults":
        """Create empty results with error message"""
        return cls(documents=[], metadata=[], distances=[], error=error_msg)

    def is_empty(self) -> bool:
        """Check if results are empty"""
        return len(self.documents) == 0


class VectorStore:
    """Vector storage using ChromaDB for document content and metadata"""

    def __init__(self, chroma_path: str, embedding_model: str, max_results: int = 5):
        logger.info(
            f"Initializing VectorStore - path: {chroma_path}, model: {embedding_model}, max_results: {max_results}"
        )
        self.max_results = max_results

        # Initialize ChromaDB client
        logger.debug("Setting up ChromaDB client")
        self.client = chromadb.PersistentClient(
            path=chroma_path, settings=Settings(anonymized_telemetry=False)
        )

        # Set up sentence transformer embedding function
        logger.debug(f"Loading embedding model: {embedding_model}")
        self.embedding_function = (
            chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
        )

        # Create collections for different types of data
        logger.debug("Creating/accessing ChromaDB collections")
        self.document_catalog = self._create_collection(
            "document_catalog"
        )  # Document titles/instructors
        self.document_content = self._create_collection(
            "document_content"
        )  # Actual document material
        logger.info("VectorStore initialization complete")

    def _create_collection(self, name: str):
        """Create or get a ChromaDB collection"""
        return self.client.get_or_create_collection(
            name=name, embedding_function=self.embedding_function
        )

    def search(
        self,
        query: str,
        document_name: Optional[str] = None,
        section_number: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> SearchResults:
        """
        Main search interface that handles document resolution and content search.

        Args:
            query: What to search for in document content
            document_name: Optional document name/title to filter by
            section_number: Optional section number to filter by
            limit: Maximum results to return

        Returns:
            SearchResults object with documents and metadata
        """
        # Step 1: Resolve document name if provided
        document_title = None
        if document_name:
            document_title = self._resolve_document_name(document_name)
            if not document_title:
                return SearchResults.empty(f"No document found matching '{document_name}'")

        # Step 2: Build filter for content search
        filter_dict = self._build_filter(document_title, section_number)

        # Step 3: Search document content
        # Use provided limit or fall back to configured max_results, ensure it's at least 1
        search_limit = limit if limit is not None else self.max_results
        if search_limit <= 0:
            search_limit = 5  # Fallback to reasonable default

        try:
            results = self.document_content.query(
                query_texts=[query], n_results=search_limit, where=filter_dict
            )
            return SearchResults.from_chroma(results)
        except Exception as e:
            return SearchResults.empty(f"Search error: {str(e)}")

    def _resolve_document_name(self, document_name: str) -> Optional[str]:
        """Use vector search to find best matching document by name"""
        try:
            results = self.document_catalog.query(query_texts=[document_name], n_results=1)

            if results["documents"][0] and results["metadatas"][0]:
                # Return the title (which is now the ID)
                return results["metadatas"][0][0]["title"]
        except Exception as e:
            print(f"Error resolving document name: {e}")

        return None

    def _build_filter(
        self, document_title: Optional[str], section_number: Optional[int]
    ) -> Optional[Dict]:
        """Build ChromaDB filter from search parameters"""
        if not document_title and section_number is None:
            return None

        # Handle different filter combinations
        if document_title and section_number is not None:
            return {
                "$and": [
                    {"document_title": document_title},
                    {"section_number": section_number},
                ]
            }

        if document_title:
            return {"document_title": document_title}

        return {"section_number": section_number}

    def add_document_metadata(self, document: Document):
        """Add document information to the catalog for semantic search"""
        import json

        logger.info(f"Adding document metadata to catalog: '{document.title}'")
        document_text = document.title

        # Build sections metadata and serialize as JSON string
        sections_metadata = []
        for section in document.sections:
            sections_metadata.append(
                {
                    "section_number": section.section_number,
                    "section_title": section.title,
                    "section_link": section.section_link,
                }
            )

        sections_with_links = sum(1 for section in document.sections if section.section_link)
        logger.debug(
            f"Document '{document.title}': {len(document.sections)} sections, {sections_with_links} with links"
        )

        metadata = {
            "title": document.title,
            "instructor": document.instructor,
            "document_link": document.document_link,
            "sections_json": json.dumps(sections_metadata),  # Serialize as JSON string
            "section_count": len(document.sections),
        }

        logger.debug(f"Adding to document_catalog collection: id='{document.title}'")
        self.document_catalog.add(
            documents=[document_text], metadatas=[metadata], ids=[document.title]
        )
        logger.info(f"Successfully added document metadata for: '{document.title}'")

    def add_document_content(self, chunks: List[DocumentChunk]):
        """Add document content chunks to the vector store"""
        if not chunks:
            logger.debug("No chunks to add to document content")
            return

        logger.info(f"Adding {len(chunks)} content chunks to vector store")
        document_title = chunks[0].document_title if chunks else "Unknown"
        logger.debug(f"Document: '{document_title}' - Processing {len(chunks)} chunks")

        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "document_title": chunk.document_title,
                "section_number": chunk.section_number,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in chunks
        ]
        # Use title with chunk index for unique IDs
        ids = [
            f"{chunk.document_title.replace(' ', '_')}_{chunk.chunk_index}"
            for chunk in chunks
        ]

        logger.debug(
            f"Adding chunks to document_content collection with IDs: {ids[0]}...{ids[-1]}"
        )
        self.document_content.add(documents=documents, metadatas=metadatas, ids=ids)
        logger.info(
            f"Successfully added {len(chunks)} chunks for document: '{document_title}'"
        )

    def clear_all_data(self):
        """Clear all data from both collections"""
        logger.warning("Clearing all data from vector store collections")
        try:
            logger.debug("Deleting document_catalog collection")
            self.client.delete_collection("document_catalog")
            logger.debug("Deleting document_content collection")
            self.client.delete_collection("document_content")
            # Recreate collections
            logger.debug("Recreating collections")
            self.document_catalog = self._create_collection("document_catalog")
            self.document_content = self._create_collection("document_content")
            logger.info("Successfully cleared and recreated all collections")
        except Exception as e:
            logger.error(f"Error clearing data: {e}")
            raise

    def get_existing_document_titles(self) -> List[str]:
        """Get all existing document titles from the vector store"""
        try:
            # Get all documents from the catalog
            results = self.document_catalog.get()
            if results and "ids" in results:
                return results["ids"]
            return []
        except Exception as e:
            print(f"Error getting existing document titles: {e}")
            return []

    def get_document_count(self) -> int:
        """Get the total number of documents in the vector store"""
        try:
            results = self.document_catalog.get()
            if results and "ids" in results:
                return len(results["ids"])
            return 0
        except Exception as e:
            print(f"Error getting document count: {e}")
            return 0

    def get_all_documents_metadata(self) -> List[Dict[str, Any]]:
        """Get metadata for all documents in the vector store"""
        import json

        try:
            results = self.document_catalog.get()
            if results and "metadatas" in results:
                # Parse sections JSON for each document
                parsed_metadata = []
                for metadata in results["metadatas"]:
                    document_meta = metadata.copy()
                    if "sections_json" in document_meta:
                        document_meta["sections"] = json.loads(document_meta["sections_json"])
                        del document_meta[
                            "sections_json"
                        ]  # Remove the JSON string version
                    parsed_metadata.append(document_meta)
                return parsed_metadata
            return []
        except Exception as e:
            print(f"Error getting documents metadata: {e}")
            return []

    def get_document_link(self, document_title: str) -> Optional[str]:
        """Get document link for a given document title"""
        try:
            # Get document by ID (title is the ID)
            results = self.document_catalog.get(ids=[document_title])
            if results and "metadatas" in results and results["metadatas"]:
                metadata = results["metadatas"][0]
                return metadata.get("document_link")
            return None
        except Exception as e:
            print(f"Error getting document link: {e}")
            return None

    def get_section_link(self, document_title: str, section_number: int) -> Optional[str]:
        """Get section link for a given document title and section number"""
        import json

        try:
            # Get document by ID (title is the ID)
            results = self.document_catalog.get(ids=[document_title])
            if results and "metadatas" in results and results["metadatas"]:
                metadata = results["metadatas"][0]
                sections_json = metadata.get("sections_json")
                if sections_json:
                    sections = json.loads(sections_json)
                    # Find the section with matching number
                    for section in sections:
                        if section.get("section_number") == section_number:
                            return section.get("section_link")
            return None
        except Exception as e:
            print(f"Error getting section link: {e}")
