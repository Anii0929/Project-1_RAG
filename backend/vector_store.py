from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from models import Course, CourseChunk
from sentence_transformers import SentenceTransformer


@dataclass
class SearchResults:
    """Container for search results with metadata"""

    documents: List[str]
    metadata: List[Dict[str, Any]]
    distances: List[float]
    error: Optional[str] = None

    @classmethod
    def from_chroma(cls, chroma_results: Dict) -> "SearchResults":
        """Create SearchResults from ChromaDB query results.

        Extracts documents, metadata, and distances from ChromaDB's response format.

        Args:
            chroma_results: Raw results dictionary from ChromaDB query.

        Returns:
            SearchResults: Structured results object with extracted data.
        """
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
        """Create empty SearchResults with an error message.

        Args:
            error_msg: Error message describing why no results were found.

        Returns:
            SearchResults: Empty results object with error set.
        """
        return cls(documents=[], metadata=[], distances=[], error=error_msg)

    def is_empty(self) -> bool:
        """Check if the search results contain any documents.

        Returns:
            bool: True if no documents were found, False otherwise.
        """
        return len(self.documents) == 0


class VectorStore:
    """Vector storage using ChromaDB for course content and metadata"""

    def __init__(self, chroma_path: str, embedding_model: str, max_results: int = 5):
        """Initialize the vector store with ChromaDB and sentence transformers.

        Args:
            chroma_path: Path where ChromaDB will store its data.
            embedding_model: Name of the sentence transformer model for embeddings.
            max_results: Maximum number of search results to return by default.

        Raises:
            ImportError: If required dependencies (chromadb, sentence_transformers) are missing.
            ConnectionError: If ChromaDB cannot be initialized.
        """
        self.max_results = max_results
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=chroma_path, settings=Settings(anonymized_telemetry=False)
        )

        # Set up sentence transformer embedding function
        self.embedding_function = (
            chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
        )

        # Create collections for different types of data
        self.course_catalog = self._create_collection(
            "course_catalog"
        )  # Course titles/instructors
        self.course_content = self._create_collection(
            "course_content"
        )  # Actual course material

    def _create_collection(self, name: str):
        """Create or retrieve a ChromaDB collection with embedding function.

        Args:
            name: Name of the collection to create or retrieve.

        Returns:
            Collection object from ChromaDB.

        Raises:
            chromadb.errors.ChromaError: If collection creation fails.
        """
        return self.client.get_or_create_collection(
            name=name, embedding_function=self.embedding_function
        )

    def search(
        self,
        query: str,
        course_name: Optional[str] = None,
        lesson_number: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> SearchResults:
        """
        Main search interface that handles course resolution and content search.

        Args:
            query: What to search for in course content
            course_name: Optional course name/title to filter by
            lesson_number: Optional lesson number to filter by
            limit: Maximum results to return

        Returns:
            SearchResults object with documents and metadata
        """
        # Step 1: Resolve course name if provided
        course_title = None
        if course_name:
            course_title = self._resolve_course_name(course_name)
            if not course_title:
                return SearchResults.empty(f"No course found matching '{course_name}'")

        # Step 2: Build filter for content search
        filter_dict = self._build_filter(course_title, lesson_number)

        # Step 3: Search course content
        # Use provided limit or fall back to configured max_results
        search_limit = limit if limit is not None else self.max_results

        try:
            results = self.course_content.query(
                query_texts=[query], n_results=search_limit, where=filter_dict
            )
            return SearchResults.from_chroma(results)
        except Exception as e:
            return SearchResults.empty(f"Search error: {str(e)}")

    def _resolve_course_name(self, course_name: str) -> Optional[str]:
        """Use semantic vector search to find the best matching course by name.

        Performs fuzzy matching on course titles to handle partial names or typos.

        Args:
            course_name: Partial or complete course name to search for.

        Returns:
            Optional[str]: Full course title if found, None otherwise.

        Examples:
            >>> store._resolve_course_name("MCP")
            'MCP: Build Rich-Context AI Apps with Anthropic'
            >>> store._resolve_course_name("nonexistent")
            None
        """
        try:
            results = self.course_catalog.query(query_texts=[course_name], n_results=1)

            if results["documents"][0] and results["metadatas"][0]:
                # Return the title (which is now the ID)
                return results["metadatas"][0][0]["title"]
        except Exception as e:
            print(f"Error resolving course name: {e}")

        return None

    def _build_filter(
        self, course_title: Optional[str], lesson_number: Optional[int]
    ) -> Optional[Dict]:
        """Build ChromaDB metadata filter from search parameters.

        Creates appropriate filter dictionaries for ChromaDB queries based on
        course title and lesson number constraints.

        Args:
            course_title: Course title to filter by, if any.
            lesson_number: Lesson number to filter by, if any.

        Returns:
            Optional[Dict]: ChromaDB filter dictionary, None if no filters.

        Examples:
            >>> store._build_filter("Course A", 1)
            {'$and': [{'course_title': 'Course A'}, {'lesson_number': 1}]}
            >>> store._build_filter(None, None)
            None
        """
        if not course_title and lesson_number is None:
            return None

        # Handle different filter combinations
        if course_title and lesson_number is not None:
            return {
                "$and": [
                    {"course_title": course_title},
                    {"lesson_number": lesson_number},
                ]
            }

        if course_title:
            return {"course_title": course_title}

        return {"lesson_number": lesson_number}

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Remove None values from metadata to prevent ChromaDB errors.

        ChromaDB v1.0.15+ cannot handle None values in metadata fields.
        This method removes any keys with None values.

        Args:
            metadata: Dictionary that may contain None values.

        Returns:
            Dict with None values filtered out.
        """
        return {k: v for k, v in metadata.items() if v is not None}

    def add_course_metadata(self, course: Course):
        """Add course metadata to the catalog collection for semantic search.

        Stores course title, instructor, links, and lesson information in a format
        optimized for semantic search and metadata retrieval.

        Args:
            course: Course object containing all metadata to store.

        Raises:
            chromadb.errors.ChromaError: If data insertion fails.
            json.JSONEncodeError: If lesson metadata cannot be serialized.
        """
        import json

        course_text = course.title

        # Build lessons metadata and serialize as JSON string
        lessons_metadata = []
        for lesson in course.lessons:
            lessons_metadata.append(
                {
                    "lesson_number": lesson.lesson_number,
                    "lesson_title": lesson.title,
                    "lesson_link": lesson.lesson_link,
                }
            )

        # Create metadata and sanitize None values
        metadata = {
            "title": course.title,
            "instructor": course.instructor,
            "course_link": course.course_link,
            "lessons_json": json.dumps(lessons_metadata),  # Serialize as JSON string
            "lesson_count": len(course.lessons),
        }

        # Sanitize to remove None values
        metadata = self._sanitize_metadata(metadata)

        self.course_catalog.add(
            documents=[course_text], metadatas=[metadata], ids=[course.title]
        )

    def add_course_content(self, chunks: List[CourseChunk]):
        """Add course content chunks to the vector store for semantic search.

        Stores text chunks with their associated metadata (course, lesson, position)
        for efficient retrieval during question answering.

        Args:
            chunks: List of CourseChunk objects containing text and metadata.

        Raises:
            chromadb.errors.ChromaError: If chunk insertion fails.
            ValueError: If chunks list is empty or contains invalid data.
        """
        if not chunks:
            return

        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "course_title": chunk.course_title,
                "lesson_number": chunk.lesson_number,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in chunks
        ]
        # Use title with chunk index for unique IDs
        ids = [
            f"{chunk.course_title.replace(' ', '_')}_{chunk.chunk_index}"
            for chunk in chunks
        ]

        self.course_content.add(documents=documents, metadatas=metadatas, ids=ids)

    def clear_all_data(self):
        """Clear all data from both collections and recreate them.

        Useful for complete data refresh during development or maintenance.

        Raises:
            chromadb.errors.ChromaError: If collection deletion or recreation fails.
        """
        try:
            self.client.delete_collection("course_catalog")
            self.client.delete_collection("course_content")
            # Recreate collections
            self.course_catalog = self._create_collection("course_catalog")
            self.course_content = self._create_collection("course_content")
        except Exception as e:
            print(f"Error clearing data: {e}")

    def get_existing_course_titles(self) -> List[str]:
        """Retrieve all existing course titles from the vector store.

        Returns:
            List[str]: List of course titles currently stored in the catalog.

        Raises:
            chromadb.errors.ChromaError: If data retrieval fails.
        """
        try:
            # Get all documents from the catalog
            results = self.course_catalog.get()
            if results and "ids" in results:
                return results["ids"]
            return []
        except Exception as e:
            print(f"Error getting existing course titles: {e}")
            return []

    def get_course_count(self) -> int:
        """Get the total number of courses stored in the vector store.

        Returns:
            int: Total number of courses in the catalog.

        Raises:
            chromadb.errors.ChromaError: If data retrieval fails.
        """
        try:
            results = self.course_catalog.get()
            if results and "ids" in results:
                return len(results["ids"])
            return 0
        except Exception as e:
            print(f"Error getting course count: {e}")
            return 0

    def get_all_courses_metadata(self) -> List[Dict[str, Any]]:
        """Retrieve complete metadata for all courses in the vector store.

        Returns parsed lesson information and all course metadata.

        Returns:
            List[Dict[str, Any]]: List of course metadata dictionaries with
                parsed lesson information.

        Raises:
            chromadb.errors.ChromaError: If data retrieval fails.
            json.JSONDecodeError: If lesson metadata cannot be parsed.
        """
        import json

        try:
            results = self.course_catalog.get()
            if results and "metadatas" in results:
                # Parse lessons JSON for each course
                parsed_metadata = []
                for metadata in results["metadatas"]:
                    course_meta = metadata.copy()
                    if "lessons_json" in course_meta:
                        course_meta["lessons"] = json.loads(course_meta["lessons_json"])
                        del course_meta[
                            "lessons_json"
                        ]  # Remove the JSON string version
                    parsed_metadata.append(course_meta)
                return parsed_metadata
            return []
        except Exception as e:
            print(f"Error getting courses metadata: {e}")
            return []

    def get_course_link(self, course_title: str) -> Optional[str]:
        """Retrieve the main course link for a given course title.

        Args:
            course_title: Exact course title to look up.

        Returns:
            Optional[str]: Course link URL if available, None otherwise.

        Raises:
            chromadb.errors.ChromaError: If data retrieval fails.
        """
        try:
            # Get course by ID (title is the ID)
            results = self.course_catalog.get(ids=[course_title])
            if results and "metadatas" in results and results["metadatas"]:
                metadata = results["metadatas"][0]
                return metadata.get("course_link")
            return None
        except Exception as e:
            print(f"Error getting course link: {e}")
            return None

    def get_lesson_link(self, course_title: str, lesson_number: int) -> Optional[str]:
        """Retrieve the link for a specific lesson within a course.

        Args:
            course_title: Exact course title containing the lesson.
            lesson_number: Lesson number to find the link for.

        Returns:
            Optional[str]: Lesson link URL if available, None otherwise.

        Raises:
            chromadb.errors.ChromaError: If data retrieval fails.
            json.JSONDecodeError: If lesson metadata cannot be parsed.
        """
        import json

        try:
            # Get course by ID (title is the ID)
            results = self.course_catalog.get(ids=[course_title])
            if results and "metadatas" in results and results["metadatas"]:
                metadata = results["metadatas"][0]
                lessons_json = metadata.get("lessons_json")
                if lessons_json:
                    lessons = json.loads(lessons_json)
                    # Find the lesson with matching number
                    for lesson in lessons:
                        if lesson.get("lesson_number") == lesson_number:
                            return lesson.get("lesson_link")
            return None
        except Exception as e:
            print(f"Error getting lesson link: {e}")
            return None
