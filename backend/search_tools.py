from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol

from logging_config import get_logger
from vector_store import SearchResults, VectorStore

logger = get_logger(__name__)


class Tool(ABC):
    """Abstract base class for all tools"""

    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class DocumentSearchTool(Tool):
    """Tool for searching document content with semantic document name matching"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "search_document_content",
            "description": "Search document materials with smart document name matching and section filtering",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in the document content",
                    },
                    "document_name": {
                        "type": "string",
                        "description": "Document title (partial matches work, e.g. 'MCP', 'Introduction')",
                    },
                    "section_number": {
                        "type": "integer",
                        "description": "Specific section number to search within (e.g. 1, 2, 3)",
                    },
                },
                "required": ["query"],
            },
        }

    def execute(
        self,
        query: str,
        document_name: Optional[str] = None,
        section_number: Optional[int] = None,
    ) -> str:
        """
        Execute the search tool with given parameters.

        Args:
            query: What to search for
            document_name: Optional document filter
            section_number: Optional section filter

        Returns:
            Formatted search results or error message
        """

        # Use the vector store's unified search interface
        results = self.store.search(
            query=query, document_name=document_name, section_number=section_number
        )

        # Handle errors
        if results.error:
            return results.error

        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if document_name:
                filter_info += f" in document '{document_name}'"
            if section_number:
                filter_info += f" in section {section_number}"
            return f"No relevant content found{filter_info}."

        # Format and return results
        return self._format_results(results)

    def _format_results(self, results: SearchResults) -> str:
        """Format search results with document and lesson context"""
        formatted = []
        sources = []  # Track sources for the UI with embedded links

        for doc, meta in zip(results.documents, results.metadata):
            document_title = meta.get("document_title", "unknown")
            section_num = meta.get("section_number")

            # Build context header
            header = f"[{document_title}"
            if section_num is not None:
                header += f" - Section {section_num}"
            header += "]"

            # Get section link if available
            section_link = None
            if section_num is not None:
                section_link = self.store.get_section_link(document_title, section_num)
                logger.debug(
                    f"Retrieved section link for '{document_title}' Section {section_num}: {section_link}"
                )

            # Create source entry with embedded link information
            source_text = document_title
            if section_num is not None:
                source_text += f" - Section {section_num}"

            # Create source object with both display text and link
            source_entry = {"text": source_text, "link": section_link}
            sources.append(source_entry)

            formatted.append(f"{header}\n{doc}")

        # Store sources for retrieval
        self.last_sources = sources
        logger.info(
            f"Formatted {len(sources)} search results with embedded section links"
        )

        return "\n\n".join(formatted)


class DocumentOutlineTool(Tool):
    """Tool for retrieving document outlines with lesson information"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "get_document_outline",
            "description": "Get document outline including document title, link, and complete section list",
            "input_schema": {
                "type": "object",
                "properties": {
                    "document_title": {
                        "type": "string",
                        "description": "Document title (partial matches work, e.g. 'MCP', 'Introduction')",
                    }
                },
                "required": ["document_title"],
            },
        }

    def execute(self, document_title: str) -> str:
        """
        Execute the outline tool to get document information.

        Args:
            document_title: Document title to get outline for

        Returns:
            Formatted document outline or error message
        """
        # Use vector search to find the best matching document
        resolved_title = self.store._resolve_document_name(document_title)
        if not resolved_title:
            return f"No document found matching '{document_title}'"        
        

        # Get document metadata including lessons
        import json

        try:
            results = self.store.document_catalog.get(ids=[resolved_title])
            if not results or not results.get("metadatas"):
                return f"No document metadata found for '{resolved_title}'"        
        

            metadata = results["metadatas"][0]
            document_link = metadata.get("document_link", "No link available")
            sections_json = metadata.get("sections_json", "[]")
            sections = json.loads(sections_json)

            # Format the response
            outline = f"**Document:** {resolved_title}\n"
            outline += f"**Document Link:** {document_link}\n\n"
            outline += "**Sections:**\n"

            if not sections:
                outline += "No sections available"
            else:
                for section in sections:
                    section_num = section.get("section_number", "N/A")
                    section_title = section.get("section_title", "Untitled")
                    outline += f"{section_num}. {section_title}\n"

            return outline

        except Exception as e:
            return f"Error retrieving document outline: {str(e)}"


class DocumentListTool(Tool):
    """Tool for listing all available document titles"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "list_all_documents",
            "description": "Get a complete list of all available document titles in the knowledge base",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }

    def execute(self) -> str:
        """
        Execute the tool to list all document titles.

        Returns:
            Formatted list of all available document titles
        """
        try:
            document_titles = self.store.get_existing_document_titles()
            
            if not document_titles:
                return "No documents are currently available in the knowledge base."
            
            # Format as numbered list
            formatted_list = []
            for i, title in enumerate(document_titles, 1):
                formatted_list.append(f"{i}. {title}")
            
            return f"Available documents ({len(document_titles)} total):\n\n" + "\n".join(formatted_list)
            
        except Exception as e:
            return f"Error retrieving document list: {str(e)}"


class ToolManager:
    """Manages available tools for the AI"""

    def __init__(self):
        self.tools = {}

    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    def get_tool_definitions(self) -> list:
        """Get all tool definitions for Anthropic tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]

    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"

        return self.tools[tool_name].execute(**kwargs)

    def get_last_sources(self) -> list:
        """Get sources from the last search operation"""
        # Check all tools for last_sources attribute
        for tool in self.tools.values():
            if hasattr(tool, "last_sources") and tool.last_sources:
                return tool.last_sources
        return []

    def reset_sources(self):
        """Reset sources from all tools that track sources"""
        for tool in self.tools.values():
            if hasattr(tool, "last_sources"):
                tool.last_sources = []
