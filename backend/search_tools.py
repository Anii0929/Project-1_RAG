from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol

from vector_store import SearchResults, VectorStore


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


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""

    def __init__(self, vector_store: VectorStore):
        """Initialize the course search tool with a vector store.

        Args:
            vector_store: VectorStore instance for performing semantic searches.
        """
        self.store = vector_store
        self.last_sources = []  # Track sources from last search

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic-compatible tool definition for course content search.

        Returns:
            Dict[str, Any]: Tool definition with name, description, and input schema
                for use with Anthropic's tool calling API.
        """
        return {
            "name": "search_course_content",
            "description": "Search WITHIN course content for specific topics, concepts, or information. Use this for detailed questions about course content, not for course structure/outline requests",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in the course content",
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')",
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "Specific lesson number to search within (e.g. 1, 2, 3)",
                    },
                },
                "required": ["query"],
            },
        }

    def execute(
        self,
        query: str,
        course_name: Optional[str] = None,
        lesson_number: Optional[int] = None,
    ) -> str:
        """
        Execute the search tool with given parameters.

        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter

        Returns:
            Formatted search results or error message
        """

        # Use the vector store's unified search interface
        results = self.store.search(
            query=query, course_name=course_name, lesson_number=lesson_number
        )

        # Handle errors
        if results.error:
            return results.error

        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."

        # Format and return results
        return self._format_results(results)

    def _format_results(self, results: SearchResults) -> str:
        """Format search results with course and lesson context headers.

        Creates readable output with course/lesson attribution and tracks
        source information for the UI.

        Args:
            results: SearchResults object containing documents and metadata.

        Returns:
            str: Formatted results with context headers and content.

        Side Effects:
            Updates self.last_sources with source information for UI display.
        """
        formatted = []
        sources = []  # Track sources for the UI

        for doc, meta in zip(results.documents, results.metadata):
            course_title = meta.get("course_title", "unknown")
            lesson_num = meta.get("lesson_number")

            # Build context header
            header = f"[{course_title}"
            if lesson_num is not None:
                header += f" - Lesson {lesson_num}"
            header += "]"

            # Build source for the UI with link information
            source_text = course_title
            if lesson_num is not None:
                source_text += f" - Lesson {lesson_num}"

            # Try to get lesson link if we have lesson number
            lesson_link = None
            if lesson_num is not None and course_title != "unknown":
                lesson_link = self.store.get_lesson_link(course_title, lesson_num)

            # Create source object with text and optional link
            source_obj = {"text": source_text, "link": lesson_link}
            sources.append(source_obj)

            formatted.append(f"{header}\n{doc}")

        # Store sources for retrieval
        self.last_sources = sources

        return "\n\n".join(formatted)


class CourseOutlineTool(Tool):
    """Tool for retrieving course outline information"""

    def __init__(self, vector_store: VectorStore):
        """Initialize the course outline tool with a vector store.

        Args:
            vector_store: VectorStore instance for retrieving course metadata.
        """
        self.store = vector_store
        self.last_sources = []

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic-compatible tool definition for course outline retrieval.

        Returns:
            Dict[str, Any]: Tool definition with name, description, and input schema
                for retrieving complete course structure and lesson lists.
        """
        return {
            "name": "get_course_outline",
            "description": "Get the COMPLETE STRUCTURE and OUTLINE of a course including title, instructor, course link, and full lesson list with titles and links. Use this for queries about course overview, structure, syllabus, what lessons are included, or when user asks 'what is the outline of [course]'",
            "input_schema": {
                "type": "object",
                "properties": {
                    "course_name": {
                        "type": "string",
                        "description": "Course title or partial title to get outline for (e.g. 'MCP', 'Introduction')",
                    }
                },
                "required": ["course_name"],
            },
        }

    def execute(self, course_name: str = None) -> str:
        """
        Execute the outline tool to get course structure.

        Args:
            course_name: Course name to get outline for

        Returns:
            Formatted course outline or error message
        """
        # Handle missing course_name parameter gracefully
        if not course_name:
            # Fallback: suggest using list_all_courses or provide course options
            titles = self.store.get_existing_course_titles()
            if not titles:
                return "No courses available. Please add some course documents first."
            return f"Please specify which course outline you want. Available courses: {', '.join(titles[:5])}. Or ask 'what courses are available?' to see all courses with details."
        # Use semantic search to find matching course
        course_title = self.store._resolve_course_name(course_name)
        if not course_title:
            return f"No course found matching '{course_name}'"
        print(
            f"Executing CourseOutlineTool for course: {course_name} -> resolved to: {course_title}"
        )
        # Get course metadata
        try:
            results = self.store.course_catalog.get(ids=[course_title])
            if not results or not results["metadatas"] or not results["metadatas"][0]:
                return f"Course metadata not found for '{course_title}'"

            metadata = results["metadatas"][0]

            # Extract course information
            title = metadata.get("title", "Unknown Title")
            instructor = metadata.get("instructor", "Unknown Instructor")
            course_link = metadata.get("course_link")
            lessons_json = metadata.get("lessons_json", "[]")

            # Parse lessons
            import json

            try:
                lessons = json.loads(lessons_json)
            except (json.JSONDecodeError, TypeError):
                lessons = []

            # Format the outline with better structure
            outline = f"# {title}"
            if course_link:
                outline += f"\n**Course Link:** {course_link}"
            if instructor:
                outline += f"\n**Instructor:** {instructor}"

            outline += f"\n\n## Course Outline ({len(lessons)} lessons)\n"

            if lessons:
                for lesson in lessons:
                    lesson_num = lesson.get("lesson_number", "?")
                    lesson_title = lesson.get("lesson_title", "Untitled")
                    lesson_link = lesson.get("lesson_link")

                    outline += f"**Lesson {lesson_num}:** {lesson_title}"
                    if lesson_link:
                        outline += f" - [View Lesson]({lesson_link})"
                    outline += "\n\n"
            else:
                outline += "No lessons found"

            # Track source for UI
            source_obj = {"text": f"{title} - Course Outline", "link": course_link}
            self.last_sources = [source_obj]
            print(f"CourseOutlineTool's outline:\n{outline}")

            return outline

        except Exception as e:
            return f"Error retrieving course outline: {str(e)}"


class CourseListTool(Tool):
    """Tool for listing all available courses"""

    def __init__(self, vector_store: VectorStore):
        """Initialize the course list tool with a vector store.

        Args:
            vector_store: VectorStore instance for retrieving course titles.
        """
        self.store = vector_store
        self.last_sources = []

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic-compatible tool definition for listing all courses.

        Returns:
            Dict[str, Any]: Tool definition with name, description, and input schema
                for listing all available courses in the system.
        """
        return {
            "name": "list_all_courses",
            "description": "List ALL available courses in the system. Use this when user asks 'what courses are available', 'show me all courses', 'list courses', or similar general course listing queries. Does not require any parameters.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }

    def execute(self) -> str:
        """Execute the course listing tool.

        Returns:
            Formatted list of all available courses.
        """
        try:
            course_titles = self.store.get_existing_course_titles()

            if not course_titles:
                return "No courses are currently available in the system."

            # Get course metadata for better display
            all_metadata = self.store.get_all_courses_metadata()

            courses_info = []
            for metadata in all_metadata:
                title = metadata.get("title", "Unknown Course")
                instructor = metadata.get("instructor", "Unknown Instructor")
                lesson_count = metadata.get("lesson_count", 0)
                course_link = metadata.get("course_link")

                course_info = f"• **{title}**"
                if instructor and instructor != "Unknown Instructor":
                    course_info += f" by {instructor}"
                if lesson_count > 0:
                    course_info += f" ({lesson_count} lessons)"
                if course_link:
                    course_info += f" - [Course Link]({course_link})"

                courses_info.append(course_info)

            # Track source for UI (general course listing)
            self.last_sources = [
                {"text": f"Course Catalog ({len(course_titles)} courses)", "link": None}
            ]

            return f"Available courses ({len(course_titles)}):\n\n" + "\n\n".join(
                courses_info
            )

        except Exception as e:
            return f"Error retrieving course list: {str(e)}"


class ToolManager:
    """Manages available tools for the AI"""

    def __init__(self):
        """Initialize the tool manager with empty tool registry."""
        self.tools = {}

    def register_tool(self, tool: Tool):
        """Register a tool that implements the Tool interface.

        Args:
            tool: Tool instance that implements get_tool_definition() and execute().

        Raises:
            ValueError: If tool definition is missing required 'name' field.
        """
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    def get_tool_definitions(self) -> list:
        """Get all registered tool definitions for Anthropic tool calling API.

        Returns:
            list: List of tool definition dictionaries compatible with
                Anthropic's tool calling format.
        """
        return [tool.get_tool_definition() for tool in self.tools.values()]

    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a registered tool by name with provided parameters.

        Args:
            tool_name: Name of the tool to execute.
            **kwargs: Parameters to pass to the tool's execute method.

        Returns:
            str: Tool execution result or error message.
        """
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"

        return self.tools[tool_name].execute(**kwargs)

    def get_last_sources(self) -> list:
        """Retrieve source information from the most recent tool execution.

        Checks all registered tools for source tracking and returns the most
        recent sources for UI display.

        Returns:
            list: List of source dictionaries with 'text' and optional 'link' keys.
        """
        # Check all tools for last_sources attribute
        for tool in self.tools.values():
            if hasattr(tool, "last_sources") and tool.last_sources:
                return tool.last_sources
        return []

    def reset_sources(self):
        """Clear source information from all tools that track sources.

        Called after retrieving sources to prevent stale source data
        from appearing in subsequent responses.
        """
        for tool in self.tools.values():
            if hasattr(tool, "last_sources"):
                tool.last_sources = []
