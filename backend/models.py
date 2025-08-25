from typing import Dict, List, Optional

from pydantic import BaseModel


class Lesson(BaseModel):
    """Represents a lesson within a course with its metadata.

    Used to structure course content and provide navigation links for users.
    Lessons are ordered sequentially within a course.

    Attributes:
        lesson_number: Sequential lesson number (0, 1, 2, etc.) for ordering.
        title: Human-readable lesson title.
        lesson_link: Optional URL link to the lesson content or video.

    Examples:
        >>> lesson = Lesson(lesson_number=1, title="Introduction to Python")
        >>> lesson.lesson_number
        1
        >>> lesson.title
        'Introduction to Python'
    """

    lesson_number: int  # Sequential lesson number (1, 2, 3, etc.)
    title: str  # Lesson title
    lesson_link: Optional[str] = None  # URL link to the lesson


class Course(BaseModel):
    """Represents a complete course with its lessons and metadata.

    Serves as the main container for course information and lesson organization.
    The title field acts as a unique identifier for the course in the vector store.

    Attributes:
        title: Full course title (used as unique identifier in the system).
        course_link: Optional URL link to the main course page or platform.
        instructor: Optional instructor name or organization.
        lessons: List of Lesson objects contained in this course.

    Examples:
        >>> course = Course(
        ...     title="Machine Learning Fundamentals",
        ...     instructor="Dr. Smith",
        ...     lessons=[Lesson(lesson_number=0, title="Introduction")]
        ... )
        >>> len(course.lessons)
        1
        >>> course.instructor
        'Dr. Smith'
    """

    title: str  # Full course title (used as unique identifier)
    course_link: Optional[str] = None  # URL link to the course
    instructor: Optional[str] = None  # Course instructor name (optional metadata)
    lessons: List[Lesson] = []  # List of lessons in this course


class CourseChunk(BaseModel):
    """Represents a text chunk from a course for vector storage and retrieval.

    Used by the vector store to maintain context and attribution for search results.
    Each chunk contains a portion of course content with metadata for proper
    source attribution and filtering.

    Attributes:
        content: The actual text content of this chunk, potentially enhanced
            with lesson context for better search relevance.
        course_title: Title of the course this chunk belongs to (for filtering).
        lesson_number: Optional lesson number if chunk is from a specific lesson.
        chunk_index: Sequential position of this chunk within the document
            (used for unique ID generation).

    Examples:
        >>> chunk = CourseChunk(
        ...     content="Machine learning is a subset of artificial intelligence...",
        ...     course_title="AI Fundamentals",
        ...     lesson_number=1,
        ...     chunk_index=0
        ... )
        >>> chunk.course_title
        'AI Fundamentals'
        >>> chunk.lesson_number
        1
    """

    content: str  # The actual text content
    course_title: str  # Which course this chunk belongs to
    lesson_number: Optional[int] = None  # Which lesson this chunk is from
    chunk_index: int  # Position of this chunk in the document
