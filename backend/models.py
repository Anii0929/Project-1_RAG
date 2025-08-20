from typing import Dict, List, Optional

from pydantic import BaseModel


class Section(BaseModel):
    """Represents a section within a document"""

    section_number: int  # Sequential section number (1, 2, 3, etc.)
    title: str  # Section title
    section_link: Optional[str] = None  # URL link to the section


class Document(BaseModel):
    """Represents a complete document with its sections"""

    title: str  # Full document title (used as unique identifier)
    document_link: Optional[str] = None  # URL link to the document
    instructor: Optional[str] = None  # Document instructor name (optional metadata)
    sections: List[Section] = []  # List of sections in this document


class DocumentChunk(BaseModel):
    """Represents a text chunk from a document for vector storage"""

    content: str  # The actual text content
    document_title: str  # Which document this chunk belongs to
    section_number: Optional[int] = None  # Which section this chunk is from
    chunk_index: int  # Position of this chunk in the document
