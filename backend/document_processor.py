import logging
import os
import re
from typing import List, Tuple

from logging_config import get_logger
from models import Course, CourseChunk, Lesson

logger = get_logger(__name__)


class DocumentProcessor:
    """Processes course documents and extracts structured information"""

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def read_file(self, file_path: str) -> str:
        """Read content from file with UTF-8 encoding"""
        logger.debug(f"Reading file: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                logger.info(
                    f"Successfully read file {file_path} - {len(content)} characters"
                )
                return content
        except UnicodeDecodeError as e:
            logger.warning(
                f"UTF-8 decode failed for {file_path}, retrying with error handling: {e}"
            )
            # If UTF-8 fails, try with error handling
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                content = file.read()
                logger.info(
                    f"Read file {file_path} with error handling - {len(content)} characters"
                )
                return content
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            raise

    def chunk_text(self, text: str) -> List[str]:
        """Split text into sentence-based chunks with overlap using config settings"""
        logger.debug(
            f"Starting text chunking - Input length: {len(text)} chars, chunk_size: {self.chunk_size}, overlap: {self.chunk_overlap}"
        )

        # Clean up the text
        text = re.sub(r"\s+", " ", text.strip())  # Normalize whitespace
        logger.debug(f"Text normalized - Length after cleanup: {len(text)} chars")

        # Better sentence splitting that handles abbreviations
        # This regex looks for periods followed by whitespace and capital letters
        # but ignores common abbreviations
        sentence_endings = re.compile(
            r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s+(?=[A-Z])"
        )
        sentences = sentence_endings.split(text)

        # Clean sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        logger.debug(f"Text split into {len(sentences)} sentences")

        chunks = []
        i = 0

        while i < len(sentences):
            current_chunk = []
            current_size = 0

            # Build chunk starting from sentence i
            for j in range(i, len(sentences)):
                sentence = sentences[j]

                # Calculate size with space
                space_size = 1 if current_chunk else 0
                total_addition = len(sentence) + space_size

                # Check if adding this sentence would exceed chunk size
                if current_size + total_addition > self.chunk_size and current_chunk:
                    break

                current_chunk.append(sentence)
                current_size += total_addition

            # Add chunk if we have content
            if current_chunk:
                chunks.append(" ".join(current_chunk))

                # Calculate overlap for next chunk
                if hasattr(self, "chunk_overlap") and self.chunk_overlap > 0:
                    # Find how many sentences to overlap
                    overlap_size = 0
                    overlap_sentences = 0

                    # Count backwards from end of current chunk
                    for k in range(len(current_chunk) - 1, -1, -1):
                        sentence_len = len(current_chunk[k]) + (
                            1 if k < len(current_chunk) - 1 else 0
                        )
                        if overlap_size + sentence_len <= self.chunk_overlap:
                            overlap_size += sentence_len
                            overlap_sentences += 1
                        else:
                            break

                    # Move start position considering overlap
                    next_start = i + len(current_chunk) - overlap_sentences
                    i = max(next_start, i + 1)  # Ensure we make progress
                else:
                    # No overlap - move to next sentence after current chunk
                    i += len(current_chunk)
            else:
                # No sentences fit, move to next
                i += 1

        logger.debug(
            f"Text chunking completed - Created {len(chunks)} chunks from {len(sentences)} sentences"
        )
        if logger.isEnabledFor(logging.DEBUG):
            avg_chunk_size = (
                sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0
            )
            logger.debug(f"Average chunk size: {avg_chunk_size:.1f} characters")

        return chunks

    def process_course_document(
        self, file_path: str
    ) -> Tuple[Course, List[CourseChunk]]:
        """
        Process a course document with expected format:
        Line 1: Course Title: [title]
        Line 2: Course Link: [url]
        Line 3: Course Instructor: [instructor]
        Following lines: Lesson markers and content
        """
        logger.info(f"=== Processing course document: {file_path} ===")
        content = self.read_file(file_path)
        filename = os.path.basename(file_path)
        logger.debug(f"Filename: {filename}, Content length: {len(content)} characters")

        lines = content.strip().split("\n")
        logger.debug(f"Document split into {len(lines)} lines")

        # Extract course metadata from first three lines
        course_title = filename  # Default fallback
        course_link = None
        instructor_name = "Unknown"

        logger.debug("=== Parsing course metadata ===")
        # Parse course title from first line
        if len(lines) >= 1 and lines[0].strip():
            title_match = re.match(
                r"^Course Title:\s*(.+)$", lines[0].strip(), re.IGNORECASE
            )
            if title_match:
                course_title = title_match.group(1).strip()
                logger.debug(f"Found course title in header: '{course_title}'")
            else:
                course_title = lines[0].strip()
                logger.debug(f"Using first line as course title: '{course_title}'")

        # Parse remaining lines for course metadata
        for i in range(1, min(len(lines), 4)):  # Check first 4 lines for metadata
            line = lines[i].strip()
            if not line:
                continue

            # Try to match course link
            link_match = re.match(r"^Course Link:\s*(.+)$", line, re.IGNORECASE)
            if link_match:
                course_link = link_match.group(1).strip()
                logger.debug(f"Found course link: '{course_link}'")
                continue

            # Try to match instructor
            instructor_match = re.match(
                r"^Course Instructor:\s*(.+)$", line, re.IGNORECASE
            )
            if instructor_match:
                instructor_name = instructor_match.group(1).strip()
                logger.debug(f"Found instructor: '{instructor_name}'")
                continue

        # Create course object with title as ID
        course = Course(
            title=course_title,
            course_link=course_link,
            instructor=instructor_name if instructor_name != "Unknown" else None,
        )
        logger.info(
            f"Created course object: title='{course_title}', link='{course_link}', instructor='{instructor_name}'"
        )

        # Process lessons and create chunks
        logger.debug("=== Processing lessons ===")
        course_chunks = []
        current_lesson = None
        lesson_title = None
        lesson_link = None
        lesson_content = []
        chunk_counter = 0

        # Start processing from line 4 (after metadata)
        start_index = 3
        if len(lines) > 3 and not lines[3].strip():
            start_index = 4  # Skip empty line after instructor
        logger.debug(f"Starting lesson processing from line {start_index}")

        i = start_index
        while i < len(lines):
            line = lines[i]

            # Check for lesson markers (e.g., "Lesson 0: Introduction")
            lesson_match = re.match(
                r"^Lesson\s+(\d+):\s*(.+)$", line.strip(), re.IGNORECASE
            )

            if lesson_match:
                # Process previous lesson if it exists
                if current_lesson is not None and lesson_content:
                    lesson_text = "\n".join(lesson_content).strip()
                    logger.debug(
                        f"Processing previous lesson {current_lesson}: '{lesson_title}' ({len(lesson_text)} chars)"
                    )
                    if lesson_text:
                        # Add lesson to course
                        lesson = Lesson(
                            lesson_number=current_lesson,
                            title=lesson_title,
                            lesson_link=lesson_link,
                        )
                        course.lessons.append(lesson)
                        logger.debug(
                            f"Added lesson {current_lesson} to course - Link: {lesson_link}"
                        )

                        # Create chunks for this lesson
                        chunks = self.chunk_text(lesson_text)
                        logger.debug(
                            f"Created {len(chunks)} chunks for lesson {current_lesson}"
                        )
                        for idx, chunk in enumerate(chunks):
                            # For the first chunk of each lesson, add lesson context
                            if idx == 0:
                                chunk_with_context = (
                                    f"Lesson {current_lesson} content: {chunk}"
                                )
                            else:
                                chunk_with_context = chunk

                            course_chunk = CourseChunk(
                                content=chunk_with_context,
                                course_title=course.title,
                                lesson_number=current_lesson,
                                chunk_index=chunk_counter,
                            )
                            course_chunks.append(course_chunk)
                            chunk_counter += 1

                # Start new lesson
                current_lesson = int(lesson_match.group(1))
                lesson_title = lesson_match.group(2).strip()
                lesson_link = None
                logger.info(f"Found new lesson {current_lesson}: '{lesson_title}'")

                # Check if next line is a lesson link
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    link_match = re.match(
                        r"^Lesson Link:\s*(.+)$", next_line, re.IGNORECASE
                    )
                    if link_match:
                        lesson_link = link_match.group(1).strip()
                        logger.debug(
                            f"Found lesson link for lesson {current_lesson}: '{lesson_link}'"
                        )
                        i += 1  # Skip the link line so it's not added to content

                lesson_content = []
            else:
                # Add line to current lesson content
                lesson_content.append(line)

            i += 1

        # Process the last lesson
        logger.debug("=== Processing final lesson ===")
        if current_lesson is not None and lesson_content:
            lesson_text = "\n".join(lesson_content).strip()
            logger.debug(
                f"Processing final lesson {current_lesson}: '{lesson_title}' ({len(lesson_text)} chars)"
            )
            if lesson_text:
                lesson = Lesson(
                    lesson_number=current_lesson,
                    title=lesson_title,
                    lesson_link=lesson_link,
                )
                course.lessons.append(lesson)
                logger.debug(
                    f"Added final lesson {current_lesson} to course - Link: {lesson_link}"
                )

                chunks = self.chunk_text(lesson_text)
                logger.debug(
                    f"Created {len(chunks)} chunks for final lesson {current_lesson}"
                )
                for idx, chunk in enumerate(chunks):
                    # For any chunk of each lesson, add lesson context & course title

                    chunk_with_context = f"Course {course_title} Lesson {current_lesson} content: {chunk}"

                    course_chunk = CourseChunk(
                        content=chunk_with_context,
                        course_title=course.title,
                        lesson_number=current_lesson,
                        chunk_index=chunk_counter,
                    )
                    course_chunks.append(course_chunk)
                    chunk_counter += 1

        # If no lessons found, treat entire content as one document
        if not course_chunks and len(lines) > 2:
            logger.warning(
                "No lessons found in document - treating entire content as single document"
            )
            remaining_content = "\n".join(lines[start_index:]).strip()
            if remaining_content:
                chunks = self.chunk_text(remaining_content)
                logger.debug(f"Created {len(chunks)} chunks from document content")
                for chunk in chunks:
                    course_chunk = CourseChunk(
                        content=chunk,
                        course_title=course.title,
                        chunk_index=chunk_counter,
                    )
                    course_chunks.append(course_chunk)
                    chunk_counter += 1

        # Final processing summary
        lesson_count = len(course.lessons)
        chunk_count = len(course_chunks)
        logger.info(f"=== PROCESSING COMPLETE ===")
        logger.info(
            f"Course: '{course.title}' - {lesson_count} lessons, {chunk_count} chunks"
        )
        if lesson_count > 0:
            lessons_with_links = sum(
                1 for lesson in course.lessons if lesson.lesson_link
            )
            logger.info(f"Lessons with links: {lessons_with_links}/{lesson_count}")

        return course, course_chunks
