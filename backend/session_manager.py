from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class Message:
    """Represents a single message in a conversation"""
    role: str     # "user" or "assistant"
    content: str  # The message content

class SessionManager:
    """Manages conversation sessions and message history"""
    
    def __init__(self, max_history: int = 5):
        """Initialize the session manager.
        
        Args:
            max_history: Maximum number of message exchanges to keep per session.
                        Each exchange contains a user message and assistant response.
        """
        self.max_history = max_history
        self.sessions: Dict[str, List[Message]] = {}
        self.session_counter = 0
    
    def create_session(self) -> str:
        """Create a new conversation session.
        
        Returns:
            str: Unique session identifier for the new conversation session.
        """
        self.session_counter += 1
        session_id = f"session_{self.session_counter}"
        self.sessions[session_id] = []
        return session_id
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to the conversation history.
        
        Args:
            session_id: Unique identifier for the conversation session.
            role: Message role, either 'user' or 'assistant'.
            content: Text content of the message.
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        message = Message(role=role, content=content)
        self.sessions[session_id].append(message)
        
        # Keep conversation history within limits
        if len(self.sessions[session_id]) > self.max_history * 2:
            self.sessions[session_id] = self.sessions[session_id][-self.max_history * 2:]
    
    def add_exchange(self, session_id: str, user_message: str, assistant_message: str):
        """Add a complete question-answer exchange to the conversation history.
        
        Args:
            session_id: Unique identifier for the conversation session.
            user_message: The user's question or input.
            assistant_message: The assistant's response.
        """
        self.add_message(session_id, "user", user_message)
        self.add_message(session_id, "assistant", assistant_message)
    
    def get_conversation_history(self, session_id: Optional[str]) -> Optional[str]:
        """Get formatted conversation history for a session.
        
        Args:
            session_id: Unique identifier for the conversation session.
                       Can be None if no session exists.
        
        Returns:
            Optional[str]: Formatted conversation history with role prefixes,
                          or None if session doesn't exist or has no messages.
        """
        if not session_id or session_id not in self.sessions:
            return None
        
        messages = self.sessions[session_id]
        if not messages:
            return None
        
        # Format messages for context
        formatted_messages = []
        for msg in messages:
            formatted_messages.append(f"{msg.role.title()}: {msg.content}")
        
        return "\n".join(formatted_messages)
    
    def clear_session(self, session_id: str):
        """Clear all messages from a session.
        
        Args:
            session_id: Unique identifier for the conversation session to clear.
        """
        if session_id in self.sessions:
            self.sessions[session_id] = []