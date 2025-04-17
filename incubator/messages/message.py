import os
import anthropic
from typing import List, Dict, Optional, Any, Tuple, Set, Callable
from dataclasses import dataclass, field
import networkx as nx
import json
import time
import uuid
from dotenv import load_dotenv



@dataclass
class Message:
    """Generic class to represent a message in any conversation"""
    role: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_api_format(self) -> Dict[str, Any]:
        """Converts the message to the format expected by the API"""
        if self.role == "user":
            return {
                "role": self.role,
                "content": [
                    {
                        "type": "text",
                        "text": self.content
                    }
                ]
            }
        elif self.role == "assistant":
            # Simple format for assistant messages
            return {
                "role": self.role,
                "content": self.content
            }
        else:
            # For system or other roles
            return {
                "role": self.role,
                "content": self.content
            }
