import os
import anthropic
from typing import List, Dict, Optional, Any, Tuple, Set, Callable
from dataclasses import dataclass, field
import networkx as nx
import json
import time
import uuid
from dotenv import load_dotenv

from incubator.messages.message import Message

class LLMClient:
    """Abstract class for LLM API clients"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the LLM client with API key"""
        self.api_key = api_key
    
    def generate_response(self, messages: List[Message], system_prompt: str = "", 
                         config: Dict[str, Any] = None) -> str:
        """Generate a response based on the messages and configuration"""
        raise NotImplementedError("Subclasses must implement this method")