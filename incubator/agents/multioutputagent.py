import os
import anthropic
from typing import List, Dict, Optional, Any, Tuple, Set, Callable
from dataclasses import dataclass, field
import networkx as nx
import json
import time
import uuid
from dotenv import load_dotenv

from incubator.agents.agent import Agent
from incubator.messages.message import Message
# Example: Multi-output agent implementation
class MultiOutputAgent(Agent):
    """Agent that generates multiple distinct outputs based on a single input"""
    
    def process(self, messages: List[Message], context: Optional[Dict[str, Any]] = None) -> str:
        """Process messages and generate a response with multiple outputs"""
        # Use the base class to generate the full response
        full_response = super().process(messages, context)
        
        # This is a simplified example - in a real implementation,
        # you would use a more robust parsing mechanism
        
        # Looking for sections marked with specific tags
        sections = {}
        
        # Try to parse sections like "## [OUTPUT_NAME] ... ##"
        import re
        pattern = r"## \[([A-Za-z0-9_]+)\](.*?)(?=## \[|$)"
        matches = re.findall(pattern, full_response, re.DOTALL)
        
        if matches:
            for output_name, content in matches:
                sections[output_name.strip()] = content.strip()
        else:
            # Fallback to the default output if no sections found
            sections["default"] = full_response
        
        return sections