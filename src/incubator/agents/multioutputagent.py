import os
import anthropic
from typing import List, Dict, Optional, Any, Tuple, Set, Callable, Union
from dataclasses import dataclass, field
import networkx as nx
import json
import time
import uuid
import re
from dotenv import load_dotenv

from incubator.agents.agent import Agent
from incubator.messages.message import Message

class MultiOutputAgent(Agent):
    """Agent that generates multiple distinct outputs based on a single input"""
    
    def __init__(self, name: str, description: str, llm_client: Any, system_prompt: str = "", 
                config: Optional[Dict[str, Any]] = None, output_keys: Optional[List[str]] = None):
        """
        Initialize a MultiOutputAgent
        
        Args:
            name: Unique identifier for the agent
            description: Human-readable description of the agent's purpose
            llm_client: The LLM client used to generate responses
            system_prompt: The system prompt that defines the agent's behavior
            config: Configuration options for the LLM
            output_keys: List of expected output section keys (optional)
        """
        super().__init__(name, description, llm_client, system_prompt, config)
        self.output_keys = output_keys or []
        
        # Ensure the system prompt includes instructions for formatting outputs
        if not "## [" in system_prompt and output_keys:
            formatted_keys = ", ".join([f"## [{key}]" for key in output_keys])
            self.system_prompt += f"\n\nYour response should be formatted with clearly labeled sections using: {formatted_keys}"
    
    def process(self, messages: List[Message], context: Optional[Dict[str, Any]] = None) -> Union[str, Dict[str, str]]:
        """Process messages and generate a response with multiple outputs"""
        # Use the base class to generate the full response
        full_response = super().process(messages, context)
        
        # Parse the response into sections
        sections = self._parse_sections(full_response)
        
        # If requested output_keys were provided, validate that all are present
        if self.output_keys and not all(key in sections for key in self.output_keys):
            missing_keys = [key for key in self.output_keys if key not in sections]
            print(f"Warning: Missing expected output sections: {missing_keys}")
            
            # Add any missing sections as empty strings
            for key in missing_keys:
                sections[key] = ""
        
        # Return the sections dictionary if multiple sections were found
        # Otherwise return the full response as a string
        if len(sections) > 1 or (self.output_keys and len(sections) == len(self.output_keys)):
            return sections
        else:
            return full_response
    
    def _parse_sections(self, text: str) -> Dict[str, str]:
        """Parse sections from text using multiple formats"""
        sections = {}
        
        # Try different section formats
        
        # Format 1: ## [KEY] content ##
        pattern1 = r"## \[([A-Za-z0-9_]+)\](.*?)(?=## \[|$)"
        matches1 = re.findall(pattern1, text, re.DOTALL)
        if matches1:
            for output_name, content in matches1:
                sections[output_name.strip()] = content.strip()
        
        # Format 2: ## KEY ## content
        if not sections:
            pattern2 = r"## ([A-Za-z0-9_]+) ##(.*?)(?=## [A-Za-z0-9_]+ ##|$)"
            matches2 = re.findall(pattern2, text, re.DOTALL)
            if matches2:
                for output_name, content in matches2:
                    sections[output_name.strip()] = content.strip()
        
        # Format 3: # KEY content
        if not sections:
            pattern3 = r"# ([A-Za-z0-9_]+)(.*?)(?=# [A-Za-z0-9_]+|$)"
            matches3 = re.findall(pattern3, text, re.DOTALL)
            if matches3:
                for output_name, content in matches3:
                    sections[output_name.strip()] = content.strip()
        
        # If no sections found, create a default section
        if not sections:
            sections["default"] = text.strip()
        
        return sections
    
    def get_specific_output(self, full_output: Union[str, Dict[str, str]], key: str) -> Optional[str]:
        """Extract a specific output by key"""
        if isinstance(full_output, dict):
            return full_output.get(key)
        elif isinstance(full_output, str) and key == "default":
            return full_output
        else:
            # Try to parse the string for the requested key
            sections = self._parse_sections(full_output)
            return sections.get(key)
    
    def combine_outputs(self, outputs: Dict[str, str], separator: str = "\n\n") -> str:
        """Combine multiple outputs into a single string"""
        combined = []
        for key, content in outputs.items():
            combined.append(f"## [{key}] ##\n{content}")
        return separator.join(combined)