import os
import anthropic
from typing import List, Dict, Optional, Any, Tuple, Set, Callable
from dataclasses import dataclass, field
import networkx as nx
import json
import time
import uuid
from dotenv import load_dotenv

from incubator.llm.llmclient import LLMClient
from incubator.messages.message import Message

class Agent:
    """Base class for all agents in the system"""
    
    def __init__(self, name: str, description: str, llm_client: LLMClient, 
                 system_prompt: str = "", config: Optional[Dict[str, Any]] = None):
        """
        Initialize an agent with its core properties
        
        Args:
            name: Unique identifier for the agent
            description: Human-readable description of the agent's purpose
            llm_client: The LLM client used to generate responses
            system_prompt: The system prompt that defines the agent's behavior
            config: Configuration options for the LLM
        """
        self.name = name
        self.description = description
        self.llm_client = llm_client
        self.system_prompt = system_prompt
        
        # Default configuration
        self.config = {
            "model": "claude-3-7-sonnet-20250219",
            "max_tokens": 20000,
            "temperature": 0.7,
            "use_thinking": False,
            "thinking_budget": 10000,
            "top_p": 1.0,
            "top_k": None
        }
        
        # Update with custom configuration if provided
        if config:
            self.config.update(config)
    
    def process(self, messages: List[Message], context: Optional[Dict[str, Any]] = None) -> str:
        """
        Process input messages and generate a response
        
        Args:
            messages: List of messages to process
            context: Additional context information (optional)
            
        Returns:
            The generated response as a string
        """
        # Apply any context-specific modifications to the system prompt
        effective_system_prompt = self._prepare_system_prompt(context)
        
        # Generate and return the response
        return self.llm_client.generate_response(
            messages=messages,
            system_prompt=effective_system_prompt,
            config=self.config
        )
    
    def _prepare_system_prompt(self, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Prepare the system prompt with context-specific information
        
        Args:
            context: Additional context information
            
        Returns:
            The prepared system prompt string
        """
        if not context:
            return self.system_prompt
        
        # Here you could implement logic to modify the system prompt based on context
        # For example, injecting specific information or instructions
        
        return self.system_prompt
