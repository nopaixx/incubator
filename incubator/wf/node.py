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


class Node:
    """Represents a node in the workflow graph"""
    
    def __init__(self, id: str, agent: Agent, 
                 input_processor: Optional[Callable[[List[Tuple[str, str, Message]]], List[Message]]] = None,
                 output_processor: Optional[Callable[[str], Dict[str, Dict[str, Any]]]] = None):
        """
        Initialize a node with its agent and processing functions
        
        Args:
            id: Unique identifier for the node
            agent: The agent that will process messages at this node
            input_processor: Function to process incoming messages from multiple sources
            output_processor: Function to process the agent's output for routing
        """
        self.id = id
        self.agent = agent
        self.input_processor = input_processor or self._default_input_processor
        self.output_processor = output_processor or self._default_output_processor
    
    def _default_input_processor(self, inputs: List[Tuple[str, str, Message]]) -> List[Message]:
        """
        Default processor that combines all input messages
        
        Args:
            inputs: List of (source_node_id, output_port, message) tuples
            
        Returns:
            List of messages to pass to the agent
        """
        # Simply extract the messages and ignore the source and port
        if not inputs:
            # Crear un mensaje vacío para evitar errores de API
            return [Message(role="user", content="Por favor, proporciona tu análisis basado en la información disponible.")]
        return [msg for _, _, msg in inputs]
    
    def _default_output_processor(self, output: str) -> Dict[str, Dict[str, Any]]:
        """
        Default processor that passes the output unchanged through a single "default" port
        
        Args:
            output: The raw output from the agent
            
        Returns:
            Dictionary mapping output ports to their content and metadata
        """
        return {
            "default": {
                "content": output,
                "metadata": {}
            }
        }
    
    def process(self, inputs: List[Tuple[str, str, Message]], context: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Process inputs through this node
        
        Args:
            inputs: List of (source_node_id, output_port, message) tuples
            context: Additional context information
            
        Returns:
            Dictionary mapping output ports to their content and metadata
        """
        try:
            # Process incoming messages
            processed_messages = self.input_processor(inputs)
            
            if not processed_messages:
                # Si no hay mensajes después del procesamiento, añadir uno para evitar errores de API
                processed_messages = [Message(
                    role="user", 
                    content="Por favor, proporciona tu análisis basado en la información disponible."
                )]
            
            # Generate a response using the agent
            raw_output = self.agent.process(processed_messages, context)
            
            # Process the output for routing
            return self.output_processor(raw_output)
            
        except Exception as e:
            error_message = f"Error en el nodo {self.id}: {str(e)}"
            # Retornar un output de error
            return {
                "default": {
                    "content": error_message,
                    "metadata": {"error": True, "exception": str(e)}
                },
                "error": {
                    "content": error_message,
                    "metadata": {"error": True, "exception": str(e)}
                }
            }