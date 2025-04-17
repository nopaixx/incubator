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


from incubator.wf.node import Node

class IterativeDialogNode(Node):
    """
    A specialized node that manages an iterative dialogue between two agents.
    """
    
    def __init__(self, id: str, 
                 agent_a: Agent, 
                 agent_b: Agent, 
                 max_iterations: int = 3,
                 termination_condition: Optional[Callable[[List[Message]], bool]] = None,
                 input_processor: Optional[Callable[[List[Tuple[str, str, Message]]], List[Message]]] = None,
                 output_processor: Optional[Callable[[str, List[Message]], Dict[str, Dict[str, Any]]]] = None):
        """
        Initialize an iterative dialogue node
        
        Args:
            id: Unique identifier for the node
            agent_a: First agent in the dialogue (initiator)
            agent_b: Second agent in the dialogue (responder)
            max_iterations: Maximum number of back-and-forth exchanges
            termination_condition: Optional function that determines if dialogue should terminate early
            input_processor: Function to process incoming messages
            output_processor: Function to process the final output
        """
        # Create a dummy agent for the Node superclass
        # (we won't use it directly, but it satisfies the Node requirements)
        dummy_agent = Agent(
            name=f"{agent_a.name}_{agent_b.name}_Dialog",
            description=f"Dialogue between {agent_a.name} and {agent_b.name}",
            llm_client=agent_a.llm_client  # Use the same LLM client as agent_a
        )
        
        # Initialize the Node superclass
        super().__init__(id=id, agent=dummy_agent, 
                         input_processor=input_processor, 
                         output_processor=output_processor or self._default_dialogue_output_processor)
        
        # Set up the dialogue controller
        self.dialogue_controller = IterativeDialogController(
            agent_a=agent_a,
            agent_b=agent_b,
            max_iterations=max_iterations,
            termination_condition=termination_condition
        )
    
    def _default_dialogue_output_processor(self, output: str, conversation: List[Message]) -> Dict[str, Dict[str, Any]]:
        """
        Default processor that provides both the final output and the full conversation
        
        Args:
            output: The final output from the dialogue
            conversation: The full conversation history
            
        Returns:
            Dictionary mapping output ports to their content and metadata
        """
        return {
            "final": {
                "content": output,
                "metadata": {"type": "final_output"}
            },
            "conversation": {
                "content": "\n\n".join([f"[{msg.role.upper()}]: {msg.content}" for msg in conversation]),
                "metadata": {"type": "conversation_history", "messages": conversation}
            }
        }
    
    def process(self, inputs: List[Tuple[str, str, Message]], context: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Process inputs through this dialogue node
        
        Args:
            inputs: List of (source_node_id, output_port, message) tuples
            context: Additional context information
            
        Returns:
            Dictionary mapping output ports to their content and metadata
        """
        # Process incoming messages using the input processor
        processed_messages = self.input_processor(inputs)
        
        # Extract the initial prompt from the processed messages
        if not processed_messages:
            initial_prompt = "Please start a discussion."
        else:
            # Use the content of the first message as the initial prompt
            initial_prompt = processed_messages[0].content
        
        # Start the dialogue
        conversation, final_output = self.dialogue_controller.start_dialogue(initial_prompt, context)
        
        # Use the output processor to format the results
        # We pass both the final output and the full conversation
        return self._default_dialogue_output_processor(final_output, conversation)
