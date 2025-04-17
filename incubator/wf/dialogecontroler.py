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


# Patterns for complex agent interactions
class IterativeDialogController:
    """
    Controller for managing iterative dialogues between two agents.
    This allows agents to have back-and-forth conversations to refine ideas.
    """
    
    def __init__(self, 
                 agent_a: Agent, 
                 agent_b: Agent, 
                 max_iterations: int = 3,
                 termination_condition: Optional[Callable[[List[Message]], bool]] = None):
        """
        Initialize an iterative dialogue controller
        
        Args:
            agent_a: First agent in the dialogue (initiator)
            agent_b: Second agent in the dialogue (responder)
            max_iterations: Maximum number of back-and-forth exchanges
            termination_condition: Optional function that determines if the dialogue should terminate early
        """
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.max_iterations = max_iterations
        self.termination_condition = termination_condition or (lambda _: False)
        self.conversation_history = []
    
    def start_dialogue(self, initial_prompt: str, context: Optional[Dict[str, Any]] = None) -> Tuple[List[Message], str]:
        """
        Start and manage an iterative dialogue between the agents
        
        Args:
            initial_prompt: The initial input to start the dialogue
            context: Additional context for the agents
            
        Returns:
            Tuple containing the full conversation history and the final refined output
        """
        # Reset conversation history
        self.conversation_history = []
        
        # Create the initial message
        initial_message = Message(role="user", content=initial_prompt)
        self.conversation_history.append(initial_message)
        
        # Start with agent A
        current_messages = [initial_message]
        
        for i in range(self.max_iterations):
            # Determine if this is the final iteration
            is_final_iteration = (i == self.max_iterations - 1)
            
            # Step 1: Agent A generates a response
            if i > 0:  # Skip first iteration for Agent A if we already have the initial prompt
                # For the final iteration, we might want to add special instructions
                if is_final_iteration and hasattr(self.agent_a, 'system_prompt'):
                    # Add a note to the system prompt for the final iteration
                    original_prompt = self.agent_a.system_prompt
                    final_prompt = original_prompt + "\n\nNOTE: This is the final iteration. Please provide your complete and refined response."
                    self.agent_a.system_prompt = final_prompt
                
                # Agent A processes the current messages
                response_a = self.agent_a.process(current_messages, context)
                
                # Restore original system prompt if modified
                if is_final_iteration and hasattr(self.agent_a, 'system_prompt'):
                    self.agent_a.system_prompt = original_prompt
                
                # Create and store the message
                message_a = Message(role="assistant", content=response_a, metadata={"agent": self.agent_a.name})
                self.conversation_history.append(message_a)
                
                # Update current messages for agent B
                current_messages = [message_a]
            
            # Check if we should terminate early
            if self.termination_condition(self.conversation_history):
                break
            
            # If this is the final iteration, we're done after agent A's response
            if is_final_iteration:
                break
            
            # Step 2: Agent B responds to agent A
            # For the final iteration of B, we might want to add special instructions
            if is_final_iteration and hasattr(self.agent_b, 'system_prompt'):
                # Add a note to the system prompt for the final iteration
                original_prompt_b = self.agent_b.system_prompt
                final_prompt_b = original_prompt_b + "\n\nNOTE: This is the final iteration. Please provide comprehensive feedback."
                self.agent_b.system_prompt = final_prompt_b
            
            # Agent B processes agent A's response
            response_b = self.agent_b.process(current_messages, context)
            
            # Restore original system prompt if modified
            if is_final_iteration and hasattr(self.agent_b, 'system_prompt'):
                self.agent_b.system_prompt = original_prompt_b
            
            # Create and store the message
            message_b = Message(role="assistant", content=response_b, metadata={"agent": self.agent_b.name})
            self.conversation_history.append(message_b)
            
            # Update current messages for the next iteration
            current_messages = [message_b]
            
            # Check if we should terminate early again
            if self.termination_condition(self.conversation_history):
                break
        
        # Return the full conversation history and the final output
        # (which is the last message in the conversation)
        return self.conversation_history, self.conversation_history[-1].content

