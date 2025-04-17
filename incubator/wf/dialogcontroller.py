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


class IterativeDialogController:
    """
    Controller for managing iterative dialogues between two agents.
    This allows agents to have back-and-forth conversations to refine ideas.
    """
    
    def __init__(self, 
                 agent_a: Agent, 
                 agent_b: Agent, 
                 max_iterations: int = 3,
                 termination_condition: Optional[Callable[[List[Message]], bool]] = None,
                 verbose: bool = False):
        """
        Initialize an iterative dialogue controller
        
        Args:
            agent_a: First agent in the dialogue (initiator)
            agent_b: Second agent in the dialogue (responder)
            max_iterations: Maximum number of back-and-forth exchanges
            termination_condition: Optional function that determines if the dialogue should terminate early
            verbose: Whether to print debug information during dialogue execution
        """
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.max_iterations = max_iterations
        self.termination_condition = termination_condition or (lambda _: False)
        self.conversation_history = []
        self.verbose = verbose
    
    def log(self, message: str) -> None:
        """Helper method for logging when verbose mode is enabled"""
        if self.verbose:
            print(f"[DialogController] {message}")
    
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
        self.log(f"Initial prompt: {initial_prompt}")
        
        # First response from agent A to the initial prompt
        current_messages = [initial_message]
        response_a = self.agent_a.process(current_messages, context)
        self.log(f"Agent A ({self.agent_a.name}) initial response generated")
        
        # Create and store the message
        message_a = Message(role="assistant", content=response_a, metadata={"agent": self.agent_a.name})
        self.conversation_history.append(message_a)
        
        # Update current messages for agent B
        current_messages = [message_a]
        
        # Main dialogue loop
        for i in range(self.max_iterations):
            self.log(f"Starting iteration {i+1}/{self.max_iterations}")
            
            # Determine if this is the final iteration
            is_final_iteration = (i == self.max_iterations - 1)
            
            # For the final iteration of B, we might want to add special instructions
            if is_final_iteration and hasattr(self.agent_b, 'system_prompt'):
                # Add a note to the system prompt for the final iteration
                original_prompt_b = self.agent_b.system_prompt
                final_prompt_b = original_prompt_b + "\n\nNOTE: This is the final iteration. Please provide comprehensive feedback."
                self.agent_b.system_prompt = final_prompt_b
            
            # Agent B processes agent A's response
            self.log(f"Agent B ({self.agent_b.name}) processing response")
            response_b = self.agent_b.process(current_messages, context)
            self.log(f"Agent B response generated")
            
            # Restore original system prompt if modified
            if is_final_iteration and hasattr(self.agent_b, 'system_prompt'):
                self.agent_b.system_prompt = original_prompt_b
            
            # Create and store the message
            message_b = Message(role="assistant", content=response_b, metadata={"agent": self.agent_b.name})
            self.conversation_history.append(message_b)
            
            # Check if we should terminate early
            if self.termination_condition(self.conversation_history):
                self.log("Termination condition met, ending dialogue early")
                break
            
            # If this is the final iteration, we're done
            if is_final_iteration:
                self.log("Final iteration completed, ending dialogue")
                break
            
            # Update current messages for the next iteration
            current_messages = [message_b]
            
            # Agent A responds to agent B (for next iteration)
            self.log(f"Agent A ({self.agent_a.name}) processing response")
            response_a = self.agent_a.process(current_messages, context)
            self.log(f"Agent A response generated")
            
            # Create and store the message
            message_a = Message(role="assistant", content=response_a, metadata={"agent": self.agent_a.name})
            self.conversation_history.append(message_a)
            
            # Update current messages for the next iteration
            current_messages = [message_a]
            
            # Check if we should terminate early again
            if self.termination_condition(self.conversation_history):
                self.log("Termination condition met, ending dialogue early")
                break
        
        # Return the full conversation history and the final output
        # (which is the last message in the conversation)
        final_message = self.conversation_history[-1]
        self.log(f"Dialogue completed. Final message from: {final_message.metadata.get('agent', 'unknown')}")
        
        return self.conversation_history, final_message.content
    
    def get_formatted_conversation(self) -> str:
        """Returns the conversation history as a formatted string"""
        formatted = []
        for msg in self.conversation_history:
            role = msg.role
            if role == "assistant" and "agent" in msg.metadata:
                role = msg.metadata["agent"]
            formatted.append(f"[{role.upper()}]: {msg.content}")
        return "\n\n".join(formatted)
    
    def get_summary(self) -> Dict[str, Any]:
        """Returns a summary of the dialogue with metadata and contents"""
        return {
            "iterations_completed": len(self.conversation_history) // 2,  # Approximate
            "agent_a": self.agent_a.name,
            "agent_b": self.agent_b.name,
            "message_count": len(self.conversation_history),
            "final_message": self.conversation_history[-1].content if self.conversation_history else "",
            "formatted_conversation": self.get_formatted_conversation()
        }