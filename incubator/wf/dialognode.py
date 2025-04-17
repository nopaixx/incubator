import os
from typing import List, Dict, Optional, Any, Tuple, Set, Callable
import json
import logging
from dotenv import load_dotenv

from incubator.agents.agent import Agent
from incubator.messages.message import Message
from incubator.wf.node import Node
from incubator.wf.dialogcontroller import IterativeDialogController

class IterativeDialogNode(Node):
    """
    A specialized node that manages an iterative dialogue between two agents.
    """
    
    def __init__(self, id: str, 
                 agent_a: Agent, 
                 agent_b: Agent, 
                 max_iterations: int = 3,
                 terminate_on_keywords: Optional[List[str]] = None,
                 termination_condition: Optional[Callable[[List[Message]], bool]] = None,
                 input_processor: Optional[Callable[[List[Tuple[str, str, Message]]], List[Message]]] = None,
                 output_processor: Optional[Callable[[str, List[Message]], Dict[str, Dict[str, Any]]]] = None,
                 content_markers: Optional[Dict[str, str]] = None,
                 verbose: bool = False):
        """
        Initialize an iterative dialogue node
        
        Args:
            id: Unique identifier for the node
            agent_a: First agent in the dialogue (initiator)
            agent_b: Second agent in the dialogue (responder)
            max_iterations: Maximum number of back-and-forth exchanges
            terminate_on_keywords: List of keywords that will trigger early termination
            termination_condition: Optional function that determines if dialogue should terminate early
            input_processor: Function to process incoming messages
            output_processor: Function to process the final output
            content_markers: Dictionary of marker names to strings that indicate specific content in the dialogue
                             (e.g. {"final_idea": "IDEA FINAL:", "code": "CODE:", "summary": "SUMMARY:"})
            verbose: Whether to log detailed progress information
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
                         output_processor=output_processor)
        
        # Set up the dialogue controller
        self.dialogue_controller = IterativeDialogController(
            agent_a=agent_a,
            agent_b=agent_b,
            max_iterations=max_iterations,
            terminate_on_keywords=terminate_on_keywords,
            termination_condition=termination_condition,
            verbose=verbose
        )
        
        self.content_markers = content_markers or {}
        self.verbose = verbose
        self.logger = logging.getLogger(f'DialogNode.{id}')
        
        # Configure logging
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(name)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
    
    def log(self, message: str) -> None:
        """Helper method for logging"""
        self.logger.info(message)
        if self.verbose:
            print(f"[DialogNode.{self.id}] {message}")
    
    def _default_output_processor(self, output: str, conversation: List[Message]) -> Dict[str, Dict[str, Any]]:
        """
        Default processor that provides the output across multiple ports
        
        Args:
            output: The final output from the dialogue
            conversation: The full conversation history
            
        Returns:
            Dictionary mapping output ports to their content and metadata
        """
        # Create a formatted conversation string
        formatted_conversation = []
        for msg in conversation:
            role = msg.role
            if role == "assistant" and "agent" in msg.metadata:
                role = msg.metadata["agent"]
            formatted_conversation.append(f"[{role.upper()}]: {msg.content}")
        
        conversation_text = "\n\n".join(formatted_conversation)
        
        # Create results dictionary with default output
        results = {
            "default": {
                "content": output,
                "metadata": {"type": "final_response"}
            },
            "conversation": {
                "content": conversation_text,
                "metadata": {"type": "conversation_history", "messages": conversation}
            }
        }
        
        # Process any content markers to extract specific content
        for port_name, marker in self.content_markers.items():
            marked_content = self.dialogue_controller.extract_marked_content(marker)
            if marked_content:
                results[port_name] = {
                    "content": marked_content,
                    "metadata": {"type": port_name, "marker": marker}
                }
        
        # Add final responses from each agent
        agent_a_name = self.dialogue_controller.agent_a.name
        agent_b_name = self.dialogue_controller.agent_b.name
        
        agent_a_final = self.dialogue_controller.get_agent_final_response(agent_a_name)
        if agent_a_final:
            port_name = f"{agent_a_name.lower()}_final"
            results[port_name] = {
                "content": agent_a_final,
                "metadata": {"type": "agent_final_response", "agent": agent_a_name}
            }
            
        agent_b_final = self.dialogue_controller.get_agent_final_response(agent_b_name)
        if agent_b_final:
            port_name = f"{agent_b_name.lower()}_final"
            results[port_name] = {
                "content": agent_b_final,
                "metadata": {"type": "agent_final_response", "agent": agent_b_name}
            }
        
        # Create summary with metadata
        dialogue_summary = {
            "iterations_completed": len(conversation) // 2,  # Approximate
            "message_count": len(conversation),
            "agents": [msg.metadata.get("agent", "user") for msg in conversation if "agent" in msg.metadata],
            "final_response": output
        }
        
        results["summary"] = {
            "content": json.dumps(dialogue_summary, indent=2),
            "metadata": {"type": "dialogue_summary"}
        }
        
        return results
    
    def _extract_initial_prompt(self, processed_messages: List[Message]) -> str:
        """Extract the initial prompt from processed messages"""
        if not processed_messages:
            self.log("No initial messages provided, using default prompt")
            return "Please start a discussion."
        
        # Use the content of the first message as the initial prompt
        initial_prompt = processed_messages[0].content
        self.log(f"Initial prompt extracted: {initial_prompt[:50]}...")
        return initial_prompt
    
    def process(self, inputs: List[Tuple[str, str, Message]], context: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Process inputs through this dialogue node
        
        Args:
            inputs: List of (source_node_id, output_port, message) tuples
            context: Additional context information
            
        Returns:
            Dictionary mapping output ports to their content and metadata
        """
        try:
            self.log(f"Processing {len(inputs)} inputs")
            
            # Process incoming messages using the input processor
            processed_messages = self.input_processor(inputs)
            self.log(f"Input processor returned {len(processed_messages)} messages")
            
            # Extract the initial prompt
            initial_prompt = self._extract_initial_prompt(processed_messages)
            
            # Start the dialogue
            self.log("Starting dialogue")
            conversation, final_output = self.dialogue_controller.start_dialogue(initial_prompt, context)
            self.log(f"Dialogue completed with {len(conversation)} messages")
            
            # Process the output
            if hasattr(self, 'output_processor') and self.output_processor:
                outputs = self.output_processor(final_output, conversation)
            else:
                outputs = self._default_output_processor(final_output, conversation)
                
            self.log(f"Output processed with {len(outputs)} ports")
            
            return outputs
            
        except Exception as e:
            error_message = f"Error in dialogue node processing: {str(e)}"
            self.log(error_message)
            import traceback
            self.log(traceback.format_exc())
            
            # Return a basic error output
            return {
                "default": {
                    "content": error_message,
                    "metadata": {"error": True}
                },
                "error": {
                    "content": error_message,
                    "metadata": {"error": True, "exception": str(e)}
                }
            }