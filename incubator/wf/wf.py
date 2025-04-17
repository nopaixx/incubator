import os
import anthropic
from typing import List, Dict, Optional, Any, Tuple, Set, Callable
from dataclasses import dataclass, field
import networkx as nx
import json
import time
import uuid
from dotenv import load_dotenv
import traceback

from incubator.agents.agent import Agent
from incubator.messages.message import Message
from incubator.llm.llmclient import LLMClient

from incubator.wf.node import Node


class WorkflowEngine:
    """Engine to execute workflows defined as directed graphs of nodes"""
    
    def __init__(self, debug: bool = False):
        """
        Initialize the workflow engine
        
        Args:
            debug: Whether to print debug information
        """
        self.graph = nx.MultiDiGraph()  # Use MultiDiGraph to allow multiple edges between nodes
        self.input_nodes = set()
        self.output_nodes = set()
        self.node_instances = {}  # Map of node_id -> Node instance
        self.debug = debug
    
    def log(self, message: str) -> None:
        """Log debug messages when debug mode is enabled"""
        if self.debug:
            print(f"[WorkflowEngine] {message}")
    
    def add_node(self, node: Node, is_input: bool = False, is_output: bool = False) -> None:
        """
        Add a node to the workflow
        
        Args:
            node: The node to add
            is_input: Whether this node can receive external input
            is_output: Whether this node produces external output
        """
        if node.id in self.graph.nodes:
            raise ValueError(f"Node with id '{node.id}' already exists")
        
        self.graph.add_node(node.id)
        self.node_instances[node.id] = node
        
        if is_input:
            self.input_nodes.add(node.id)
        
        if is_output:
            self.output_nodes.add(node.id)
    
    def add_edge(self, from_node_id: str, to_node_id: str, 
                from_port: str = "default", to_port: str = "input", 
                metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a directed edge between nodes with specific ports
        
        Args:
            from_node_id: Source node ID
            to_node_id: Target node ID
            from_port: Output port of the source node (default: "default")
            to_port: Input port of the target node (default: "input")
            metadata: Additional metadata for the edge
        """
        if from_node_id not in self.graph.nodes:
            raise ValueError(f"Source node '{from_node_id}' does not exist")
        
        if to_node_id not in self.graph.nodes:
            raise ValueError(f"Target node '{to_node_id}' does not exist")
        
        # Create edge data with port information
        edge_data = {
            "from_port": from_port,
            "to_port": to_port,
            "metadata": metadata or {}
        }
        
        self.graph.add_edge(from_node_id, to_node_id, key=f"{from_port}_to_{to_port}", **edge_data)
    
    def validate(self) -> bool:
        """
        Validate the workflow graph
        
        Returns:
            True if the graph is valid, False otherwise
        """
        # Check if there are any input nodes
        if not self.input_nodes:
            print("Error: No input nodes defined")
            return False
        
        # Check if there are any output nodes
        if not self.output_nodes:
            print("Error: No output nodes defined")
            return False
        
        # Check for cycles
        if not nx.is_directed_acyclic_graph(self.graph):
            print("Error: Workflow contains cycles")
            return False
        
        # Check reachability from input nodes to output nodes
        for input_id in self.input_nodes:
            can_reach_output = False
            for output_id in self.output_nodes:
                if nx.has_path(self.graph, input_id, output_id):
                    can_reach_output = True
                    break
            
            if not can_reach_output:
                print(f"Error: Input node '{input_id}' cannot reach any output node")
                return False
        
        return True
    
    def execute(self, inputs: Dict[str, str], context: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Execute the workflow with the given inputs
        
        Args:
            inputs: Dictionary mapping input node IDs to input content
            context: Additional context information
            
        Returns:
            Dictionary mapping output node IDs to their port outputs (with full content and metadata)
        """
        if not self.validate():
            raise ValueError("Invalid workflow graph")
        
        # Check that all input keys are valid input nodes
        for node_id in inputs.keys():
            if node_id not in self.input_nodes:
                raise ValueError(f"'{node_id}' is not a valid input node")
        
        # Initialize message storage
        node_outputs = {}  # Map of node_id -> port_id -> output data
        
        # Create initial messages for input nodes
        for node_id, content in inputs.items():
            # Create a user message for each input
            message = Message(role="user", content=content)
            node_outputs[node_id] = {
                "default": {
                    "content": content, 
                    "message": message, 
                    "metadata": {}
                }
            }
        
        # Process nodes in topological sort order
        for node_id in nx.topological_sort(self.graph):
            try:
                # Skip if this is an input node that we've already processed
                if node_id in inputs:
                    continue
                
                self.log(f"Processing node: {node_id}")
                
                # Get the node instance
                node = self.node_instances[node_id]
                
                # Collect inputs from predecessor nodes based on edge connections
                node_inputs = []
                
                # For each predecessor
                for pred_id in self.graph.predecessors(node_id):
                    # Skip if the predecessor hasn't been processed yet
                    if pred_id not in node_outputs:
                        continue
                    
                    # Get all edges between this predecessor and the current node
                    edges = self.graph.get_edge_data(pred_id, node_id)
                    
                    # For each edge
                    for edge_key, edge_data in edges.items():
                        from_port = edge_data["from_port"]
                        to_port = edge_data["to_port"]
                        
                        # Check if the source port exists
                        if from_port in node_outputs[pred_id]:
                            port_output = node_outputs[pred_id][from_port]
                            
                            # Create message from this output
                            message = Message(
                                role="user", 
                                content=port_output["content"],
                                metadata=port_output.get("metadata", {})
                            )
                            
                            # Add to inputs with source information and port
                            node_inputs.append((pred_id, from_port, message))
                
                # Process the node
                port_outputs = node.process(node_inputs, context)
                
                # Store each port output
                output_data = {}
                for port_id, port_output in port_outputs.items():
                    # Create a message from the output if not already present
                    if "message" not in port_output:
                        message = Message(
                            role="assistant", 
                            content=port_output["content"],
                            metadata=port_output.get("metadata", {})
                        )
                        port_output["message"] = message
                    
                    # Store with port information
                    output_data[port_id] = port_output
                
                node_outputs[node_id] = output_data
                self.log(f"Node {node_id} processed with output ports: {list(output_data.keys())}")
                
            except Exception as e:
                error_message = f"Error processing node {node_id}: {str(e)}"
                self.log(f"ERROR: {error_message}")
                self.log(traceback.format_exc())
                
                # Create error output for this node
                node_outputs[node_id] = {
                    "error": {
                        "content": error_message,
                        "metadata": {"error": True, "exception": str(e)},
                        "message": Message(role="assistant", content=error_message)
                    }
                }
        
        # Collect outputs
        results = {}
        for node_id in self.output_nodes:
            if node_id in node_outputs:
                # Include complete port data for the output nodes
                results[node_id] = node_outputs[node_id]
        
        return results
    
    def save(self, filepath: str) -> None:
        """
        Save the workflow configuration to a file
        
        Args:
            filepath: Path to save the configuration
        """
        # Create a serializable representation of the workflow
        data = {
            "nodes": [
                {
                    "id": node_id,
                    "is_input": node_id in self.input_nodes,
                    "is_output": node_id in self.output_nodes,
                    "agent": {
                        "name": self.node_instances[node_id].agent.name,
                        "description": self.node_instances[node_id].agent.description,
                        "system_prompt": self.node_instances[node_id].agent.system_prompt,
                        "config": self.node_instances[node_id].agent.config
                    }
                }
                for node_id in self.graph.nodes
            ],
            "edges": [
                {
                    "from": u,
                    "to": v,
                    "from_port": data["from_port"],
                    "to_port": data["to_port"],
                    "metadata": data.get("metadata", {})
                }
                for u, v, key, data in self.graph.edges(data=True, keys=True)
            ]
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str, llm_client: LLMClient, debug: bool = False) -> 'WorkflowEngine':
        """
        Load a workflow configuration from a file
        
        Args:
            filepath: Path to the configuration file
            llm_client: LLM client to use for the agents
            debug: Whether to enable debug mode
            
        Returns:
            Initialized WorkflowEngine
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Create a new workflow engine
        workflow = cls(debug=debug)
        
        # Add nodes
        for node_data in data["nodes"]:
            # Create the agent
            agent_data = node_data["agent"]
            agent = Agent(
                name=agent_data["name"],
                description=agent_data["description"],
                llm_client=llm_client,
                system_prompt=agent_data["system_prompt"],
                config=agent_data["config"]
            )
            
            # Create the node
            node = Node(id=node_data["id"], agent=agent)
            
            # Add to the workflow
            workflow.add_node(
                node=node,
                is_input=node_data["is_input"],
                is_output=node_data["is_output"]
            )
        
        # Add edges
        for edge_data in data["edges"]:
            workflow.add_edge(
                from_node_id=edge_data["from"],
                to_node_id=edge_data["to"],
                from_port=edge_data.get("from_port", "default"),
                to_port=edge_data.get("to_port", "input"),
                metadata=edge_data.get("metadata", {})
            )
        
        return workflow