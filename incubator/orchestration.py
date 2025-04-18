from typing import Any, Dict, List, Tuple, Union, Optional
from incubator.messages.message import Message
from incubator.agents.agent import Agent
from incubator.formatters import InputFormatter, DefaultFormatter
import rich
from rich.console import Console
from rich.panel import Panel
import networkx as nx
import matplotlib.pyplot as plt
import os
from datetime import datetime

console = Console()

class OrchestrationPipeline:
    """
    Orchestrates a configurable sequence of agent calls, supporting alternating sequences,
    multiple inputs per step, and full conversation tracking. Logs live interaction.
    """

    def __init__(self, visualize_graph: bool = False, debug_mode: bool = False):
        self.agents: Dict[str, Agent] = {}
        # Each step: (agent_name, iterations, input_from: List[str] or "last")
        self.steps: List[Tuple[str, int, Union[str, List[str]]]] = []
        self.last_outputs: Dict[str, Any] = {}
        self.graph = nx.DiGraph()  # Directed graph for visualization
        self.visualize_graph = visualize_graph
        self.debug_mode = debug_mode
        self.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Formatters configuration
        self.default_formatter = DefaultFormatter()
        self.custom_formatters: Dict[Tuple[frozenset, str], InputFormatter] = {}

    def add_node(self, name: str, agent: Agent) -> None:
        """Add an agent to the pipeline"""
        if name in self.agents:
            raise ValueError(f"Agent name '{name}' already registered.")
        self.agents[name] = agent
        # Add node to the graph
        self.graph.add_node(name, agent_type=agent.__class__.__name__)

    def add_edge(self, to_node: str, iterations: int = 1, input_from: Union[str, List[str]] = "last") -> None:
        """Add a connection between agents in the pipeline"""
        if to_node not in self.agents:
            raise ValueError(f"Agent '{to_node}' not registered.")
        if iterations < 1:
            raise ValueError("Iterations must be >= 1")
        
        # Add the step
        self.steps.append((to_node, iterations, input_from))
        
        # Add edges to the graph
        if input_from == "last":
            # This is a placeholder edge, will be replaced with actual edges during execution
            self.graph.add_edge("_last_", to_node, iterations=iterations)
        else:
            # Add edges from all source nodes
            sources = input_from if isinstance(input_from, list) else [input_from]
            for source in sources:
                if source in self.agents:
                    self.graph.add_edge(source, to_node, iterations=iterations)

    def add_alternating(self, first: str, second: str, rounds: int) -> None:
        """Add alternating calls between two agents for a specified number of rounds"""
        if first not in self.agents or second not in self.agents:
            raise ValueError("Both first and second agents must be registered.")
        
        for _ in range(rounds):
            self.steps.append((first, 1, "last"))
            self.steps.append((second, 1, "last"))
            
            # Add edges to the graph
            self.graph.add_edge(first, second, iterations=1)
            # If not the last round, add edge back from second to first
            if _ < rounds - 1:
                self.graph.add_edge(second, first, iterations=1)

    def register_formatter(self, formatter: InputFormatter, sources: List[str], destination: str) -> None:
        """
        Register a custom formatter for specific source-to-destination connections
        
        Args:
            formatter: The formatter to use
            sources: List of source agent names
            destination: Destination agent name
        """
        if not all(source in self.agents for source in sources):
            missing = [source for source in sources if source not in self.agents]
            raise ValueError(f"Source agents not registered: {missing}")
        
        if destination not in self.agents:
            raise ValueError(f"Destination agent '{destination}' not registered.")
        
        # Use frozenset for sources to allow unordered comparison
        key = (frozenset(sources), destination)
        self.custom_formatters[key] = formatter

    def set_default_formatter(self, formatter: InputFormatter) -> None:
        """
        Set the default formatter to use when no custom formatter matches
        
        Args:
            formatter: The formatter to use as default
        """
        self.default_formatter = formatter

    def get_formatter(self, sources: List[str], destination: str) -> InputFormatter:
        """
        Get the appropriate formatter for the given sources and destination
        
        Args:
            sources: List of source agent names
            destination: Destination agent name
            
        Returns:
            The formatter to use
        """
        # Check if we have a specific formatter for this exact combination
        key = (frozenset(sources), destination)
        if key in self.custom_formatters:
            return self.custom_formatters[key]
        
        # Fall back to the default formatter
        return self.default_formatter

    def format_combined_inputs(self, inputs: List[Message], destination: str) -> str:
        """
        Format multiple input messages using the appropriate formatter
        
        Args:
            inputs: List of messages to format
            destination: The destination agent name
            
        Returns:
            Formatted input string
        """
        # Extract source names from the inputs
        sources = [msg.role for msg in inputs if msg.role != "user"]
        
        # Get the appropriate formatter
        formatter = self.get_formatter(sources, destination)
        
        # Format the inputs
        return formatter.format(inputs)

    def visualize(self, output_path: str = "agent_graph.png") -> None:
        """Generate and save a visualization of the agent pipeline"""
        if not self.visualize_graph:
            return
            
        # Create a clean graph for visualization (remove _last_ placeholder)
        vis_graph = self.graph.copy()
        if "_last_" in vis_graph:
            vis_graph.remove_node("_last_")
            
        # Set up plot
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(vis_graph)
        
        # Draw nodes with colors based on agent type
        agent_types = nx.get_node_attributes(vis_graph, 'agent_type')
        node_colors = [hash(agent_types.get(node, "default")) % 256 / 256.0 for node in vis_graph.nodes()]
        
        nx.draw_networkx_nodes(vis_graph, pos, node_color=node_colors, node_size=800, alpha=0.8)
        nx.draw_networkx_labels(vis_graph, pos, font_weight='bold')
        
        # Draw edges with iteration counts as labels
        edge_labels = {(u, v): f"x{d['iterations']}" for u, v, d in vis_graph.edges(data=True) if 'iterations' in d}
        nx.draw_networkx_edges(vis_graph, pos, width=1.5, arrowsize=20)
        nx.draw_networkx_edge_labels(vis_graph, pos, edge_labels=edge_labels)
        
        plt.title("Agent Pipeline Flow")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        console.print(f"[bold green]Pipeline visualization saved to {output_path}[/bold green]")
        plt.close()

    def debug_log(self, message: str) -> None:
        """Log debug information if debug mode is enabled"""
        if self.debug_mode:
            console.print(f"[dim][DEBUG] {message}[/dim]")

    def run(self, seed: Any) -> Tuple[Any, List[Tuple[str, Any]], List[Message]]:
        """Execute the pipeline with the given seed input"""
        history: List[Tuple[str, Any]] = []
        conversation: List[Message] = []
        current = seed
        conversation.append(Message(role="user", content=str(seed)))
        console.rule("[bold blue]Conversación Iniciada")
        console.print(f"[bold magenta]user:[/bold magenta] {seed}\n")
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Track which agent was last to feed into "last" connections
        last_agent = "user"
        
        for step_idx, (agent_name, iterations, input_from) in enumerate(self.steps):
            agent = self.agents[agent_name]
            
            # Update the graph for "last" connections
            if input_from == "last":
                # If the graph has a placeholder edge from _last_ to this agent
                if self.graph.has_edge("_last_", agent_name):
                    # Remove the placeholder edge
                    self.graph.remove_edge("_last_", agent_name)
                    # Add the actual edge from the last agent
                    self.graph.add_edge(last_agent, agent_name)
            
            for iter_idx in range(iterations):
                self.debug_log(f"Step {step_idx+1}.{iter_idx+1}: {agent_name} processing input from {input_from}")
                
                # Prepare inputs based on source
                if input_from == "last":
                    # Use the last output as input
                    inputs = [Message(role="user", content=str(current))]
                else:
                    # Collect inputs from specified sources
                    sources = input_from if isinstance(input_from, list) else [input_from]
                    inputs = []
                    
                    for src in sources:
                        if src in self.last_outputs:
                            # Create a message from this source agent
                            msg_content = str(self.last_outputs[src])
                            # If we have multiple sources, indicate which source this is from
                            if len(sources) > 1:
                                inputs.append(Message(role=src, content=msg_content))
                            else:
                                inputs.append(Message(role="user", content=msg_content))
                        else:
                            self.debug_log(f"Warning: Source '{src}' not found in last_outputs")
                    
                    # If we have multiple inputs, combine them with clear separation
                    if len(inputs) > 1:
                        combined_content = self.format_combined_inputs(inputs, agent_name)
                        inputs = [Message(role="user", content=combined_content)]

                # Mostrar inputs
                console.rule(f"[bold green]{agent_name.upper()} recibe inputs")
                for msg in inputs:
                    console.print(Panel.fit(f"{msg.content}", title=f"{msg.role} → {agent_name}", style="dim"))

                # Validar inputs antes de llamar al agente
                if not inputs:
                    console.print(f"[bold red]⚠️ {agent_name} no recibió inputs. Se omite esta llamada.[/bold red]")
                    # Mantener el contenido actual sin cambio
                    response = current
                elif any(not (msg.content and msg.content.strip()) for msg in inputs):
                    console.print(f"[bold red]⚠️ {agent_name} recibió inputs vacíos. Se omite esta llamada.[/bold red]")
                    # Mantener el contenido actual sin cambio
                    response = current
                else:
                    try:
                        # Process the inputs with the agent
                        response = agent.process(inputs)
                        
                        # Log the full conversation to a file
                        log_file = f"logs/conversation_{self.conversation_id}.txt"
                        with open(log_file, "a", encoding="utf-8") as f:
                            f.write(f"\n\n--- {agent_name} (STEP {step_idx+1}.{iter_idx+1}) ---\n")
                            for idx, msg in enumerate(inputs):
                                f.write(f"INPUT {idx+1} ({msg.role}):\n{msg.content}\n\n")
                            f.write(f"RESPONSE:\n{response}\n")
                            
                    except Exception as e:
                        console.print(f"[bold red]❌ Error en {agent_name}: {str(e)}[/bold red]")
                        console.print_exception()
                        response = f"Error en el agente {agent_name}: {str(e)}"

                # Mostrar output
                console.rule(f"[bold yellow]{agent_name.upper()} genera output")
                console.print(Panel.fit(f"{response}", title=f"{agent_name}", style="bold"))

                # Update history and state
                history.append((agent_name, response))
                conversation.append(Message(role=agent_name, content=response))
                current = response
                self.last_outputs[agent_name] = response
                last_agent = agent_name

        console.rule("[bold red]Conversación Finalizada")
        
        # Generate visualization if enabled
        if self.visualize_graph:
            self.visualize(f"logs/pipeline_{self.conversation_id}.png")
            
        return current, history, conversation


# Example usage that can be uncommented
"""
if __name__ == "__main__":
    from dotenv import load_dotenv
    from incubator.agents.multioutputagent import MultiOutputAgent
    from incubator.agents.agent import Agent
    from incubator.llm.antropic import AnthropicClient
    from incubator.formatters import IdeadorCuradorFormatter, CodigoRevisorFormatter

    load_dotenv()
    llm = AnthropicClient()

    ideador = MultiOutputAgent(
        name="ideador",
        description="Genera ideas desde una semilla.",
        llm_client=llm,
        system_prompt="A partir de la semilla, genera dos ideas etiquetadas ## [idea1] y ## [idea2]."
    )
    curador = Agent(
        name="curador",
        description="Refina las ideas previas.",
        llm_client=llm,
        system_prompt="Toma la última idea y mejora su claridad y precisión."
    )
    sintetizador = Agent(
        name="sintetizador",
        description="Sintetiza ideas en un concepto final.",
        llm_client=llm,
        system_prompt="Sintetiza las propuestas en una idea final coherente."
    )
    desarrollador = Agent(
        name="desarrollador",
        description="Genera un boceto de implementación de la idea en Python.",
        llm_client=llm,
        system_prompt="Toma la idea final y genera un boceto de código Python que implemente esa idea."
    )
    revisor_codigo = Agent(
        name="revisor_codigo",
        description="Revisa y mejora el código generado.",
        llm_client=llm,
        system_prompt="Revisa el código Python y sugiere mejoras, corrige bugs y optimiza."
    )

    pipeline = OrchestrationPipeline(visualize_graph=True, debug_mode=True)
    pipeline.add_node("ideador", ideador)
    pipeline.add_node("curador", curador)
    pipeline.add_node("sintetizador", sintetizador)
    pipeline.add_node("desarrollador", desarrollador)
    pipeline.add_node("revisor_codigo", revisor_codigo)

    # Registrar formatters personalizados para casos específicos
    pipeline.register_formatter(
        IdeadorCuradorFormatter(), 
        sources=["ideador", "curador"], 
        destination="sintetizador"
    )
    
    pipeline.register_formatter(
        CodigoRevisorFormatter(),
        sources=["sintetizador", "desarrollador"],
        destination="revisor_codigo"
    )

    # Fase 1: ideación y curación
    pipeline.add_alternating("ideador", "curador", rounds=2)
    
    # Fase 2: sintetizar 
    pipeline.add_edge("sintetizador", iterations=1, input_from=["ideador", "curador"])

    # Fase 3: desarrollo 
    pipeline.add_edge("desarrollador", iterations=1, input_from="sintetizador")

    # Fase 4: revisión de código con input combinado
    pipeline.add_edge("revisor_codigo", iterations=1, input_from=["sintetizador", "desarrollador"])
    pipeline.add_alternating("desarrollador", "revisor_codigo", rounds=1)

    final, history, convo = pipeline.run("estrategia de inversion para pequeños ahorradores")

    console.print("\n[bold cyan]=== Código final revisado ===[/bold cyan]")
    console.print(final)
"""