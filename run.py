# File: incubator/orchestration.py
from typing import Any, Dict, List, Tuple, Union
from incubator.messages.message import Message
from incubator.agents.agent import Agent
import rich
from rich.console import Console
from rich.panel import Panel

console = Console()

class OrchestrationPipeline:
    """
    Orchestrates a configurable sequence of agent calls, supporting alternating sequences,
    multiple inputs per step, and full conversation tracking. Logs live interaction.
    """

    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        # Each step: (agent_name, iterations, input_from: List[str] or "last")
        self.steps: List[Tuple[str, int, Union[str, List[str]]]] = []
        self.last_outputs: Dict[str, Any] = {}

    def add_node(self, name: str, agent: Agent) -> None:
        if name in self.agents:
            raise ValueError(f"Agent name '{name}' already registered.")
        self.agents[name] = agent

    def add_edge(self, to_node: str, iterations: int = 1, input_from: Union[str, List[str]] = "last") -> None:
        if to_node not in self.agents:
            raise ValueError(f"Agent '{to_node}' not registered.")
        if iterations < 1:
            raise ValueError("Iterations must be >= 1")
        self.steps.append((to_node, iterations, input_from))

    def add_alternating(self, first: str, second: str, rounds: int) -> None:
        if first not in self.agents or second not in self.agents:
            raise ValueError("Both first and second agents must be registered.")
        for _ in range(rounds):
            self.steps.append((first, 1, "last"))
            self.steps.append((second, 1, "last"))

    def run(self, seed: Any) -> Tuple[Any, List[Tuple[str, Any]], List[Message]]:
        history: List[Tuple[str, Any]] = []
        conversation: List[Message] = []
        current = seed
        conversation.append(Message(role="user", content=str(seed)))
        console.rule("[bold blue]Conversación Iniciada")
        console.print(f"[bold magenta]user:[/bold magenta] {seed}\n")

        for agent_name, iterations, input_from in self.steps:
            agent = self.agents[agent_name]
            for _ in range(iterations):
                if input_from == "last":
                    inputs = [Message(role="user", content=str(current))]
                else:
                    sources = input_from if isinstance(input_from, list) else [input_from]
                    inputs = [Message(role=src, content=str(self.last_outputs[src])) for src in sources if src in self.last_outputs]

                # Mostrar inputs
                console.rule(f"[bold green]{agent_name.upper()} recibe inputs")
                for msg in inputs:
                    console.print(Panel.fit(f"{msg.content}", title=f"{msg.role} → {agent_name}", style="dim"))

                # Validar inputs antes de llamar al agente
                if not inputs or any(not (msg.content and msg.content.strip()) for msg in inputs):
                    console.print(f"[bold red]⚠️ {agent_name} recibió inputs vacíos o inválidos. Se omite esta llamada.[/bold red]")
                    # Mantener el contenido actual sin cambio
                    response = current
                else:
                    response = agent.process(inputs)

                # Mostrar output
                console.rule(f"[bold yellow]{agent_name.upper()} genera output")
                console.print(Panel.fit(f"{response}", title=f"{agent_name}", style="bold"))

                history.append((agent_name, response))
                conversation.append(Message(role=agent_name, content=response))
                current = response
                self.last_outputs[agent_name] = response

        console.rule("[bold red]Conversación Finalizada")
        return current, history, conversation


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    from incubator.agents.multioutputagent import MultiOutputAgent
    from incubator.agents.agent import Agent
    from incubator.llm.antropic import AnthropicClient

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

    pipeline = OrchestrationPipeline()
    pipeline.add_node("ideador", ideador)
    pipeline.add_node("curador", curador)
    pipeline.add_node("desarrollador", desarrollador)
    pipeline.add_node("revisor_codigo", revisor_codigo)

    # Fase 1: ideación y curación
    pipeline.add_alternating("ideador", "curador", rounds=3)

    # Fase 2: desarrollo con input del curador
    pipeline.add_edge("desarrollador", iterations=1, input_from="curador")

    # Fase 3: revisión con input combinado de desarrollador + curador
    pipeline.add_alternating("revisor_codigo", "desarrollador", rounds=3)
    pipeline.add_edge("revisor_codigo", iterations=1, input_from=["desarrollador", "curador"])

    final, history, convo = pipeline.run("estrategia de inversion")

    console.print("\n[bold cyan]=== Código final revisado ===[/bold cyan]")
    console.print(final)