# Sistema Agnóstico de Flujos de Trabajo Multi-Agente (AgentFlow)

Un framework flexible para diseñar y ejecutar flujos de trabajo con múltiples agentes LLM que pueden comunicarse a través de cualquier topología de grafo.

## Características Principales

- 🔄 **Arquitectura basada en grafos**: Define cualquier flujo de comunicación entre agentes
- 🔀 **Múltiples entradas/salidas**: Cualquier nodo puede tener múltiples puertos de entrada/salida
- 🔄 **Diálogos iterativos**: Permite que los agentes conversen entre sí para refinar ideas
- 🧩 **Modular y extensible**: Fácil de ampliar con nuevos tipos de agentes o LLMs
- 💾 **Serializable**: Guarda y carga configuraciones de flujos de trabajo
- 🔍 **Validación automática**: Verifica la coherencia del grafo de flujo

## Instalación

```bash
pip install -r requirements.txt
```

Requisitos principales:
- networkx
- anthropic (o cualquier otra API de LLM que implementes)
- python-dotenv

## Uso Básico

```python
from workflow_system import WorkflowEngine, Node, Agent, AnthropicClient

# Crear cliente LLM
llm_client = AnthropicClient(api_key="tu-api-key")

# Crear agentes
agent1 = Agent(
    name="AgentName",
    description="Agent description",
    llm_client=llm_client,
    system_prompt="Your system prompt here"
)

# Crear nodos
node1 = Node(id="node1", agent=agent1)

# Configurar flujo de trabajo
workflow = WorkflowEngine()
workflow.add_node(node1, is_input=True, is_output=False)
# ... añadir más nodos y conexiones

# Ejecutar flujo de trabajo
results = workflow.execute({"node1": "Input content here"})
```

## Patrones de Flujo de Trabajo

### Flujo de Trabajo Lineal

```python
workflow.add_node(node1, is_input=True)
workflow.add_node(node2)
workflow.add_node(node3, is_output=True)

workflow.add_edge("node1", "node2")
workflow.add_edge("node2", "node3")
```

### Flujo con Múltiples Salidas

```python
# Conectar salidas múltiples de un clasificador
workflow.add_edge("classifier", "technical_processor", from_port="technical")
workflow.add_edge("classifier", "creative_processor", from_port="creative")
workflow.add_edge("classifier", "summary_processor", from_port="summary")
```

### Diálogo Iterativo

```python
dialogue_node = IterativeDialogNode(
    id="dialogue",
    agent_a=explorer,
    agent_b=curator,
    max_iterations=3
)

workflow.add_node(dialogue_node)
```

## Ejemplos Incluidos

1. **Análisis de Estrategias Cuantitativas**: Agentes Explorador y Curador que iteran sobre estrategias de trading antes de pasarlas a un Desarrollador
2. **Análisis de Contenido Multi-faceta**: Clasificación y procesamiento especializado de contenido por diferentes tipos de análisis

## Personalización

### Crear Agentes Personalizados

```python
class MyCustomAgent(Agent):
    def process(self, messages, context=None):
        # Implementación personalizada
        return modified_output
```

### Procesadores de Entrada/Salida Personalizados

```python
def my_input_processor(inputs):
    # Combinar o transformar entradas
    return processed_messages

node = Node(
    id="custom_node",
    agent=my_agent,
    input_processor=my_input_processor,
    output_processor=my_output_processor
)
```

## Guardar y Cargar Flujos de Trabajo

```python
# Guardar configuración
workflow.save("my_workflow.json")

# Cargar configuración
loaded_workflow = WorkflowEngine.load("my_workflow.json", llm_client)
```

## Licencia

MIT
