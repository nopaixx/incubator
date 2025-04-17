# 🧠 AgentFlow

## Sistema Agnóstico de Flujos de Trabajo Multi-Agente

AgentFlow es un framework flexible y potente que permite diseñar y ejecutar flujos de trabajo complejos con múltiples agentes de lenguaje (LLM) que pueden comunicarse a través de cualquier topología de grafo dirigido.

![AgentFlow Concept](https://via.placeholder.com/800x400?text=AgentFlow+Concept)

## 🌟 Características Principales

- **🔄 Arquitectura basada en grafos**: Define cualquier flujo de comunicación entre agentes usando grafos dirigidos
- **🔀 Múltiples entradas/salidas**: Cada nodo puede tener múltiples puertos de entrada/salida para un enrutamiento sofisticado
- **🤝 Diálogos iterativos**: Permite que los agentes mantengan conversaciones entre sí para refinar ideas y soluciones
- **🧩 Modular y extensible**: Diseñado para ser fácilmente ampliable con nuevos tipos de agentes, LLMs o funcionalidades
- **💾 Serializable**: Capacidad para guardar y cargar configuraciones completas de flujos de trabajo
- **🔍 Validación automática**: Verificación de la coherencia y viabilidad del grafo de flujo

## 🚀 Instalación

```bash
# Clonar el repositorio
git clone https://github.com/your-username/incubator.git
cd incubator

# Instalar utilizando Poetry
poetry install

# Activar el entorno virtual
poetry shell
```

### Requisitos principales
- Python 3.11+
- NetworkX
- Anthropic Python SDK
- Python-dotenv
- Pandas, NumPy, Matplotlib (para el módulo de estrategias cuantitativas)

Todas las dependencias están especificadas en el archivo `pyproject.toml` y serán instaladas automáticamente por Poetry.

## 🛠️ Uso Básico

```python
from incubator.wf.wf import WorkflowEngine
from incubator.wf.node import Node
from incubator.agents.agent import Agent
from incubator.llm.antropic import AnthropicClient
from incubator.messages.message import Message

# Crear cliente LLM
llm_client = AnthropicClient(api_key="tu-api-key")

# Crear agentes
agent1 = Agent(
    name="AgenteAnalista",
    description="Analiza datos y genera insights",
    llm_client=llm_client,
    system_prompt="Eres un analista experto que examina datos y extrae conclusiones significativas."
)

agent2 = Agent(
    name="AgentePresentador",
    description="Convierte análisis técnicos en presentaciones claras",
    llm_client=llm_client,
    system_prompt="Eres un experto en comunicación que transforma análisis complejos en explicaciones claras y concisas."
)

# Crear nodos del flujo de trabajo
node1 = Node(id="analizador", agent=agent1)
node2 = Node(id="presentador", agent=agent2)

# Configurar flujo de trabajo
workflow = WorkflowEngine()
workflow.add_node(node1, is_input=True, is_output=False)
workflow.add_node(node2, is_input=False, is_output=True)
workflow.add_edge("analizador", "presentador")

# Ejecutar flujo de trabajo
results = workflow.execute({"analizador": "Análisis de los datos del mercado financiero de los últimos 6 meses..."})

# Obtener resultado final
final_output = results["presentador"]["default"]
print(final_output)
```

## 📋 Patrones de Flujo de Trabajo

AgentFlow soporta diversos patrones de flujo de trabajo para casos de uso complejos:

### Flujo Lineal
```python
workflow.add_node(node1, is_input=True)
workflow.add_node(node2)
workflow.add_node(node3, is_output=True)

workflow.add_edge("node1", "node2")
workflow.add_edge("node2", "node3")
```

### Flujo con Clasificación y Routing
```python
# El clasificador dirige diferentes tipos de contenido a procesadores especializados
workflow.add_edge("clasificador", "procesador_técnico", from_port="técnico")
workflow.add_edge("clasificador", "procesador_creativo", from_port="creativo")
workflow.add_edge("clasificador", "procesador_resumen", from_port="resumen")
```

### Diálogo Iterativo entre Agentes
```python
from incubator.wf.dialogenodo import IterativeDialogNode

# Crear nodo de diálogo entre agentes que pueden iterar varias veces
dialogo_node = IterativeDialogNode(
    id="dialogo_estrategia",
    agent_a=agente_explorador,
    agent_b=agente_curador,
    max_iterations=3
)

workflow.add_node(dialogo_node, is_input=True, is_output=True)
```

## 💡 Casos de Uso Incluidos

El repositorio incluye ejemplos prácticos completos:

### 1. Incubadora de Estrategias Cuantitativas

Un sistema avanzado para desarrollar, evaluar e implementar estrategias de trading algorítmico:

```python
from incubator.legacy.part1 import Conversacion
from incubator.legacy.part2 import ImplementacionEstrategia

# Generar idea de estrategia
conversacion = Conversacion(semilla="Estrategia de arbitraje en criptomonedas", max_turnos=3)
mensajes, idea_final = conversacion.iniciar()

# Implementar y evaluar la estrategia
implementacion = ImplementacionEstrategia(idea=idea_final, max_iteraciones=3)
mensajes, codigo_final, resultados, veredicto = implementacion.iniciar()
```

### 2. Sistema de Análisis Multi-faceta

Un framework para descomponer problemas complejos en múltiples ángulos de análisis especializados.

## 🧑‍💻 Personalización

### Crear Agentes Personalizados

```python
class AnalistaFinanciero(Agent):
    def process(self, messages, context=None):
        # Lógica personalizada para análisis financiero
        processed_messages = self._preprocess_financial_data(messages)
        response = super().process(processed_messages, context)
        return self._format_financial_insights(response)
        
    def _preprocess_financial_data(self, messages):
        # Conversión de datos financieros para mejor procesamiento
        # ...
        
    def _format_financial_insights(self, response):
        # Formateo para destacar insights financieros clave
        # ...
```

### Procesadores Personalizados para Nodos

```python
def procesador_entrada_financiera(inputs):
    """Procesa y normaliza datos financieros de múltiples fuentes"""
    processed_messages = []
    for source_id, port, message in inputs:
        # Normalización y enriquecimiento de datos financieros
        # ...
        processed_messages.append(message)
    return processed_messages

nodo_analisis = Node(
    id="analisis_financiero",
    agent=analista_financiero,
    input_processor=procesador_entrada_financiera,
    output_processor=procesador_salida_financiera
)
```

## 📦 Guardar y Cargar Flujos de Trabajo

AgentFlow permite persistir y reutilizar flujos de trabajo completos:

```python
# Guardar configuración de flujo de trabajo
workflow.save("workflows/estrategia_trading.json")

# Cargar configuración posteriormente
nuevo_workflow = WorkflowEngine.load("workflows/estrategia_trading.json", llm_client)
```

## 🔍 Estructura del Repositorio

```
incubator/
├── agents/             # Implementaciones de agentes
│   ├── agent.py        # Clase base para todos los agentes
│   └── multioutputagent.py  # Agente con capacidad multi-salida
├── llm/                # Clientes para diferentes LLMs
│   ├── llmclient.py    # Clase base para clientes LLM
│   └── antropic.py     # Cliente específico para Anthropic Claude
├── messages/           # Manejo de mensajes
│   └── message.py      # Clase para representar mensajes
├── wf/                 # Sistema de flujo de trabajo
│   ├── node.py         # Implementación de nodos
│   ├── wf.py           # Motor de flujo de trabajo
│   ├── dialogenodo.py  # Nodo especializado en diálogos
│   └── dialogecontroler.py  # Controlador de diálogos iterativos
└── legacy/             # Implementaciones específicas
    ├── part1.py        # Generación de ideas (Explorador-Curador)
    ├── part2.py        # Implementación de estrategias
    └── part2_images.py # Evaluación con imágenes
```

## 📚 Documentación Extendida

Para documentación más detallada sobre la arquitectura y los componentes, consulta la [Wiki del proyecto](https://github.com/your-username/incubator/wiki).

## 🛣️ Roadmap

- [ ] Soporte para más proveedores de LLM (OpenAI, Gemini, etc.)
- [ ] Interfaz de usuario para diseño visual de flujos
- [ ] Herramientas de monitoreo y análisis de rendimiento
- [ ] Mejora de capacidades de persistencia y escalabilidad

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, sigue estos pasos:

1. Haz fork del repositorio
2. Crea una rama para tu característica (`git checkout -b feature/amazing-feature`)
3. Haz commit de tus cambios (`git commit -m 'Add some amazing feature'`)
4. Haz push a la rama (`git push origin feature/amazing-feature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 📬 Contacto

Angel Lordan - [@tu_twitter](https://twitter.com/tu_twitter) - nopaixx@gmail.com

Link del proyecto: [https://github.com/your-username/incubator](https://github.com/your-username/incubator)