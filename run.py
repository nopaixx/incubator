# File: run.py
from typing import Any, Dict, List, Tuple, Union
from dotenv import load_dotenv
import os
import argparse
import shutil
from datetime import datetime
from rich.console import Console
from rich.panel import Panel

# Importar los componentes necesarios
from incubator.messages.message import Message
from incubator.agents.agent import Agent
from incubator.agents.multioutputagent import MultiOutputAgent
from incubator.llm.antropic import AnthropicClient
from incubator.orchestration import OrchestrationPipeline
import time

console = Console()

def setup_pipeline(seed_topic: str, ideacion_rounds: int = 3, desarrollo_rounds: int = 2, visualize: bool = True, debug: bool = False):
    """Configure and run the agent pipeline"""
    # Cargar variables de entorno
    load_dotenv()
    
    # Verificar la clave API
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[bold red]Error: No se encontró ANTHROPIC_API_KEY en las variables de entorno[/bold red]")
        return
    
    # Inicializar cliente LLM
    llm = AnthropicClient(api_key=api_key)
    
    # Crear los agentes
    console.print("[bold blue]Configurando agentes...[/bold blue]")
    
    # Configuración básica común para acelerar respuestas
    config_base = {
        "max_tokens": 5000,  # Reducir el tamaño máximo para respuestas más cortas
        "temperature": 0.7,
        "use_thinking": False,  # Desactivar thinking para acelerar
    }
    
    ideador = Agent(
        name="ideador",
        description="Genera y refina una única idea innovadora",
        llm_client=llm,
        system_prompt="""
        Eres el Agente Ideador en un sistema de incubación de ideas para hedge funds cuantitativos. 
        Tu especialidad es generar estrategias algorítmicas innovadoras centradas en el S&P 500.
        Recibiras feedback de un usuario especialita curador de estrategia debes hacerle caso 
        
        IMPORTANTE: Debes generar UNA SOLA idea O USAR LAS SUGERENCIAS DEL ususario para refinar la estrategia cuantitativa por iteración.
        No es necesario cambiar la idea original a no ser que sea necesario, se trata de que puedas mejorarla con el feedback del ususario
        
        Tu función es:
        
        1. Generar UNA o MEJORA la idea de estrategia cuantitativa innovadora y detallada para operar en el S&P 500
        2. Explicar los fundamentos matemáticos/estadísticos y el funcionamiento de la estrategia
        3. Detallar las señales específicas, timeframes, factores y métricas a utilizar
        4. Destacar las ventajas potenciales de la estrategia (alpha, Sharpe, drawdown, etc.)
        5. Responder a las preguntas o sugerencias del Curador
        
        Considera aspectos como:
        - Factores de mercado (momentum, valor, volatilidad, etc.)
        - Análisis estadísticos (cointegración, regresión, clusterización)
        - Indicadores técnicos o fundamentales innovadores
        - Optimización de ejecución y manejo de costos de transacción
        - Gestión de riesgo y diversificación de la estrategia
        - Evita valores hardcodeados
        - Piensa en la autoadaptacion y modelos estadisticos o predictivos de ML
        - Evita Threshold o valores hardcodeados utiliza optimizaciones inferencias o estrategias autoadaptivas
        
        Limitaciones técnicas:
        - La estrategia debe ser implementable en Python utilizando la biblioteca yfinance para datos
        - No proporciones código, solo la lógica y metodología detallada
        - La idea pasará a un siguiente nivel de desarrollo donde será implementada
        
        Sé específico, técnico y detallado. Piensa en estrategias implementables y backtestables.
        """,
        config=config_base
    )
    
    curador = Agent(
        name="curador",
        description="Propone mejoras específicas a la idea en desarrollo",
        llm_client=llm,
        system_prompt="""
        Eres el Agente Curador en un sistema de incubación de ideas para hedge funds cuantitativos.
        Tienes amplia experiencia en evaluación de estrategias algorítmicas enfocadas en el S&P 500.
        Tu función es evaluar la estrategia propuesta por el Ideador y proponer mejoras específicas, pero no debes hacer ninguna estrategia solo proponer mejoras al ideador. Debes:
        
        1. Analizar críticamente la estrategia considerando:
           - Ratio de Sharpe esperado y robustez estadística
           - Capacidad para generar alpha verdadero (no beta disfrazado)
           - Escalabilidad y capacidad (cuánto capital puede manejar)
           - Costos de implementación y transacción
           - Exposición a factores de riesgo conocidos
           - Riesgo de sobreoptimización o data snooping
           - Factibilidad de implementación con yfinance en Python
           - Facilidad de implementar y evaluar
        
        2. Proponer mejoras específicas y técnicas como:
           - Backtest y walk forward
           - Avoid look ahead bias
           - Refinamiento de parámetros o señales
           - Mejoras en la gestión de riesgo
           - Optimización de ejecución
           - Complementos con otros factores o señales
           - Evita Threshold o valores hardcodeados utiliza optimizaciones inferencias o estrategias autoadaptivas
        
        3. Formular preguntas técnicas específicas para aclarar aspectos ambiguos
        
        Limitaciones técnicas:
        - La estrategia debe ser implementable en Python utilizando la biblioteca yfinance para datos
        - No proporciones código, solo mejoras y recomendaciones conceptuales
        - La idea pasará a un siguiente nivel de desarrollo donde será implementada
        - No importa la gestion de stocks delisted o no survied
        
        Mantén un enfoque riguroso y escéptico, como lo haría un gestor de riesgos experimentado.

        TU TAREA ES ITERAR LA IDEA Y  MOSTRAR CODIGO
        DESARROLLAR EL CODIGO ES TAREA DE OTRO AGENTE ESPECIALIDADO EN DESAROLLO
        
        """,
        config={
        "max_tokens": 5000,  # Reducir el tamaño máximo para respuestas más cortas
        "temperature": 0.1,
        "use_thinking": False,  # Desactivar thinking para acelerar
        }
    )

    sintetizador = Agent(
        name="sintetizador",
        description="Sintetiza y consolida las ideas del ideador y curador en una propuesta final",
        llm_client=llm,
        system_prompt="""Eres un sintetizador de ideas. Tu trabajo es tomar las contribuciones tanto del ideador como del curador
        y producir una versión definitiva, clara y bien estructurada de la idea.

        INSTRUCCIONES IMPORTANTES:
        - Recibirás las últimas versiones tanto del ideador como del curador.
        - Debes integrar lo mejor de ambas contribuciones.
        - Tu resultado debe ser una idea COMPLETA y FINAL, lista para ser implementada.
        - Incluye un título claro y descriptivo al inicio.
        - Estructura la idea en secciones lógicas.
        - Sé conciso pero completo. Asegúrate de que todos los aspectos importantes estén cubiertos.
        - La idea debe ser realizable con los datos de yfinance
        - No importa la gestion de stocks delisted o no survied
        - Lógica exacta de entrada/salida
        - Parámetros específicos
        - Gestión de riesgo
        - Expectativas de desempeño
        - Metricas
        - Backtest y walkfoward look ahead bias etc..
        - Consideraciones de implementación técnica

        LIMITACIONES TECNICAS
        - Solo puede ser implementado con los datos propocionados por la libreria de yfinance

        NO PROPROCIONES CODIGO solo las instrucciones tecnicas de implementacion
        
        Comienza tu respuesta con:
        "# IDEA FINAL: [TÍTULO]"
        
        Y luego organiza el contenido en secciones claras como:
        "## Descripción"
        "## Características principales"
        "## Detalles de la Implementación"
        """,
        config={
        "max_tokens": 10000,  # Reducir el tamaño máximo para respuestas más cortas
        "temperature": 0.,
        "use_thinking": False,  # Desactivar thinking para acelerar
        }
    )
    
    desarrollador = Agent(
        name="desarrollador",
        description="Implementa la idea conceptual en código Python funcional",
        llm_client=llm,
        system_prompt="""
        Eres el Agente Desarrollador en un sistema de implementación de estrategias cuantitativas para hedge funds.
        Tu especialidad es convertir ideas de estrategias en código Python funcional utilizando la biblioteca yfinance.
        
        Debes implementar la estrategia proporcionada como un programa Python completo y funcional. Tu código debe:
        
        1. Utilizar la biblioteca yfinance para obtener datos del S&P 500
        2. Implementar la lógica exacta descrita en la estrategia
        3. Generar y guardar métricas de rendimiento, gráficos y análisis
        4. Evita threholds y parametros hardcodeados, utiliza optimizacion o inferencia autoadaptativa
        5. Manejar errores y validar datos adecuadamente
        6. Estar bien documentado con comentarios claros
        7. Se implementa todos los metodos descritos incuilido backtest y walkforward 
        8- No existe el look ahead bias ni haces sesgos al utilizar datos futuros para la inferencia que no tendrias en una implemtacion real
        9- En esta estapa es importante que seas conciso en el codigo
        
        NOTA AL DESARROLLADOR
        * yfinance tiene el parametro auto_adjust=True por defecto en la version instalada
        recent example of yfinance version installed yf.download(stock, start=start_date, end=end_date)['Close'] # auto_adjust=True by default
        * No se puede usar la libreria pymc3
        * Cuando tengas que usar stock del sp500 descarga TODOS los activos de wikipedia
        * No importa la gestion de stocks delisted o no survied
        
        # Crear directorios para resultados
        os.makedirs('./artifacts/results', exist_ok=True)
        os.makedirs('./artifacts/results/figures', exist_ok=True)
        os.makedirs('./artifacts/results/data', exist_ok=True)
        
        # Configurar logging
        logging.basicConfig(
            filename='./artifacts/errors.txt',
            level=logging.ERROR,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        **Pay attention to typical pandas series errors with scalars and NaN**
        
        IMPORTANTE - FORMATO DE SALIDA:
        
        1. Todos los resultados, gráficos y archivos como metricas en csv o txt de salida deben guardarse en la carpeta './artifacts/results/'
        2. Caputara errores y guardalos dentro de la carpeta './artifacts/errors.txt'
        
        Tu código debe garantizar que:
        - Se creen todas las carpetas necesarias si no existen
        - Se guarden métricas de rendimiento (Sharpe, drawdown, etc.) en archivos CSV
        - Se generen visualizaciones (gráficos de rendimiento, señales, etc.)
        - Se manejen adecuadamente los errores y se registren en un archivo de log dentro de './artifacts/errors.txt'
        
        **provide a full error traceback into ./artifacts/error.txt for future improvements**
        
        Proporciona un programa Python completo, listo para ejecutar, que implemente la estrategia de forma óptima.
        Primero menciona las mejoras aplicadas y luego saca el programa python
        """,
        config={
        "max_tokens": 40000,  # Reducir el tamaño máximo para respuestas más cortas
        "temperature": 0.,
        "use_thinking": False,  # Desactivar thinking para acelerar
        }
    )
    
    revisor_codigo = Agent(
        name="revisor_codigo",
        description="Analiza el código Python y propone mejoras específicas",
        llm_client=llm,
        system_prompt="""Eres un revisor de código Python que analiza implementaciones y sugiere mejoras específicas.

        INSTRUCCIONES IMPORTANTES:
        - Recibiras TANTO la idea conceptual original COMO el código implementado.
        - Tu trabajo es verificar si el código cumple con los requisitos de la idea.
        - Ten en cuenta que el desarrollador solo tiene acceso a datos de la api de yfinance
        - 
        - NO reescribas todo el código, sólo proporciona comentarios específicos sobre:
          * Mencionar lineas de codigo o funciones con error o problematicas
          * Errores en el tratamiento de datos
          * Bugs o problemas potenciales
          * Mejoras de eficiencia y legibilidad
          * Alineación con la idea original si se puede
          
        - Sé CONCISO Y PRIORIZA LAS REVISIONES MAS IMPORTANTES PARA QUE PODAMOS EJECUTAR EL PROGRAMA Y EVALUAR LAS METRICAS EL PROGRAMA DEBE DE FUNCIONAR EN LA EJECUCION
        

        **Algunos errores conocidos**
        -No se puede usar la libreria pymc3
        -yfinance tiene el parametro auto_adjust=True por defecto en la version instalada recent example of yfinance version installed yf.download(stock, start=start_date, end=end_date)['Close'] # auto_adjust=True by default
        -alineacion de series temporales con pandas y numpy manejo de nans y 
        -tratamiento correcto de features y alineacion
        -evitar trampas con el walk forward test y el look ahead bias
        -no datos sintetico o inventados, ya que desvirtua el objetivo o analisis
        -evitar en la medida de lo posible threholds o valores por defecto mira si se pueden autoadaptar
        -No importa la gestion de stocks delisted o no survied
        
        Estructura tus comentarios así:
        1. ¿El código implementa correctamente la idea? (Sí/No/Parcialmente)
        2. Lista numerada de sugerencias específicas
        3. Menciona siempre las mejoras mas importantes (especialmente el look ahead bias y sobretodo que el codigo compile y funcione cuando se ejecute)
        """,
        config={
        "max_tokens": 10000,  # Reducir el tamaño máximo para respuestas más cortas
        "temperature": 0.,
        "use_thinking": False,  # Desactivar thinking para acelerar
        }
    )
    
    # Crear el pipeline
    pipeline = OrchestrationPipeline(visualize_graph=visualize, debug_mode=debug)
    
    # Añadir los agentes al pipeline
    pipeline.add_node("ideador", ideador)
    pipeline.add_node("curador", curador)
    pipeline.add_node("sintetizador", sintetizador)
    pipeline.add_node("desarrollador", desarrollador)
    pipeline.add_node("revisor_codigo", revisor_codigo)
    
    # Configurar el flujo de trabajo unificado y claro
    
    # FASE 1: CICLO DE IDEACIÓN
    # ------------------------
    # Primera idea del ideador (toma el input inicial del usuario)
    pipeline.add_edge("ideador", iterations=1, input_from="last")

    # Primera reacción del curador a la idea inicial
    pipeline.add_edge("curador", iterations=1, input_from="ideador")

    # Ciclos de refinamiento entre ideador y curador
    for i in range(ideacion_rounds-1):
        # Ideador refina basado en feedback del curador
        pipeline.add_edge("ideador", iterations=1, input_from="curador")
        # Curador ofrece nuevo feedback sobre la idea refinada
        pipeline.add_edge("curador", iterations=1, input_from="ideador")

    # FASE INTERMEDIA: SÍNTESIS DE LA IDEA FINAL
    # --------------------------------------
    # El sintetizador recibe tanto la última versión del ideador como los comentarios del curador
    pipeline.add_edge("sintetizador", iterations=1, input_from=["ideador", "curador"])

    # FASE 2: CICLO DE DESARROLLO
    # --------------------------
    # Desarrollador implementa la idea sintetizada
    pipeline.add_edge("desarrollador", iterations=1, input_from="sintetizador")
    
    # Primera revisión: analiza tanto la idea original como la implementación
    pipeline.add_edge("revisor_codigo", iterations=1, input_from=["sintetizador", "desarrollador"])

    # Ciclos de refinamiento del código
    for i in range(desarrollo_rounds-1):
        # Desarrollador mejora el código basado en la revisión
        pipeline.add_edge("desarrollador", iterations=1, input_from="revisor_codigo")
    
        # Revisor analiza las mejoras, considerando siempre la idea original
        pipeline.add_edge("revisor_codigo", iterations=1, input_from=["sintetizador", "desarrollador"])
    
    # Ejecutar el pipeline
    console.print(f"[bold green]Iniciando pipeline con semilla: '{seed_topic}'[/bold green]")
    final_output, history, conversation = pipeline.run(seed_topic)
    
    console.print("\n[bold cyan]=== Código final revisado ===[/bold cyan]")
    console.print(Panel.fit(f"{final_output}", title="Resultado Final", style="bold blue"))
    
    # Guardar los resultados en archivos
    save_output_files(history, seed_topic)
    
    return final_output, history, conversation

def save_output_files(history: List[Tuple[str, str]], seed_topic: str):
    """Guarda los resultados finales en archivos dentro de la carpeta /idea/"""
    # Crear un nombre de carpeta basado en el tema y la fecha/hora actual
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_slug = seed_topic.lower().replace(" ", "_")[:20]  # Limitamos a 20 caracteres
    folder_name = f"idea_{seed_slug}_{timestamp}"
    folder_path = os.path.join("idea", folder_name)
    
    # Crear la carpeta idea si no existe
    os.makedirs("idea", exist_ok=True)
    # Crear la carpeta específica para esta ejecución
    os.makedirs(folder_path, exist_ok=True)
    
    console.print(f"[bold green]Guardando resultados en: {folder_path}[/bold green]")
    
    # Buscar la última salida del sintetizador para la idea final
    idea_final = None
    for agent_name, content in history:
        if agent_name == "sintetizador":
            idea_final = content
    
    # Buscar la última salida del desarrollador para el código
    codigo_final = None
    for agent_name, content in reversed(history):  # Buscamos desde el final
        if agent_name == "desarrollador":
            codigo_final = content
            break
    
    # Guardar la idea final
    if idea_final:
        with open(os.path.join(folder_path, "idea.txt"), "w", encoding="utf-8") as f:
            f.write(idea_final)
        console.print("[bold green]✓[/bold green] Idea guardada en idea.txt")
    else:
        console.print("[bold red]✗[/bold red] No se encontró la idea final del sintetizador")
    
    # Guardar el código final
    if codigo_final:
        # Extraer solo el código Python (asumiendo que está entre bloques de código)
        import re
        code_blocks = re.findall(r'```python\n(.*?)```', codigo_final, re.DOTALL)
        
        if code_blocks:
            # Usar el bloque de código más largo (probablemente el principal)
            codigo_python = max(code_blocks, key=len)
            with open(os.path.join(folder_path, "codigo_python.py"), "w", encoding="utf-8") as f:
                f.write(codigo_python)
            console.print("[bold green]✓[/bold green] Código guardado en codigo_python.py")
        else:
            # Si no encontramos bloques de código delimitados, guardamos todo el contenido
            with open(os.path.join(folder_path, "codigo_python.py"), "w", encoding="utf-8") as f:
                f.write(codigo_final)
            console.print("[bold yellow]⚠[/bold yellow] No se encontraron bloques de código Python delimitados. Se guardó el contenido completo.")
    else:
        console.print("[bold red]✗[/bold red] No se encontró el código final del desarrollador")
    
    # Guardar también un archivo con el historial completo para referencia
    with open(os.path.join(folder_path, "historial_completo.txt"), "w", encoding="utf-8") as f:
        for agent_name, content in history:
            f.write(f"\n\n{'='*50}\n{agent_name.upper()}\n{'='*50}\n\n")
            f.write(content)
    console.print("[bold green]✓[/bold green] Historial completo guardado en historial_completo.txt")

if __name__ == "__main__":
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Ejecutar pipeline de agentes para generación de código")
    parser.add_argument("seed", type=str, help="Tema semilla para iniciar el proceso creativo")
    parser.add_argument("--ideacion", type=int, default=3, help="Número de ciclos en fase de ideación (default: 3)")
    parser.add_argument("--desarrollo", type=int, default=2, help="Número de ciclos en fase de desarrollo (default: 2)")
    parser.add_argument("--no-visualize", action="store_false", dest="visualize", help="Desactivar visualización del grafo")
    parser.add_argument("--debug", action="store_true", help="Activar modo debug con mensajes adicionales")
    
    args = parser.parse_args()
    
    # Ejecutar el pipeline
    setup_pipeline(
        args.seed, 
        ideacion_rounds=args.ideacion, 
        desarrollo_rounds=args.desarrollo,
        visualize=args.visualize, 
        debug=args.debug
    )