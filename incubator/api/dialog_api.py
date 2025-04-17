import os
import json
import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime

from incubator.llm.antropic import AnthropicClient
from incubator.agents.agent import Agent
from incubator.wf.enhanced_workflow import EnhancedWorkflowEngine
from incubator.wf.enhanced_dialognode import EnhancedDialogNode
from incubator.messages.message import Message

class DialogWorkflowAPI:
    """
    API simplificada para crear y ejecutar workflows de diálogo
    con validación y corrección mejorada de resultados.
    """
    
    def __init__(self, api_key: Optional[str] = None,
                 log_level: str = "INFO", 
                 save_results: bool = True,
                 results_dir: str = "resultados",
                 verbose: bool = False):
        """
        Inicializa la API de workflow de diálogo
        
        Args:
            api_key: Clave API opcional (se usará la variable de entorno si no se proporciona)
            log_level: Nivel de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            save_results: Si se deben guardar los resultados automáticamente
            results_dir: Directorio base para guardar resultados
            verbose: Si se debe mostrar información detallada durante la ejecución
        """
        # Configuración de API
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("No se encontró clave API. Proporciona api_key o configura ANTHROPIC_API_KEY")
        
        # Cliente LLM
        self.llm_client = AnthropicClient(api_key=self.api_key)
        
        # Configuración de guardado
        self.save_results = save_results
        self.results_dir = results_dir
        if save_results:
            os.makedirs(results_dir, exist_ok=True)
        
        # Configuración de logs
        self.verbose = verbose
        log_levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        self.log_level = log_levels.get(log_level.upper(), logging.INFO)
        
        # Configurar logger
        self.logger = logging.getLogger("DialogWorkflowAPI")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s: %(message)s', datefmt='%H:%M:%S')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(self.log_level)
    
    def log(self, message: str, level: str = "info") -> None:
        """Registra un mensaje en el log"""
        if level.lower() == "debug":
            self.logger.debug(message)
        elif level.lower() == "info":
            self.logger.info(message)
        elif level.lower() == "warning":
            self.logger.warning(message)
        elif level.lower() == "error":
            self.logger.error(message)
        elif level.lower() == "critical":
            self.logger.critical(message)
        
        if self.verbose and level.lower() != "debug":
            print(f"[DialogWorkflowAPI] {message}")
    
    def create_dialog_workflow(self, 
                              domain: str, 
                              workflow_id: Optional[str] = None,
                              max_iterations: int = 3,
                              content_markers: Optional[Dict[str, str]] = None,
                              agent_a_name: str = "Generator",
                              agent_b_name: str = "Evaluator") -> Tuple[EnhancedWorkflowEngine, EnhancedDialogNode]:
        """
        Crea un workflow de diálogo entre dos agentes
        
        Args:
            domain: Dominio específico para configurar los agentes
            workflow_id: ID opcional para el workflow
            max_iterations: Máximo número de intercambios entre agentes
            content_markers: Marcadores de contenido personalizados
            agent_a_name: Nombre del primer agente
            agent_b_name: Nombre del segundo agente
            
        Returns:
            Tupla con (workflow, nodo_diálogo)
        """
        workflow_id = workflow_id or f"dialog_{domain.lower().replace(' ', '_')}_{int(datetime.now().timestamp())}"
        node_id = f"{agent_a_name.lower()}_{agent_b_name.lower()}_dialog"
        
        self.log(f"Creando workflow de diálogo para dominio: {domain}")
        self.log(f"ID de workflow: {workflow_id}, ID de nodo: {node_id}")
        
        # Crear agente A (Generador)
        generator = Agent(
            name=agent_a_name,
            description=f"Generador de contenido para {domain}",
            llm_client=self.llm_client,
            system_prompt=f"""
            Eres un generador especializado de contenido para el dominio de {domain}.
            
            Tu tarea es crear propuestas originales, detalladas y bien fundamentadas
            basadas en el input inicial. Debes:
            
            1. Analizar el contexto y requisitos proporcionados
            2. Generar contenido relevante con detalles específicos
            3. Incorporar perspectivas innovadoras y consideraciones importantes
            4. Responder constructivamente a las críticas y sugerencias del Evaluador
            
            Sé creativo, preciso y completo en tus propuestas.
            """
        )
        
        # Crear agente B (Evaluador)
        evaluator = Agent(
            name=agent_b_name,
            description=f"Evaluador de contenido para {domain}",
            llm_client=self.llm_client,
            system_prompt=f"""
            Eres un evaluador especializado de contenido para el dominio de {domain}.
            
            Tu tarea es analizar críticamente las propuestas del Generador y
            proporcionar retroalimentación constructiva para mejorarlas. Debes:
            
            1. Evaluar la calidad, originalidad y relevancia del contenido
            2. Identificar puntos fuertes y áreas de mejora
            3. Sugerir modificaciones específicas y adiciones
            4. Plantear preguntas que provoquen más desarrollo
            
            En la iteración final (o cuando consideres que el contenido está listo),
            debes presentar tu conclusión marcándola claramente con "CONCLUSIÓN:"
            seguida de tu evaluación final y cualquier recomendación adicional.
            
            Sé analítico, específico y constructivo en tus evaluaciones.
            """
        )
        
        # Definir marcadores de contenido predeterminados
        default_markers = {
            "conclusion": "CONCLUSIÓN:",
            "conclusion_alt": "CONCLUSION:",
            "recommendation": "RECOMENDACIÓN:",
            "recommendation_alt": "RECOMENDACION:",
            "final_proposal": "PROPUESTA FINAL:"
        }
        
        # Combinar con marcadores personalizados si se proporcionan
        if content_markers:
            default_markers.update(content_markers)
        
        # Definir puertos esperados
        expected_ports = [
            "default", "conversation", "summary", 
            "conclusion", "generator_final", "evaluator_final"
        ]
        
        # Crear nodo de diálogo mejorado
        dialog_node = EnhancedDialogNode(
            id=node_id,
            agent_a=generator,
            agent_b=evaluator,
            max_iterations=max_iterations,
            terminate_on_keywords=["CONCLUSIÓN:", "CONCLUSION:"],
            content_markers=default_markers,
            expected_ports=expected_ports,
            enable_auto_correction=True,
            debug_mode=self.verbose,
            verbose=self.verbose
        )
        
        # Crear workflow mejorado
        workflow = EnhancedWorkflowEngine(
            workflow_id=workflow_id,
            expected_node_outputs={node_id: expected_ports},
            enable_auto_correction=True,
            debug=self.verbose
        )
        
        # Añadir nodo al workflow
        workflow.add_node(dialog_node, is_input=True, is_output=True)
        
        return workflow, dialog_node
    
    def run_dialog(self, 
                  initial_prompt: str, 
                  domain: str = "general",
                  max_iterations: int = 3,
                  workflow_id: Optional[str] = None,
                  output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Ejecuta un diálogo completo y devuelve los resultados procesados
        
        Args:
            initial_prompt: Prompt inicial para el diálogo
            domain: Dominio específico para los agentes
            max_iterations: Máximo número de intercambios
            workflow_id: ID opcional para el workflow
            output_dir: Directorio para guardar resultados (opcional)
            
        Returns:
            Diccionario con resultados procesados
        """
        try:
            # Directorio de resultados
            output_dir = output_dir or self.results_dir
            if self.save_results:
                os.makedirs(output_dir, exist_ok=True)
            
            # Mostrar mensaje de inicio
            self.log(f"Iniciando diálogo en dominio '{domain}' con prompt: '{initial_prompt}'")
            
            # Configurar el workflow
            workflow, dialog_node = self.create_dialog_workflow(
                domain=domain,
                workflow_id=workflow_id,
                max_iterations=max_iterations
            )
            
            # Ejecutar el workflow
            self.log("Ejecutando diálogo entre agentes...")
            results = workflow.execute(
                inputs={dialog_node.id: initial_prompt},
                save_results=self.save_results,
                results_dir=output_dir
            )
            
            # Procesar y mostrar resultados
            processed_results = {}
            
            if dialog_node.id in results:
                node_outputs = results[dialog_node.id]
                
                # Procesar múltiples salidas
                for port_name, port_data in node_outputs.items():
                    processed_results[port_name] = port_data["content"]
                
                # Mostrar resumen de puertos disponibles
                if self.verbose:
                    self.log("===== PUERTOS DE SALIDA DISPONIBLES =====")
                    for port in node_outputs.keys():
                        self.log(f"- {port}")
                
                # Mostrar conclusión si está disponible
                if "conclusion" in node_outputs:
                    if self.verbose:
                        self.log("===== CONCLUSIÓN =====")
                        self.log(node_outputs["conclusion"]["content"])
                
                # Mostrar respuesta final del evaluador
                if "evaluator_final" in node_outputs:
                    if self.verbose:
                        self.log("===== EVALUACIÓN FINAL =====")
                        self.log(node_outputs["evaluator_final"]["content"])
            
            # Guardar resultados procesados
            if self.save_results:
                self._save_processed_results(processed_results, domain, output_dir)
            
            return processed_results
            
        except Exception as e:
            self.log(f"Error en la ejecución del flujo de trabajo: {str(e)}", level="error")
            import traceback
            self.log(traceback.format_exc(), level="debug")
            return {"error": str(e)}
    
    def _save_processed_results(self, results: Dict[str, Any], domain: str, output_dir: str) -> str:
        """
        Guarda los resultados procesados en archivos
        
        Args:
            results: Resultados a guardar
            domain: Dominio del diálogo
            output_dir: Directorio base para guardar resultados
            
        Returns:
            Ruta al directorio de resultados
        """
        # Crear directorio con timestamp para evitar sobrescrituras
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        domain_safe = domain.lower().replace(" ", "_")
        full_dir = os.path.join(output_dir, f"dialog_{domain_safe}_{timestamp}")
        os.makedirs(full_dir, exist_ok=True)
        
        # Guardar cada resultado en un archivo separado
        for key, content in results.items():
            # Omitir contenido no string (como diccionarios)
            if not isinstance(content, str):
                continue
                
            filename = f"{key.replace('_', '-')}.txt"
            if key == "summary" and content.startswith("{"):
                filename = "summary.json"
                
            with open(os.path.join(full_dir, filename), "w", encoding="utf-8") as f:
                f.write(content)
        
        # Guardar todos los resultados
        with open(os.path.join(full_dir, "resultados_completos.json"), "w", encoding="utf-8") as f:
            # Filtrar para solo incluir datos serializables
            serializable = {}
            for k, v in results.items():
                if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                    serializable[k] = v
                else:
                    serializable[k] = str(v)
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        
        self.log(f"Resultados guardados en: {full_dir}")
        return full_dir