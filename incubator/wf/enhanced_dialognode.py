import os
from typing import List, Dict, Optional, Any, Tuple, Set, Callable
import json
import logging
import re
from dotenv import load_dotenv

from incubator.agents.agent import Agent
from incubator.messages.message import Message
from incubator.wf.node import Node
from incubator.wf.dialogcontroller import IterativeDialogController
from incubator.wf.output_validator import OutputValidator

class EnhancedDialogNode(Node):
    """
    Nodo mejorado para gestionar diálogos iterativos entre dos agentes,
    con validación y corrección automática de salidas.
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
                 expected_ports: Optional[List[str]] = None,
                 enable_auto_correction: bool = True,
                 debug_mode: bool = False,
                 verbose: bool = False):
        """
        Inicializa un nodo de diálogo mejorado
        
        Args:
            id: Identificador único para el nodo
            agent_a: Primer agente del diálogo (iniciador)
            agent_b: Segundo agente del diálogo (respondedor)
            max_iterations: Máximo número de intercambios
            terminate_on_keywords: Lista de palabras clave que activarán terminación anticipada
            termination_condition: Función opcional que determina si el diálogo debe terminar antes
            input_processor: Función para procesar mensajes entrantes
            output_processor: Función para procesar la salida final
            content_markers: Diccionario de nombres de marcadores a cadenas que indican contenido específico
                            (ej. {"final_idea": "IDEA FINAL:", "code": "CODE:", "summary": "SUMMARY:"})
            expected_ports: Lista de puertos de salida que se espera que genere este nodo
            enable_auto_correction: Si se deben aplicar correcciones automáticas a las salidas
            debug_mode: Si se debe generar información adicional de depuración
            verbose: Si se debe registrar información detallada del progreso
        """
        # Crear un agente ficticio para la superclase Node
        # (no lo usaremos directamente, pero satisface los requisitos de Node)
        dummy_agent = Agent(
            name=f"{agent_a.name}_{agent_b.name}_Dialog",
            description=f"Diálogo entre {agent_a.name} y {agent_b.name}",
            llm_client=agent_a.llm_client  # Usar el mismo cliente LLM que agent_a
        )
        
        # Inicializar la superclase Node
        super().__init__(id=id, agent=dummy_agent, 
                         input_processor=input_processor, 
                         output_processor=output_processor)
        
        # Configurar el controlador de diálogo
        self.dialogue_controller = IterativeDialogController(
            agent_a=agent_a,
            agent_b=agent_b,
            max_iterations=max_iterations,
            terminate_on_keywords=terminate_on_keywords,
            termination_condition=termination_condition,
            verbose=verbose
        )
        
        # Guardar parámetros de configuración
        self.content_markers = content_markers or {
            "conclusion": "CONCLUSIÓN:",
            "conclusion_alt": "CONCLUSION:" 
        }
        self.expected_ports = expected_ports or [
            "default", "conversation", "summary", 
            f"{agent_a.name.lower()}_final", f"{agent_b.name.lower()}_final"
        ]
        self.enable_auto_correction = enable_auto_correction
        self.debug_mode = debug_mode
        self.verbose = verbose
        
        # Configurar logger
        self.logger = logging.getLogger(f'EnhancedDialogNode.{id}')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(name)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
    
    def log(self, message: str) -> None:
        """Método auxiliar para registrar información"""
        self.logger.info(message)
        if self.verbose:
            print(f"[EnhancedDialogNode.{self.id}] {message}")
    
    def _enhanced_output_processor(self, output: str, conversation: List[Message]) -> Dict[str, Dict[str, Any]]:
        """
        Procesador mejorado que proporciona salidas a través de múltiples puertos
        con validación y corrección automática
        
        Args:
            output: La salida final del diálogo
            conversation: El historial completo de la conversación
            
        Returns:
            Diccionario que mapea puertos de salida a su contenido y metadatos
        """
        # Crear un texto de conversación formateado
        formatted_conversation = []
        for msg in conversation:
            role = msg.role
            if role == "assistant" and "agent" in msg.metadata:
                role = msg.metadata["agent"]
            formatted_conversation.append(f"[{role.upper()}]: {msg.content}")
        
        conversation_text = "\n\n".join(formatted_conversation)
        
        # Crear diccionario de resultados con salida predeterminada
        results = {
            "default": {
                "content": output,
                "metadata": {"type": "final_response"}
            },
            "conversation": {
                "content": conversation_text,
                "metadata": {"type": "conversation_history", "message_count": len(conversation)}
            }
        }
        
        # Extraer información específica de marcadores
        for port_name, marker in self.content_markers.items():
            # Intentar encontrar el contenido marcado
            marker_content = None
            
            # Buscar en la salida final primero
            if marker in output:
                parts = output.split(marker, 1)
                if len(parts) > 1:
                    marker_content = parts[1].strip()
            
            # Si no se encuentra en la salida final, buscar en todo el historial de conversación
            if not marker_content:
                for msg in reversed(conversation):  # Comenzar desde los mensajes más recientes
                    content = msg.content
                    if marker in content:
                        parts = content.split(marker, 1)
                        if len(parts) > 1:
                            marker_content = parts[1].strip()
                            break
            
            # Si se encuentra contenido para este marcador, añadirlo a los resultados
            if marker_content:
                # Limitar el contenido hasta el siguiente marcador o separador importante
                # Buscar otros marcadores que pudieran aparecer después
                for other_marker in self.content_markers.values():
                    if other_marker != marker and other_marker in marker_content:
                        marker_content = marker_content.split(other_marker, 1)[0].strip()
                
                # También buscar separadores comunes como dobles saltos de línea seguidos de mayúsculas
                pattern = r'\n\s*\n\s*[A-Z]'
                match = re.search(pattern, marker_content)
                if match:
                    end_pos = match.start() + 1  # +1 para incluir el primer salto de línea
                    marker_content = marker_content[:end_pos].strip()
                
                results[port_name] = {
                    "content": marker_content,
                    "metadata": {"type": port_name, "marker": marker, "extracted": True}
                }
            else:
                if self.debug_mode:
                    self.log(f"No se encontró contenido para el marcador '{marker}' del puerto '{port_name}'")
        
        # Usar OutputValidator para extraer contenido para marcadores no encontrados
        for port_name, marker in self.content_markers.items():
            if port_name not in results:
                # Intentar extraer con métodos alternativos
                if "conclusion" in port_name.lower():
                    conclusion = OutputValidator.extract_conclusion_content(conversation_text)
                    if conclusion:
                        results[port_name] = {
                            "content": conclusion,
                            "metadata": {
                                "type": port_name, 
                                "marker": marker, 
                                "extracted": True,
                                "extraction_method": "alternative"
                            }
                        }
        
        # Añadir respuestas finales de cada agente
        agent_a_name = self.dialogue_controller.agent_a.name
        agent_b_name = self.dialogue_controller.agent_b.name
        
        # Obtener última respuesta del agente A
        agent_a_final = None
        agent_a_port = f"{agent_a_name.lower()}_final"
        
        # Buscar la última respuesta del agente A
        for msg in reversed(conversation):
            if msg.role == "assistant" and msg.metadata.get("agent") == agent_a_name:
                agent_a_final = msg.content
                break
        
        if agent_a_final:
            results[agent_a_port] = {
                "content": agent_a_final,
                "metadata": {"type": "agent_final_response", "agent": agent_a_name}
            }
        
        # Obtener última respuesta del agente B
        agent_b_final = None
        agent_b_port = f"{agent_b_name.lower()}_final"
        
        # Buscar la última respuesta del agente B
        for msg in reversed(conversation):
            if msg.role == "assistant" and msg.metadata.get("agent") == agent_b_name:
                agent_b_final = msg.content
                break
        
        if agent_b_final:
            results[agent_b_port] = {
                "content": agent_b_final,
                "metadata": {"type": "agent_final_response", "agent": agent_b_name}
            }
        
        # Crear resumen con metadatos
        dialogue_summary = {
            "iterations_completed": len(conversation) // 2,  # Aproximado
            "message_count": len(conversation),
            "agents": [agent_a_name, agent_b_name],
            "final_agent": conversation[-1].metadata.get("agent") if conversation and "agent" in conversation[-1].metadata else None,
            "contains_conclusion": any("conclusion" in port for port in results.keys()),
            "extracted_ports": list(results.keys())
        }
        
        results["summary"] = {
            "content": json.dumps(dialogue_summary, indent=2),
            "metadata": {"type": "dialogue_summary"}
        }
        
        # Validar y corregir outputs si está habilitado
        if self.enable_auto_correction:
            # Verificar que todos los puertos esperados estén presentes
            missing_ports = [port for port in self.expected_ports if port not in results]
            
            if missing_ports:
                self.log(f"Corrigiendo puertos faltantes: {missing_ports}")
                
                # Crear contenido por defecto para puertos faltantes
                default_content = "No se generó contenido específico para este puerto durante el diálogo."
                default_outputs = OutputValidator.create_default_outputs(missing_ports, default_content)
                
                # Añadir puertos faltantes
                for port, data in default_outputs.items():
                    results[port] = data
        
        # Añadir debug_info si está en modo debug
        if self.debug_mode:
            debug_info = {
                "conversation_length": len(conversation),
                "marker_matches": {
                    marker: output.count(marker) + conversation_text.count(marker)
                    for marker in self.content_markers.values()
                },
                "message_distribution": {
                    "user": sum(1 for msg in conversation if msg.role == "user"),
                    agent_a_name: sum(1 for msg in conversation 
                                    if msg.role == "assistant" and msg.metadata.get("agent") == agent_a_name),
                    agent_b_name: sum(1 for msg in conversation 
                                    if msg.role == "assistant" and msg.metadata.get("agent") == agent_b_name)
                },
                "ports_generated": list(results.keys())
            }
            
            results["debug_info"] = {
                "content": json.dumps(debug_info, indent=2),
                "metadata": {"type": "debug_info"}
            }
        
        return results
    
    def _extract_initial_prompt(self, processed_messages: List[Message]) -> str:
        """Extrae el prompt inicial de los mensajes procesados"""
        if not processed_messages:
            self.log("No se proporcionaron mensajes iniciales, usando prompt por defecto")
            return "Por favor, inicia una discusión."
        
        # Usar el contenido del primer mensaje como prompt inicial
        initial_prompt = processed_messages[0].content
        self.log(f"Prompt inicial extraído: {initial_prompt[:50]}...")
        return initial_prompt
    
    def process(self, inputs: List[Tuple[str, str, Message]], context: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Procesa entradas a través de este nodo de diálogo
        
        Args:
            inputs: Lista de tuplas (source_node_id, output_port, message)
            context: Información adicional de contexto
            
        Returns:
            Diccionario que mapea puertos de salida a su contenido y metadatos
        """
        try:
            self.log(f"Procesando {len(inputs)} entradas")
            
            # Procesar mensajes entrantes usando el procesador de entrada
            processed_messages = self.input_processor(inputs)
            self.log(f"Procesador de entrada devolvió {len(processed_messages)} mensajes")
            
            # Extraer el prompt inicial
            initial_prompt = self._extract_initial_prompt(processed_messages)
            
            # Iniciar el diálogo
            self.log("Iniciando diálogo")
            conversation, final_output = self.dialogue_controller.start_dialogue(initial_prompt, context)
            self.log(f"Diálogo completado con {len(conversation)} mensajes")
            
            # Procesar la salida
            if hasattr(self, 'output_processor') and self.output_processor:
                outputs = self.output_processor(final_output, conversation)
                self.log(f"Usando procesador de salida personalizado: generados {len(outputs)} puertos")
            else:
                outputs = self._enhanced_output_processor(final_output, conversation)
                self.log(f"Usando procesador de salida mejorado: generados {len(outputs)} puertos")
            
            # Verificar outputs
            self.log(f"Puertos de salida generados: {list(outputs.keys())}")
            
            return outputs
            
        except Exception as e:
            error_message = f"Error en procesamiento de nodo de diálogo: {str(e)}"
            self.log(error_message)
            import traceback
            self.log(traceback.format_exc())
            
            # Devolver una salida básica de error
            return {
                "default": {
                    "content": error_message,
                    "metadata": {"error": True}
                },
                "error": {
                    "content": error_message,
                    "metadata": {"error": True, "exception": str(e), "traceback": traceback.format_exc()}
                }
            }