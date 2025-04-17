import os
import json
import time
import traceback
import logging
from typing import List, Dict, Optional, Any, Tuple, Set, Callable, Union
import networkx as nx
from datetime import datetime

from incubator.agents.agent import Agent
from incubator.messages.message import Message
from incubator.llm.llmclient import LLMClient
from incubator.wf.node import Node
from incubator.wf.monitor import WorkflowMonitor
from incubator.wf.output_validator import OutputValidator

class EnhancedWorkflowEngine:
    """Motor mejorado de ejecución de workflows definidos como grafos dirigidos de nodos"""
    
    def __init__(self, 
                workflow_id: str = None,
                monitor: WorkflowMonitor = None,
                expected_node_outputs: Dict[str, List[str]] = None,
                enable_auto_correction: bool = True,
                output_validators: Dict[str, Dict[str, Callable]] = None,
                debug: bool = False):
        """
        Inicializa el motor de workflow mejorado
        
        Args:
            workflow_id: Identificador único para este workflow
            monitor: Monitor de workflow (se creará uno por defecto si no se proporciona)
            expected_node_outputs: Diccionario con los puertos de salida esperados para cada nodo
            enable_auto_correction: Si se deben aplicar correcciones automáticas en caso de error
            output_validators: Validadores específicos para outputs de cada nodo
            debug: Si se debe activar el modo de depuración
        """
        self.workflow_id = workflow_id or f"workflow_{int(time.time())}"
        self.graph = nx.MultiDiGraph()  # Usar MultiDiGraph para permitir múltiples aristas entre nodos
        self.input_nodes = set()
        self.output_nodes = set()
        self.node_instances = {}  # Mapa de node_id -> instancia de Node
        self.debug = debug
        
        # Monitor para seguimiento y diagnóstico
        self.monitor = monitor or WorkflowMonitor(workflow_id=self.workflow_id, verbose=debug)
        
        # Configuraciones para validación y corrección de outputs
        self.expected_node_outputs = expected_node_outputs or {}
        self.enable_auto_correction = enable_auto_correction
        self.output_validators = output_validators or {}
        
        # Estadísticas de ejecución
        self.execution_stats = {
            "last_execution_time": None,
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0
        }
        
        # Configurar logging
        self.logger = logging.getLogger(f"EnhancedWorkflowEngine.{self.workflow_id}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(name)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO if debug else logging.WARNING)
    
    def log(self, message: str, level: str = "info") -> None:
        """Registra un mensaje en el log del workflow"""
        if level == "debug":
            self.logger.debug(message)
        elif level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "critical":
            self.logger.critical(message)
        
        if self.debug and level != "debug":
            print(f"[WorkflowEngine.{self.workflow_id}] {message}")
    
    def add_node(self, node: Node, is_input: bool = False, is_output: bool = False,
                expected_outputs: List[str] = None) -> None:
        """
        Añade un nodo al workflow
        
        Args:
            node: El nodo a añadir
            is_input: Si este nodo puede recibir entrada externa
            is_output: Si este nodo produce salida externa
            expected_outputs: Lista de puertos de salida esperados para este nodo
        """
        if node.id in self.graph.nodes:
            raise ValueError(f"Ya existe un nodo con id '{node.id}'")
        
        self.graph.add_node(node.id)
        self.node_instances[node.id] = node
        
        if is_input:
            self.input_nodes.add(node.id)
        
        if is_output:
            self.output_nodes.add(node.id)
        
        # Registrar puertos de salida esperados si se proporcionan
        if expected_outputs:
            self.expected_node_outputs[node.id] = expected_outputs
            self.log(f"Registrados puertos esperados para nodo {node.id}: {expected_outputs}")
    
    def add_edge(self, from_node_id: str, to_node_id: str, 
                from_port: str = "default", to_port: str = "input", 
                metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Añade una arista dirigida entre nodos con puertos específicos
        
        Args:
            from_node_id: ID del nodo origen
            to_node_id: ID del nodo destino
            from_port: Puerto de salida del nodo origen (default: "default")
            to_port: Puerto de entrada del nodo destino (default: "input")
            metadata: Metadatos adicionales para la arista
        """
        if from_node_id not in self.graph.nodes:
            raise ValueError(f"El nodo origen '{from_node_id}' no existe")
        
        if to_node_id not in self.graph.nodes:
            raise ValueError(f"El nodo destino '{to_node_id}' no existe")
        
        # Crear datos de la arista con información de puertos
        edge_data = {
            "from_port": from_port,
            "to_port": to_port,
            "metadata": metadata or {}
        }
        
        self.graph.add_edge(from_node_id, to_node_id, key=f"{from_port}_to_{to_port}", **edge_data)
        self.log(f"Añadida conexión: {from_node_id}:{from_port} -> {to_node_id}:{to_port}")
    
    def validate(self) -> bool:
        """
        Valida el grafo del workflow
        
        Returns:
            True si el grafo es válido, False en caso contrario
        """
        validation_errors = []
        
        # Verificar si hay nodos de entrada
        if not self.input_nodes:
            validation_errors.append("Error: No hay nodos de entrada definidos")
        
        # Verificar si hay nodos de salida
        if not self.output_nodes:
            validation_errors.append("Error: No hay nodos de salida definidos")
        
        # Verificar ciclos
        if not nx.is_directed_acyclic_graph(self.graph):
            validation_errors.append("Error: El workflow contiene ciclos")
        
        # Verificar alcanzabilidad desde nodos de entrada a nodos de salida
        for input_id in self.input_nodes:
            can_reach_output = False
            for output_id in self.output_nodes:
                if nx.has_path(self.graph, input_id, output_id):
                    can_reach_output = True
                    break
            
            if not can_reach_output:
                validation_errors.append(f"Error: El nodo de entrada '{input_id}' no puede alcanzar ningún nodo de salida")
        
        # Mostrar errores si los hay
        if validation_errors:
            for error in validation_errors:
                self.log(error, level="error")
            return False
        
        return True
    
    def execute(self, inputs: Dict[str, str], context: Optional[Dict[str, Any]] = None,
               save_results: bool = True, results_dir: Optional[str] = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Ejecuta el workflow con las entradas proporcionadas
        
        Args:
            inputs: Diccionario que mapea IDs de nodos de entrada a contenido
            context: Información adicional de contexto
            save_results: Si se deben guardar los resultados de la ejecución
            results_dir: Directorio donde guardar los resultados (opcional)
            
        Returns:
            Diccionario que mapea IDs de nodos de salida a sus puertos de salida (con contenido y metadatos)
        """
        # Actualizar estadísticas
        self.execution_stats["total_executions"] += 1
        self.execution_stats["last_execution_time"] = time.time()
        
        # Iniciar monitoreo
        self.monitor.start_execution()
        
        try:
            if not self.validate():
                raise ValueError("Grafo de workflow inválido")
            
            # Verificar que todas las claves de entrada sean nodos de entrada válidos
            for node_id in inputs.keys():
                if node_id not in self.input_nodes:
                    raise ValueError(f"'{node_id}' no es un nodo de entrada válido")
            
            # Inicializar almacenamiento de mensajes
            node_outputs = {}  # Mapa de node_id -> port_id -> datos de salida
            
            # Crear mensajes iniciales para nodos de entrada
            for node_id, content in inputs.items():
                # Crear un mensaje de usuario para cada entrada
                message = Message(role="user", content=content)
                node_outputs[node_id] = {
                    "default": {
                        "content": content, 
                        "message": message, 
                        "metadata": {}
                    }
                }
                self.log(f"Entrada proporcionada para nodo: {node_id}")
            
            # Procesar nodos en orden topológico
            for node_id in nx.topological_sort(self.graph):
                try:
                    # Omitir si es un nodo de entrada que ya hemos procesado
                    if node_id in inputs:
                        continue
                    
                    self.log(f"Procesando nodo: {node_id}")
                    
                    # Obtener la instancia del nodo
                    node = self.node_instances[node_id]
                    
                    # Registrar inicio de ejecución del nodo
                    node_type = node.__class__.__name__
                    self.monitor.register_node_start(node_id, node_type)
                    
                    # Recopilar entradas de nodos predecesores basadas en conexiones
                    node_inputs = []
                    
                    # Para cada predecesor
                    for pred_id in self.graph.predecessors(node_id):
                        # Omitir si el predecesor no ha sido procesado aún
                        if pred_id not in node_outputs:
                            continue
                        
                        # Obtener todas las aristas entre este predecesor y el nodo actual
                        edges = self.graph.get_edge_data(pred_id, node_id)
                        
                        # Para cada arista
                        for edge_key, edge_data in edges.items():
                            from_port = edge_data["from_port"]
                            to_port = edge_data["to_port"]
                            
                            # Verificar si existe el puerto de origen
                            if from_port in node_outputs[pred_id]:
                                port_output = node_outputs[pred_id][from_port]
                                
                                # Crear mensaje desde esta salida
                                message = Message(
                                    role="user", 
                                    content=port_output["content"],
                                    metadata=port_output.get("metadata", {})
                                )
                                
                                # Añadir a entradas con información de origen y puerto
                                node_inputs.append((pred_id, from_port, message))
                                
                                # Registrar flujo de mensaje
                                self.monitor.register_message_flow(
                                    from_node=pred_id,
                                    to_node=node_id,
                                    from_port=from_port,
                                    to_port=to_port,
                                    message_type=message.role,
                                    content_preview=message.content
                                )
                    
                    # Registrar cantidad de entradas
                    self.monitor.register_node_input(node_id, len(node_inputs))
                    
                    # Procesar el nodo
                    port_outputs = node.process(node_inputs, context)
                    
                    # Validar y corregir outputs si es necesario
                    if node_id in self.expected_node_outputs:
                        expected_ports = self.expected_node_outputs[node_id]
                        validators = self.output_validators.get(node_id, {})
                        
                        # Validar outputs
                        is_valid, validation_errors, validation_meta = OutputValidator.validate_node_output(
                            port_outputs, expected_ports, validators
                        )
                        
                        # Si hay errores y está habilitada la corrección automática
                        if not is_valid and self.enable_auto_correction:
                            self.log(f"Corrigiendo outputs de nodo {node_id}: {validation_meta}", level="warning")
                            
                            # Asegurar que todos los puertos esperados existen
                            missing_ports = validation_meta.get("missing_ports", [])
                            if missing_ports:
                                # Crear puertos faltantes
                                default_content = f"Contenido no generado para el puerto {missing_ports}"
                                additional_outputs = OutputValidator.create_default_outputs(
                                    missing_ports, default_content
                                )
                                # Añadir puertos faltantes a los outputs
                                port_outputs.update(additional_outputs)
                                self.log(f"Añadidos puertos faltantes a nodo {node_id}: {missing_ports}")
                    
                    # Registrar outputs
                    self.monitor.register_node_output(node_id, port_outputs)
                    
                    # Almacenar cada salida de puerto
                    output_data = {}
                    for port_id, port_output in port_outputs.items():
                        # Crear un mensaje desde la salida si no está ya presente
                        if "message" not in port_output:
                            message = Message(
                                role="assistant", 
                                content=port_output["content"],
                                metadata=port_output.get("metadata", {})
                            )
                            port_output["message"] = message
                        
                        # Almacenar con información de puerto
                        output_data[port_id] = port_output
                    
                    node_outputs[node_id] = output_data
                    self.log(f"Nodo {node_id} procesado con puertos de salida: {list(output_data.keys())}")
                    
                    # Registrar finalización exitosa
                    self.monitor.register_node_completion(node_id, success=True)
                    
                except Exception as e:
                    error_message = f"Error al procesar nodo {node_id}: {str(e)}"
                    self.log(error_message, level="error")
                    self.log(traceback.format_exc(), level="debug")
                    
                    # Registrar error
                    self.monitor.register_node_completion(node_id, success=False, error=str(e))
                    self.monitor.register_error(node_id, str(e), traceback.format_exc())
                    
                    # Crear salida de error para este nodo
                    node_outputs[node_id] = {
                        "error": {
                            "content": error_message,
                            "metadata": {"error": True, "exception": str(e)},
                            "message": Message(role="assistant", content=error_message)
                        },
                        "default": {
                            "content": f"Error en el procesamiento del nodo: {str(e)}",
                            "metadata": {"error": True, "exception": str(e)},
                            "message": Message(role="assistant", content=f"Error en el procesamiento del nodo: {str(e)}")
                        }
                    }
            
            # Recopilar salidas
            results = {}
            for node_id in self.output_nodes:
                if node_id in node_outputs:
                    # Incluir datos completos de puertos para los nodos de salida
                    results[node_id] = node_outputs[node_id]
            
            # Actualizar estadísticas
            self.execution_stats["successful_executions"] += 1
            
            # Finalizar monitoreo
            self.monitor.end_execution()
            
            # Guardar resultados si está habilitado
            if save_results and results_dir:
                self._save_execution_results(results, results_dir)
                
                # También guardar informe de monitoreo
                self.monitor.save_report(results_dir, f"{self.workflow_id}_execution")
            
            return results
            
        except Exception as e:
            # Actualizar estadísticas
            self.execution_stats["failed_executions"] += 1
            
            # Registrar error
            error_message = f"Error general en la ejecución del workflow: {str(e)}"
            self.log(error_message, level="error")
            self.log(traceback.format_exc(), level="debug")
            
            # Finalizar monitoreo
            self.monitor.register_error("workflow", str(e), traceback.format_exc())
            self.monitor.end_execution()
            
            # Crear resultados de error
            error_results = {
                node_id: {
                    "error": {
                        "content": error_message,
                        "metadata": {"error": True, "exception": str(e)}
                    }
                } for node_id in self.output_nodes
            }
            
            # Guardar informe de error si está habilitado
            if save_results and results_dir:
                self._save_execution_results(error_results, results_dir, is_error=True)
                self.monitor.save_report(results_dir, f"{self.workflow_id}_error")
            
            return error_results
    
    def _save_execution_results(self, results: Dict[str, Dict[str, Dict[str, Any]]], 
                               output_dir: str, is_error: bool = False) -> str:
        """
        Guarda los resultados de ejecución en un directorio
        
        Args:
            results: Resultados a guardar
            output_dir: Directorio base donde guardar
            is_error: Si son resultados de error
            
        Returns:
            Ruta al directorio de resultados
        """
        # Crear directorio con timestamp para evitar sobrescribir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        status = "error" if is_error else "success"
        result_dir = os.path.join(output_dir, f"{self.workflow_id}_{status}_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)
        
        # Guardar metadatos de ejecución
        metadata = {
            "workflow_id": self.workflow_id,
            "execution_time": timestamp,
            "status": status,
            "nodes": {
                "total": len(self.graph.nodes),
                "input": len(self.input_nodes),
                "output": len(self.output_nodes)
            }
        }
        
        with open(os.path.join(result_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        # Procesar y guardar resultados para cada nodo de salida
        for node_id, node_outputs in results.items():
            # Crear directorio para cada nodo
            node_dir = os.path.join(result_dir, node_id)
            os.makedirs(node_dir, exist_ok=True)
            
            # Guardar cada puerto por separado
            for port_name, port_data in node_outputs.items():
                # Obtener el contenido
                content = port_data.get("content", "")
                
                # Detectar formato para guardar apropiadamente
                is_json = False
                try:
                    if isinstance(content, str) and (content.startswith("{") or content.startswith("[")):
                        json.loads(content)
                        is_json = True
                except:
                    is_json = False
                
                # Determinar extensión y guardar
                ext = "json" if is_json else "txt"
                filename = f"{port_name}.{ext}"
                
                with open(os.path.join(node_dir, filename), "w", encoding="utf-8") as f:
                    f.write(content)
                
                # Guardar también metadatos
                if "metadata" in port_data:
                    with open(os.path.join(node_dir, f"{port_name}_metadata.json"), "w", encoding="utf-8") as f:
                        # Asegurar que solo se guarden metadatos serializables
                        serializable_metadata = {}
                        for k, v in port_data["metadata"].items():
                            if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                                serializable_metadata[k] = v
                        json.dump(serializable_metadata, f, indent=2)
            
            # Guardar todos los outputs del nodo en un solo archivo
            with open(os.path.join(node_dir, "all_outputs.json"), "w", encoding="utf-8") as f:
                # Filtrar para solo incluir contenido y metadatos serializables
                serializable_outputs = {}
                for port, data in node_outputs.items():
                    serializable_outputs[port] = {
                        "content": data.get("content", "")
                    }
                    
                    # Incluir metadatos serializables
                    if "metadata" in data:
                        serializable_outputs[port]["metadata"] = {}
                        for k, v in data["metadata"].items():
                            if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                                serializable_outputs[port]["metadata"][k] = v
                
                json.dump(serializable_outputs, f, indent=2)
        
        # Guardar resultados completos
        results_file = os.path.join(result_dir, "all_results.json")
        
        with open(results_file, "w", encoding="utf-8") as f:
            # Filtrar para solo incluir datos serializables
            serializable_results = {}
            
            for node_id, node_outputs in results.items():
                serializable_results[node_id] = {}
                
                for port_name, port_data in node_outputs.items():
                    serializable_results[node_id][port_name] = {
                        "content": port_data.get("content", "")
                    }
                    
                    # Incluir metadatos serializables
                    if "metadata" in port_data:
                        serializable_results[node_id][port_name]["metadata"] = {}
                        for k, v in port_data["metadata"].items():
                            if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                                serializable_results[node_id][port_name]["metadata"][k] = v
            
            json.dump(serializable_results, f, indent=2)
        
        self.log(f"Resultados guardados en: {result_dir}")
        return result_dir
    
    def get_node_info(self, node_id: str) -> Dict[str, Any]:
        """
        Obtiene información detallada sobre un nodo específico
        
        Args:
            node_id: ID del nodo
            
        Returns:
            Diccionario con información del nodo
        """
        if node_id not in self.graph.nodes:
            raise ValueError(f"No existe un nodo con id '{node_id}'")
        
        node = self.node_instances[node_id]
        
        # Obtener predecesores y sucesores
        predecessors = list(self.graph.predecessors(node_id))
        successors = list(self.graph.successors(node_id))
        
        # Obtener conexiones de entrada
        incoming_connections = []
        for pred in predecessors:
            edges = self.graph.get_edge_data(pred, node_id)
            for key, data in edges.items():
                incoming_connections.append({
                    "from_node": pred,
                    "from_port": data["from_port"],
                    "to_port": data["to_port"],
                    "metadata": data.get("metadata", {})
                })
        
        # Obtener conexiones de salida
        outgoing_connections = []
        for succ in successors:
            edges = self.graph.get_edge_data(node_id, succ)
            for key, data in edges.items():
                outgoing_connections.append({
                    "to_node": succ,
                    "from_port": data["from_port"],
                    "to_port": data["to_port"],
                    "metadata": data.get("metadata", {})
                })
        
        # Determinar tipo y estado del nodo
        node_type = node.__class__.__name__
        is_input = node_id in self.input_nodes
        is_output = node_id in self.output_nodes
        expected_outputs = self.expected_node_outputs.get(node_id, [])
        
        return {
            "id": node_id,
            "type": node_type,
            "is_input": is_input,
            "is_output": is_output,
            "predecessors": predecessors,
            "successors": successors,
            "incoming_connections": incoming_connections,
            "outgoing_connections": outgoing_connections,
            "expected_outputs": expected_outputs
        }
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """
        Obtiene un resumen completo del workflow
        
        Returns:
            Diccionario con información resumida del workflow
        """
        return {
            "id": self.workflow_id,
            "node_count": len(self.graph.nodes),
            "edge_count": self.graph.number_of_edges(),
            "input_nodes": list(self.input_nodes),
            "output_nodes": list(self.output_nodes),
            "execution_stats": self.execution_stats,
            "is_valid": self.validate(),
            "nodes": {
                node_id: {
                    "type": self.node_instances[node_id].__class__.__name__,
                    "is_input": node_id in self.input_nodes,
                    "is_output": node_id in self.output_nodes,
                    "connections": {
                        "in": len(list(self.graph.predecessors(node_id))),
                        "out": len(list(self.graph.successors(node_id)))
                    }
                }
                for node_id in self.graph.nodes
            }
        }
    
    def visualize(self, output_path: Optional[str] = None) -> str:
        """
        Genera una visualización del grafo del workflow usando DOT
        
        Args:
            output_path: Ruta donde guardar la visualización (opcional)
            
        Returns:
            Cadena DOT representando el grafo
        """
        try:
            import networkx as nx
            
            # Crear grafo de visualización
            dot_graph = nx.nx_pydot.to_pydot(self.graph)
            
            # Personalizar nodos
            for node in dot_graph.get_nodes():
                node_id = node.get_name().strip('"')
                
                # Personalizar según tipo
                if node_id in self.input_nodes and node_id in self.output_nodes:
                    node.set_style("filled")
                    node.set_fillcolor("lightgreen")
                    node.set_shape("box")
                elif node_id in self.input_nodes:
                    node.set_style("filled")
                    node.set_fillcolor("lightblue")
                    node.set_shape("ellipse")
                elif node_id in self.output_nodes:
                    node.set_style("filled")
                    node.set_fillcolor("lightyellow")
                    node.set_shape("ellipse")
            
            # Personalizar aristas
            for edge in dot_graph.get_edges():
                # Añadir información de puertos
                from_node = edge.get_source().strip('"')
                to_node = edge.get_destination().strip('"')
                
                # Obtener datos de la arista
                edge_data = self.graph.get_edge_data(from_node, to_node)
                
                # Puede haber múltiples aristas, usar solo la primera para simplificar
                if edge_data:
                    first_edge_key = list(edge_data.keys())[0]
                    first_edge = edge_data[first_edge_key]
                    
                    from_port = first_edge.get("from_port", "default")
                    to_port = first_edge.get("to_port", "input")
                    
                    # Establecer etiqueta con información de puertos
                    edge.set_label(f"{from_port} -> {to_port}")
            
            # Guardar visualización si se proporcionó ruta
            if output_path:
                extension = output_path.split(".")[-1].lower()
                
                if extension == "dot":
                    with open(output_path, "w") as f:
                        f.write(dot_graph.to_string())
                elif extension in ["png", "jpg", "jpeg", "pdf", "svg"]:
                    dot_graph.write(output_path, format=extension)
                else:
                    self.log(f"Formato no soportado: {extension}. Guardando como .dot", level="warning")
                    with open(f"{output_path}.dot", "w") as f:
                        f.write(dot_graph.to_string())
            
            return dot_graph.to_string()
            
        except ImportError:
            self.log("Se requiere pydot para visualización. Instálalo con: pip install pydot", level="error")
            return "Error: pydot no instalado"
        except Exception as e:
            self.log(f"Error al generar visualización: {str(e)}", level="error")
            return f"Error: {str(e)}"
    
    def from_legacy_workflow(self, legacy_workflow, copy_nodes: bool = True):
        """
        Convierte un workflow existente de la versión anterior a la versión mejorada
        
        Args:
            legacy_workflow: Instancia de WorkflowEngine anterior
            copy_nodes: Si se deben copiar los nodos (True) o solo referenciarlos (False)
        """
        # Copiar nodos
        for node_id in legacy_workflow.graph.nodes:
            if node_id not in self.graph.nodes:
                node = legacy_workflow.node_instances[node_id]
                
                # Determinar tipo de nodo
                is_input = node_id in legacy_workflow.input_nodes
                is_output = node_id in legacy_workflow.output_nodes
                
                if copy_nodes:
                    # Solo copiar nodos si se solicita
                    self.add_node(node, is_input=is_input, is_output=is_output)
                else:
                    # Referenciar los mismos nodos (compartirá instancias con el workflow original)
                    self.graph.add_node(node_id)
                    self.node_instances[node_id] = node
                    
                    if is_input:
                        self.input_nodes.add(node_id)
                    
                    if is_output:
                        self.output_nodes.add(node_id)
        
        # Copiar aristas
        for u, v, key, data in legacy_workflow.graph.edges(data=True, keys=True):
            from_port = data.get("from_port", "default")
            to_port = data.get("to_port", "input")
            metadata = data.get("metadata", {})
            
            if u in self.graph.nodes and v in self.graph.nodes:
                self.add_edge(u, v, from_port, to_port, metadata)
        
        self.log(f"Convertido workflow anterior con {len(self.graph.nodes)} nodos y {self.graph.number_of_edges()} conexiones")
        return self
    
    @staticmethod
    def convert_from_legacy(legacy_workflow, workflow_id: str = None, 
                          debug: bool = False) -> 'EnhancedWorkflowEngine':
        """
        Método estático para convertir un workflow legado a la versión mejorada
        
        Args:
            legacy_workflow: Instancia de WorkflowEngine anterior
            workflow_id: ID para el nuevo workflow (opcional)
            debug: Si se debe activar modo de depuración
            
        Returns:
            Nueva instancia de EnhancedWorkflowEngine
        """
        workflow_id = workflow_id or f"converted_{int(time.time())}"
        new_workflow = EnhancedWorkflowEngine(workflow_id=workflow_id, debug=debug)
        return new_workflow.from_legacy_workflow(legacy_workflow)