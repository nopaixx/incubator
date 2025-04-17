import os
from typing import List, Dict, Optional, Any, Tuple, Set, Callable, Union
import json
import logging
import time
from datetime import datetime

class WorkflowMonitor:
    """
    Monitor para supervisar y diagnosticar la ejecución de workflows.
    Proporciona herramientas para seguimiento, diagnóstico y depuración.
    """
    
    def __init__(self, workflow_id: str = None, verbose: bool = False):
        """
        Inicializa el monitor de workflow
        
        Args:
            workflow_id: Identificador único para este workflow (opcional)
            verbose: Si se deben mostrar mensajes detallados
        """
        self.workflow_id = workflow_id or f"workflow_{int(time.time())}"
        self.verbose = verbose
        self.execution_start: Optional[float] = None
        self.execution_end: Optional[float] = None
        self.node_executions: Dict[str, Dict[str, Any]] = {}
        self.message_flow: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []
        
        # Configurar logger
        self.logger = logging.getLogger(f"WorkflowMonitor.{self.workflow_id}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(name)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
    
    def log(self, message: str, level: str = "info") -> None:
        """Registra un mensaje en el log del monitor"""
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
        
        if self.verbose and level != "debug":
            print(f"[{self.workflow_id}] {message}")
    
    def start_execution(self) -> None:
        """Marca el inicio de la ejecución del workflow"""
        self.execution_start = time.time()
        self.log(f"Iniciando ejecución del workflow: {self.workflow_id}")
    
    def end_execution(self) -> None:
        """Marca el fin de la ejecución del workflow y calcula la duración"""
        self.execution_end = time.time()
        duration = self.execution_end - (self.execution_start or self.execution_end)
        self.log(f"Ejecución finalizada. Duración: {duration:.2f} segundos")
    
    def register_node_start(self, node_id: str, node_type: str = None) -> None:
        """Registra el inicio de ejecución de un nodo"""
        if node_id not in self.node_executions:
            self.node_executions[node_id] = {
                "starts": 0,
                "completions": 0,
                "errors": 0,
                "type": node_type,
                "total_duration": 0,
                "executions": []
            }
        
        self.node_executions[node_id]["starts"] += 1
        execution = {
            "start_time": time.time(),
            "end_time": None,
            "duration": None,
            "success": None,
            "inputs": 0,
            "outputs": {}
        }
        
        self.node_executions[node_id]["executions"].append(execution)
        self.log(f"Iniciando nodo: {node_id}")
    
    def register_node_input(self, node_id: str, input_count: int) -> None:
        """Registra la cantidad de entradas que recibe un nodo"""
        if node_id in self.node_executions and self.node_executions[node_id]["executions"]:
            current_execution = self.node_executions[node_id]["executions"][-1]
            current_execution["inputs"] = input_count
            self.log(f"Nodo {node_id} recibió {input_count} entradas")
    
    def register_node_output(self, node_id: str, port_outputs: Dict[str, Any]) -> None:
        """Registra las salidas generadas por un nodo"""
        if node_id in self.node_executions and self.node_executions[node_id]["executions"]:
            current_execution = self.node_executions[node_id]["executions"][-1]
            current_execution["outputs"] = {
                port: {"content_length": len(str(data.get("content", "")))}
                for port, data in port_outputs.items()
            }
            port_count = len(port_outputs)
            self.log(f"Nodo {node_id} generó salidas en {port_count} puertos: {list(port_outputs.keys())}")
    
    def register_node_completion(self, node_id: str, success: bool = True, error: str = None) -> None:
        """Registra la finalización de la ejecución de un nodo"""
        if node_id in self.node_executions and self.node_executions[node_id]["executions"]:
            current_time = time.time()
            current_execution = self.node_executions[node_id]["executions"][-1]
            
            # Calcular duración
            start_time = current_execution["start_time"]
            duration = current_time - start_time
            
            # Actualizar estadísticas
            current_execution["end_time"] = current_time
            current_execution["duration"] = duration
            current_execution["success"] = success
            if error:
                current_execution["error"] = error
            
            # Actualizar totales
            self.node_executions[node_id]["completions"] += 1
            self.node_executions[node_id]["total_duration"] += duration
            if not success:
                self.node_executions[node_id]["errors"] += 1
                self.register_error(node_id, error or "Error desconocido")
            
            status = "exitosamente" if success else "con errores"
            self.log(f"Nodo {node_id} completado {status} en {duration:.2f} segundos")
    
    def register_message_flow(self, from_node: str, to_node: str, 
                             from_port: str, to_port: str,
                             message_type: str, content_preview: str) -> None:
        """Registra el flujo de un mensaje entre nodos"""
        flow_data = {
            "timestamp": time.time(),
            "from_node": from_node,
            "to_node": to_node,
            "from_port": from_port,
            "to_port": to_port,
            "message_type": message_type,
            "content_preview": content_preview[:100] + ("..." if len(content_preview) > 100 else "")
        }
        
        self.message_flow.append(flow_data)
        self.log(f"Mensaje desde {from_node}:{from_port} -> {to_node}:{to_port}")
    
    def register_error(self, node_id: str, error_message: str, details: Any = None) -> None:
        """Registra un error ocurrido durante la ejecución"""
        error_data = {
            "timestamp": time.time(),
            "node_id": node_id,
            "error_message": error_message,
            "details": details
        }
        
        self.errors.append(error_data)
        self.log(f"ERROR en nodo {node_id}: {error_message}", level="error")
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Obtiene un resumen de la ejecución del workflow"""
        execution_time = None
        if self.execution_start:
            end_time = self.execution_end or time.time()
            execution_time = end_time - self.execution_start
        
        return {
            "workflow_id": self.workflow_id,
            "start_time": datetime.fromtimestamp(self.execution_start).isoformat() if self.execution_start else None,
            "end_time": datetime.fromtimestamp(self.execution_end).isoformat() if self.execution_end else None,
            "execution_time": f"{execution_time:.2f}s" if execution_time else None,
            "nodes_executed": len(self.node_executions),
            "total_node_executions": sum(node["starts"] for node in self.node_executions.values()),
            "successful_nodes": sum(1 for node in self.node_executions.values() if node["errors"] == 0),
            "failed_nodes": sum(1 for node in self.node_executions.values() if node["errors"] > 0),
            "total_errors": len(self.errors),
            "messages_passed": len(self.message_flow)
        }
    
    def get_node_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas detalladas sobre la ejecución de cada nodo"""
        stats = {}
        for node_id, data in self.node_executions.items():
            # Calcular tiempo promedio de ejecución
            avg_duration = 0
            if data["completions"] > 0:
                avg_duration = data["total_duration"] / data["completions"]
            
            # Extraer información sobre puertos usados
            output_ports_used = set()
            for execution in data["executions"]:
                if "outputs" in execution:
                    output_ports_used.update(execution["outputs"].keys())
            
            # Armar estadísticas del nodo
            stats[node_id] = {
                "type": data["type"],
                "executions": data["starts"],
                "completions": data["completions"],
                "success_rate": (data["completions"] - data["errors"]) / data["completions"] if data["completions"] > 0 else 0,
                "average_duration": f"{avg_duration:.2f}s",
                "output_ports_used": list(output_ports_used),
                "error_count": data["errors"]
            }
        
        return stats
    
    def get_error_report(self) -> List[Dict[str, Any]]:
        """Obtiene un informe detallado de los errores ocurridos"""
        return [{
            "node_id": error["node_id"],
            "timestamp": datetime.fromtimestamp(error["timestamp"]).isoformat(),
            "error_message": error["error_message"],
            "details": error["details"] if "details" in error else None
        } for error in self.errors]
    
    def save_report(self, output_dir: str, filename_prefix: str = None) -> str:
        """
        Guarda un informe completo de la ejecución del workflow
        
        Args:
            output_dir: Directorio donde guardar el informe
            filename_prefix: Prefijo para los nombres de archivo (opcional)
            
        Returns:
            Ruta al archivo de informe principal
        """
        os.makedirs(output_dir, exist_ok=True)
        prefix = filename_prefix or self.workflow_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generar informe completo
        report = {
            "summary": self.get_execution_summary(),
            "node_statistics": self.get_node_statistics(),
            "errors": self.get_error_report(),
            "message_flow": self.message_flow
        }
        
        # Guardar informe principal
        report_path = os.path.join(output_dir, f"{prefix}_report_{timestamp}.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.log(f"Informe de ejecución guardado en: {report_path}")
        return report_path
    
    def diagnose_workflow_issues(self) -> Dict[str, Any]:
        """
        Realiza un diagnóstico de posibles problemas en el workflow
        
        Returns:
            Diccionario con problemas identificados y recomendaciones
        """
        issues = []
        warnings = []
        recommendations = []
        
        # Verificar nodos que no completaron su ejecución
        incomplete_nodes = []
        for node_id, data in self.node_executions.items():
            if data["starts"] > data["completions"]:
                incomplete_nodes.append(node_id)
                issues.append(f"Nodo {node_id} inició {data['starts']} veces pero completó solo {data['completions']}")
        
        # Verificar nodos sin salidas
        nodes_without_outputs = []
        for node_id, data in self.node_executions.items():
            if any(not execution.get("outputs", {}) for execution in data["executions"] if execution.get("success", False)):
                nodes_without_outputs.append(node_id)
                warnings.append(f"Nodo {node_id} completó ejecución exitosamente pero no generó salidas")
        
        # Verificar nodos sin entradas
        nodes_without_inputs = []
        for node_id, data in self.node_executions.items():
            if any(execution.get("inputs", 0) == 0 for execution in data["executions"]):
                nodes_without_inputs.append(node_id)
                warnings.append(f"Nodo {node_id} ejecutado sin recibir entradas")
        
        # Generar recomendaciones
        if incomplete_nodes:
            recommendations.append("Revisar posibles bloqueos o excepciones no capturadas en los nodos que no completaron")
        
        if nodes_without_outputs:
            recommendations.append("Verificar que todos los nodos produzcan salidas en los puertos esperados")
            recommendations.append("Revisar los procesadores de salida en los nodos sin outputs")
        
        if self.errors:
            recommendations.append("Corregir errores reportados en nodos con fallos")
        
        # Diagnóstico específico para DialogNode
        dialog_nodes = [node_id for node_id, data in self.node_executions.items() 
                       if data.get("type") == "IterativeDialogNode"]
        
        for node_id in dialog_nodes:
            # Verificar si los puertos esperados existen en las salidas
            expected_ports = ["default", "conversation", "summary"]
            for execution in self.node_executions[node_id]["executions"]:
                if execution.get("success", False):
                    missing_ports = [port for port in expected_ports if port not in execution.get("outputs", {})]
                    if missing_ports:
                        warnings.append(f"Nodo de diálogo {node_id} no generó salidas en puertos esperados: {', '.join(missing_ports)}")
                        recommendations.append(f"Revisar el procesador de salida del nodo {node_id} para asegurar que genere todos los puertos esperados")
        
        return {
            "issues": issues,
            "warnings": warnings,
            "recommendations": recommendations,
            "incomplete_nodes": incomplete_nodes,
            "nodes_without_outputs": nodes_without_outputs,
            "nodes_without_inputs": nodes_without_inputs,
            "error_count": len(self.errors)
        }