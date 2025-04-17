import os
from typing import List, Dict, Optional, Any, Tuple, Set, Callable
import logging
import json
from datetime import datetime

from incubator.wf.wf import WorkflowEngine as LegacyWorkflowEngine
from incubator.wf.dialognode import IterativeDialogNode as LegacyDialogNode 
from incubator.wf.enhanced_workflow import EnhancedWorkflowEngine
from incubator.wf.enhanced_dialognode import EnhancedDialogNode
from incubator.wf.node import Node
from incubator.agents.agent import Agent
from incubator.messages.message import Message

class WorkflowAdapter:
    """
    Adaptador para mantener compatibilidad entre versiones anteriores 
    y mejoradas de los componentes de workflow.
    """
    
    @staticmethod
    def legacy_to_enhanced_workflow(legacy_workflow: LegacyWorkflowEngine, 
                                  workflow_id: Optional[str] = None,
                                  enable_auto_correction: bool = True,
                                  debug: bool = False) -> EnhancedWorkflowEngine:
        """
        Convierte un WorkflowEngine existente a un EnhancedWorkflowEngine
        
        Args:
            legacy_workflow: Instancia de WorkflowEngine a convertir
            workflow_id: ID para el nuevo workflow (opcional)
            enable_auto_correction: Si se habilita la corrección automática de outputs
            debug: Si se activa el modo de depuración
            
        Returns:
            Instancia de EnhancedWorkflowEngine con la misma configuración
        """
        # Crear ID de workflow si no se proporciona
        workflow_id = workflow_id or f"enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Crear nuevo workflow mejorado
        enhanced_workflow = EnhancedWorkflowEngine(
            workflow_id=workflow_id,
            enable_auto_correction=enable_auto_correction,
            debug=debug
        )
        
        # Copiar todos los nodos
        for node_id in legacy_workflow.graph.nodes:
            if node_id in legacy_workflow.node_instances:
                node = legacy_workflow.node_instances[node_id]
                is_input = node_id in legacy_workflow.input_nodes
                is_output = node_id in legacy_workflow.output_nodes
                
                # Determinar puertos esperados basados en el tipo de nodo
                expected_outputs = None
                if isinstance(node, LegacyDialogNode):
                    expected_outputs = [
                        "default", "conversation", "summary", 
                        f"{node.dialogue_controller.agent_a.name.lower()}_final", 
                        f"{node.dialogue_controller.agent_b.name.lower()}_final"
                    ]
                    
                    # Verificar marcadores de contenido y añadir como puertos esperados
                    if hasattr(node, 'content_markers') and node.content_markers:
                        expected_outputs.extend(list(node.content_markers.keys()))
                
                # Añadir nodo al workflow mejorado
                enhanced_workflow.add_node(
                    node=node, 
                    is_input=is_input, 
                    is_output=is_output,
                    expected_outputs=expected_outputs
                )
        
        # Copiar todas las conexiones (aristas)
        for u, v, key, data in legacy_workflow.graph.edges(data=True, keys=True):
            from_port = data.get("from_port", "default")
            to_port = data.get("to_port", "input")
            metadata = data.get("metadata", {})
            
            enhanced_workflow.add_edge(
                from_node_id=u,
                to_node_id=v,
                from_port=from_port,
                to_port=to_port,
                metadata=metadata
            )
        
        return enhanced_workflow
    
    @staticmethod
    def legacy_to_enhanced_dialog_node(legacy_node: LegacyDialogNode,
                                     id: Optional[str] = None,
                                     enable_auto_correction: bool = True,
                                     verbose: bool = False) -> EnhancedDialogNode:
        """
        Convierte un IterativeDialogNode existente a un EnhancedDialogNode
        
        Args:
            legacy_node: Instancia de IterativeDialogNode a convertir
            id: Nuevo ID para el nodo (opcional)
            enable_auto_correction: Si se habilita la corrección automática
            verbose: Si se activa el modo detallado
            
        Returns:
            Instancia de EnhancedDialogNode con la misma configuración
        """
        # Usar el ID existente si no se proporciona uno nuevo
        id = id or legacy_node.id
        
        # Obtener configuraciones del nodo original
        agent_a = legacy_node.dialogue_controller.agent_a
        agent_b = legacy_node.dialogue_controller.agent_b
        max_iterations = legacy_node.dialogue_controller.max_iterations
        terminate_on_keywords = legacy_node.dialogue_controller.terminate_on_keywords
        termination_condition = legacy_node.dialogue_controller.termination_condition
        
        # Capturar procesadores personalizados si existen
        input_processor = None
        if hasattr(legacy_node, 'input_processor') and legacy_node.input_processor:
            if legacy_node.input_processor.__name__ != '_default_input_processor':
                input_processor = legacy_node.input_processor
        
        output_processor = None
        if hasattr(legacy_node, 'output_processor') and legacy_node.output_processor:
            if legacy_node.output_processor.__name__ != '_default_output_processor':
                output_processor = legacy_node.output_processor
        
        # Obtener marcadores de contenido
        content_markers = legacy_node.content_markers if hasattr(legacy_node, 'content_markers') else None
        
        # Crear nodo mejorado
        enhanced_node = EnhancedDialogNode(
            id=id,
            agent_a=agent_a,
            agent_b=agent_b,
            max_iterations=max_iterations,
            terminate_on_keywords=terminate_on_keywords,
            termination_condition=termination_condition,
            input_processor=input_processor,
            output_processor=output_processor,
            content_markers=content_markers,
            enable_auto_correction=enable_auto_correction,
            verbose=verbose
        )
        
        return enhanced_node
    
    @staticmethod
    def wrap_legacy_workflow(legacy_workflow: LegacyWorkflowEngine,
                           save_results: bool = True,
                           results_dir: str = "resultados",
                           workflow_id: Optional[str] = None,
                           debug: bool = False) -> Callable:
        """
        Crea una función wrapper que ejecuta un workflow legado con las 
        funcionalidades mejoradas, sin modificar el workflow original
        
        Args:
            legacy_workflow: Workflow legado a envolver
            save_results: Si se deben guardar resultados automáticamente  
            results_dir: Directorio para guardar resultados
            workflow_id: ID para el workflow mejorado (opcional)
            debug: Si se activa el modo de depuración
            
        Returns:
            Función que ejecuta el workflow con las mejoras
        """
        # Crear directorio de resultados si no existe
        if save_results:
            os.makedirs(results_dir, exist_ok=True)
        
        # Función wrapper
        def execute_with_enhancements(inputs: Dict[str, str], 
                                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
            """
            Ejecuta el workflow legado con las mejoras de monitoreo, validación y corrección
            
            Args:
                inputs: Entradas para el workflow
                context: Contexto adicional (opcional)
                
            Returns:
                Resultados del workflow con formato mejorado
            """
            # Convertir a workflow mejorado
            enhanced_workflow = WorkflowAdapter.legacy_to_enhanced_workflow(
                legacy_workflow=legacy_workflow,
                workflow_id=workflow_id,
                debug=debug
            )
            
            # Ejecutar el workflow mejorado
            results = enhanced_workflow.execute(
                inputs=inputs,
                context=context,
                save_results=save_results,
                results_dir=results_dir
            )
            
            return results
        
        return execute_with_enhancements