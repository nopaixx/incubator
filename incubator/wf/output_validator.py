import os
from typing import List, Dict, Optional, Any, Tuple, Set, Callable
import re
import json

class OutputValidator:
    """
    Valida y corrige outputs producidos por nodos en un workflow
    para asegurar que todos los puertos esperados estén presentes y
    que el contenido tenga un formato adecuado.
    """
    
    @staticmethod
    def validate_node_output(outputs: Dict[str, Dict[str, Any]], 
                            expected_ports: List[str] = None,
                            content_validators: Dict[str, Callable] = None) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Valida los outputs de un nodo
        
        Args:
            outputs: Diccionario de outputs del nodo {port_name: {"content": content, "metadata": metadata}}
            expected_ports: Lista de nombres de puertos que se esperan (opcional)
            content_validators: Diccionario de funciones validadoras para cada puerto {port_name: validator_function}
            
        Returns:
            Tupla de (es_válido, lista_de_errores, metadatos_de_validación)
        """
        is_valid = True
        errors = []
        validation_meta = {
            "missing_ports": [],
            "invalid_content": {},
            "empty_content": []
        }
        
        # Verificar presencia de puertos esperados
        if expected_ports:
            missing_ports = [port for port in expected_ports if port not in outputs]
            if missing_ports:
                is_valid = False
                for port in missing_ports:
                    errors.append(f"Puerto esperado '{port}' no encontrado en outputs")
                validation_meta["missing_ports"] = missing_ports
        
        # Verificar contenido con validadores específicos
        if content_validators:
            for port_name, validator in content_validators.items():
                if port_name in outputs:
                    content = outputs[port_name].get("content", "")
                    # Verificar si el contenido está vacío
                    if not content:
                        is_valid = False
                        errors.append(f"Contenido vacío en puerto '{port_name}'")
                        validation_meta["empty_content"].append(port_name)
                        continue
                    
                    # Aplicar validador específico si el contenido no está vacío
                    try:
                        if not validator(content):
                            is_valid = False
                            errors.append(f"Validación fallida para contenido en puerto '{port_name}'")
                            validation_meta["invalid_content"][port_name] = "No cumple criterios de validación"
                    except Exception as e:
                        is_valid = False
                        errors.append(f"Error al validar contenido en puerto '{port_name}': {str(e)}")
                        validation_meta["invalid_content"][port_name] = str(e)
        
        # Verificar que todos los puertos tengan contenido no vacío
        for port_name, port_data in outputs.items():
            content = port_data.get("content", "")
            if not content and port_name not in validation_meta["empty_content"]:
                is_valid = False
                errors.append(f"Contenido vacío en puerto '{port_name}'")
                validation_meta["empty_content"].append(port_name)
        
        return is_valid, errors, validation_meta
    
    @staticmethod
    def create_default_outputs(expected_ports: List[str], 
                              template_content: str = None) -> Dict[str, Dict[str, Any]]:
        """
        Crea outputs por defecto para puertos faltantes
        
        Args:
            expected_ports: Lista de nombres de puertos que se deben crear
            template_content: Contenido plantilla para usar (opcional)
            
        Returns:
            Diccionario de outputs con los puertos esperados
        """
        default_outputs = {}
        default_content = template_content or "No se generó contenido para este puerto"
        
        for port in expected_ports:
            default_outputs[port] = {
                "content": default_content,
                "metadata": {"auto_generated": True, "reason": "puerto_faltante"}
            }
        
        return default_outputs
    
    @staticmethod
    def ensure_all_ports(outputs: Dict[str, Dict[str, Any]], 
                         expected_ports: List[str],
                         template_message: str = None) -> Dict[str, Dict[str, Any]]:
        """
        Asegura que todos los puertos esperados estén presentes,
        creando los faltantes con contenido predeterminado
        
        Args:
            outputs: Diccionario actual de outputs
            expected_ports: Lista de puertos que deben existir
            template_message: Mensaje a usar para puertos faltantes (opcional)
            
        Returns:
            Diccionario actualizado con todos los puertos esperados
        """
        # Copiar para no modificar el original
        updated_outputs = outputs.copy()
        
        # Identificar puertos faltantes
        missing_ports = [port for port in expected_ports if port not in updated_outputs]
        
        # Crear contenido para puertos faltantes
        if missing_ports:
            default_message = template_message or "Este puerto no generó contenido en la ejecución actual"
            for port in missing_ports:
                updated_outputs[port] = {
                    "content": default_message,
                    "metadata": {"auto_generated": True, "reason": "puerto_faltante"}
                }
        
        return updated_outputs
    
    @staticmethod
    def extract_marked_content(content: str, markers: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """
        Extrae secciones marcadas de un contenido y las asigna a puertos específicos
        
        Args:
            content: Texto completo del que extraer contenido marcado
            markers: Diccionario de {nombre_puerto: marca_texto}
            
        Returns:
            Diccionario de outputs con el contenido extraído
        """
        extracted_outputs = {}
        
        for port_name, marker in markers.items():
            # Intentar encontrar el marcador en el contenido
            marker_index = content.find(marker)
            
            if marker_index >= 0:
                # Extraer desde el marcador hasta el siguiente marcador o el fin
                start_pos = marker_index + len(marker)
                end_pos = len(content)
                
                # Buscar el siguiente marcador, si existe
                next_markers = [content.find(m, start_pos) for m in markers.values() 
                               if content.find(m, start_pos) > 0]
                if next_markers:
                    end_pos = min(next_markers)
                
                # Extraer y limpiar el contenido
                extracted_content = content[start_pos:end_pos].strip()
                
                # Crear el output para este puerto
                extracted_outputs[port_name] = {
                    "content": extracted_content,
                    "metadata": {
                        "extracted": True, 
                        "marker": marker,
                        "extracted_length": len(extracted_content)
                    }
                }
        
        return extracted_outputs
    
    @staticmethod
    def parse_json_safely(content: str, default_value: Any = None) -> Any:
        """
        Intenta parsear un string como JSON con manejo seguro de errores
        
        Args:
            content: String que se intentará parsear como JSON
            default_value: Valor a devolver si falla el parseo
            
        Returns:
            El objeto JSON parseado o el valor por defecto
        """
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return default_value
    
    @staticmethod
    def extract_conclusion_content(content: str, conclusion_markers: List[str] = None) -> Optional[str]:
        """
        Extrae contenido de conclusión usando varios marcadores posibles
        
        Args:
            content: Texto completo del que extraer la conclusión
            conclusion_markers: Lista de posibles marcadores de conclusión
            
        Returns:
            Texto de conclusión o None si no se encuentra
        """
        markers = conclusion_markers or [
            "CONCLUSIÓN:", "CONCLUSION:", "CONCLUSIÓN FINAL:", "CONCLUSION FINAL:",
            "VEREDICTO:", "RESULTADO FINAL:", "RESULTADO:", "DECISIÓN FINAL:", "DECISION FINAL:"
        ]
        
        for marker in markers:
            marker_index = content.find(marker)
            if marker_index >= 0:
                # Extraer desde el marcador hasta el fin o un separador claro
                start_pos = marker_index + len(marker)
                
                # Buscar posibles terminadores como doble salto de línea o un nuevo encabezado
                end_markers = ["\n\n", "\r\n\r\n"]
                end_positions = [content.find(em, start_pos) for em in end_markers if content.find(em, start_pos) > 0]
                
                # También buscar un nuevo encabezado (línea que empieza con #)
                header_match = re.search(r'\n#+\s', content[start_pos:])
                if header_match:
                    end_positions.append(start_pos + header_match.start())
                
                # Determinar dónde termina la conclusión
                end_pos = len(content)
                if end_positions:
                    end_pos = min(p for p in end_positions if p > 0) + start_pos
                
                # Extraer y limpiar
                conclusion = content[start_pos:end_pos].strip()
                return conclusion
        
        return None