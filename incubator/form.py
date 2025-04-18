from typing import List, Optional, Dict, Any
from incubator.messages.message import Message

class InputFormatter:
    """
    Interfaz base para formateadores de entrada.
    Los formateadores se encargan de combinar múltiples mensajes en uno solo.
    """
    
    def format(self, inputs: List[Message], context: Optional[Dict[str, Any]] = None) -> str:
        """
        Formatea una lista de mensajes de entrada en un único mensaje.
        
        Args:
            inputs: Lista de mensajes a formatear
            context: Contexto opcional para el formateo
            
        Returns:
            El mensaje formateado como string
        """
        raise NotImplementedError("Subclasses must implement this method")

class DefaultFormatter(InputFormatter):
    """
    Formateador por defecto que combina los mensajes con separadores claros.
    """
    
    def format(self, inputs: List[Message], context: Optional[Dict[str, Any]] = None) -> str:
        """Formatea múltiples mensajes con una estructura clara"""
        if len(inputs) == 1:
            return inputs[0].content
            
        formatted = "=== ENTRADAS MÚLTIPLES ===\n\n"
        for i, msg in enumerate(inputs, 1):
            # Incluir información del rol si está disponible
            role_info = f" (de {msg.role})" if msg.role != "user" else ""
            formatted += f"ENTRADA #{i}{role_info}:\n"
            formatted += f"{msg.content}\n\n"
            
        return formatted

# Estos formatters específicos pueden implementarse por los usuarios según necesidades
class IdeadorCuradorFormatter(InputFormatter):
    """Formatter específico para combinar inputs del ideador y curador hacia el sintetizador"""
    
    def format(self, inputs: List[Message], context: Optional[Dict[str, Any]] = None) -> str:
        """Formato especializado para la síntesis de ideas"""
        # Encontrar los mensajes específicos
        ideador_msg = next((msg for msg in inputs if msg.role == "ideador"), None)
        curador_msg = next((msg for msg in inputs if msg.role == "curador"), None)
        
        if ideador_msg and curador_msg:
            formatted = "=== SÍNTESIS DE IDEA ===\n\n"
            formatted += "PROPUESTA FINAL DEL IDEADOR:\n"
            formatted += f"{ideador_msg.content}\n\n"
            formatted += "FEEDBACK FINAL DEL CURADOR:\n"
            formatted += f"{curador_msg.content}\n\n"
            formatted += "Tu tarea es sintetizar ambas contribuciones en una IDEA FINAL coherente y completa.\n"
            return formatted
        else:
            # Si faltan mensajes esperados, usar el formato predeterminado
            return DefaultFormatter().format(inputs, context)

class CodigoRevisorFormatter(InputFormatter):
    """Formatter específico para combinar inputs del sintetizador y desarrollador hacia el revisor"""
    
    def format(self, inputs: List[Message], context: Optional[Dict[str, Any]] = None) -> str:
        """Formato especializado para revisión de código"""
        # Encontrar los mensajes específicos
        idea_msg = next((msg for msg in inputs if msg.role == "sintetizador"), None)
        codigo_msg = next((msg for msg in inputs if msg.role == "desarrollador"), None)
        
        if idea_msg and codigo_msg:
            formatted = "=== REVISIÓN DE CÓDIGO ===\n\n"
            formatted += "IDEA CONCEPTUAL (sintetizada):\n"
            formatted += f"{idea_msg.content}\n\n"
            formatted += "IMPLEMENTACIÓN EN CÓDIGO (del desarrollador):\n"
            formatted += f"{codigo_msg.content}\n\n"
            formatted += "Por favor analiza si el código implementa correctamente la idea conceptual y sugiere mejoras específicas.\n"
            return formatted
        else:
            # Si faltan mensajes esperados, usar el formato predeterminado
            return DefaultFormatter().format(inputs, context)