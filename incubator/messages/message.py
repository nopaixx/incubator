# File: incubator/messages/message.py
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass

@dataclass
class Message:
    """
    Clase base para representar mensajes en el sistema de agentes.
    Compatible con la API de Anthropic y otras APIs de LLM.
    """
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    
    def to_api_format(self) -> Dict[str, Any]:
        """
        Convierte el mensaje al formato adecuado para la API.
        Los formatos compatibles con la API de Anthropic son:
        - role: "user" o "assistant"
        - content: string con el contenido del mensaje
        """
        msg = {
            "role": self.role,
            "content": self.content
        }
        
        # Añadimos metadata si existe
        if self.metadata:
            msg["metadata"] = self.metadata
            
        return msg
    
    def __str__(self) -> str:
        """Representación en string del mensaje"""
        return self.content
        
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Message':
        """Crea un mensaje a partir de un diccionario"""
        return Message(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            metadata=data.get("metadata")
        )
    
    @staticmethod
    def format_conversation(messages: List['Message']) -> str:
        """
        Formatea una lista de mensajes como una conversación
        para presentación o debugging.
        """
        formatted = ""
        for msg in messages:
            formatted += f"[{msg.role}]: {msg.content}\n\n"
        return formatted
    
    @staticmethod
    def combine_messages(messages: List['Message'], separator: str = "\n\n") -> str:
        """
        Combina el contenido de varios mensajes en uno solo.
        Útil para consolidar inputs de múltiples fuentes.
        """
        contents = [msg.content for msg in messages if msg.content]
        return separator.join(contents)