import os
import anthropic
from typing import List, Dict, Optional, Any, Tuple, Set, Callable
from dataclasses import dataclass, field
import networkx as nx
import json
import time
import uuid
from dotenv import load_dotenv

from incubator.llm.llmclient import LLMClient
from incubator.messages.message import Message

class AnthropicClient(LLMClient):
    """Client for Anthropic's Claude API"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def generate_response(self, messages: List[Message], system_prompt: str = "", 
                         config: Dict[str, Any] = None) -> str:
        """Generate a response using Anthropic's Claude API"""
        if config is None:
            config = {}
        
        # Convertir todos los mensajes a formato API con roles válidos
        api_messages = []
        for m in messages:
            # Convertir mensaje a formato API
            msg_dict = m.to_api_format()
            
            # Asegurar que todos los mensajes tengan un rol válido para la API de Anthropic
            # que solo acepta "user" o "assistant"
            if msg_dict.get("role") not in ["user", "assistant"]:
                # Si el rol no es válido, convertirlo a "user"
                msg_dict["role"] = "user"
                
                # Opcionalmente, podemos preservar el rol original en el contenido
                # msg_dict["content"] = f"[Del agente {m.role}]: {msg_dict.get('content', '')}"
            
            api_messages.append(msg_dict)
        
        # Default configuration
        params = {
            "model": config.get("model", "claude-3-7-sonnet-20250219"),
            "max_tokens": config.get("max_tokens", 20000),
            "system": system_prompt,
            "messages": api_messages  # Usar los mensajes convertidos
        }
        
        # Apply optional parameters
        if config.get("use_thinking", False):
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": config.get("thinking_budget", 10000)
            }
        else:
            if "temperature" in config:
                params["temperature"] = config["temperature"]
            if "top_p" in config and config["top_p"] != 1.0:
                params["top_p"] = config["top_p"]
            if "top_k" in config and config["top_k"] is not None:
                params["top_k"] = config["top_k"]
        
        try:
            response = ""
            # Stream the response
            with self.client.messages.stream(**params) as stream:
                for event in stream:
                    if event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            response += event.delta.text
                        elif event.delta.type == "thinking_delta" and config.get("use_thinking", False):
                            # Optional: capture model's thinking
                            # thinking_content += event.delta.thinking
                            pass
            
            return response
            
        except Exception as e:
            error_msg = f"Error calling the API: {e}"
            print(error_msg)
            return f"Error: {e}"