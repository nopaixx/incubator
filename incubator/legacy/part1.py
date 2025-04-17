import os
import json
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import anthropic

# Cargar variables de entorno (para la API_KEY de Claude)
load_dotenv()

class Mensaje:
    """Clase para representar un mensaje en la conversación entre agentes"""
    
    def __init__(self, rol: str, contenido: str):
        # Normalizar el rol para la API de Anthropic (cambiar 'human' a 'user')
        if rol == 'human':
            self.rol = 'user'
        else:
            self.rol = rol
        self.contenido = contenido
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "role": self.rol,
            "content": self.contenido
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "Mensaje":
        return cls(data["role"], data["content"])


class Agente:
    """Clase base para todos los agentes del sistema"""
    
    def __init__(self, nombre: str, descripcion: str, api_key: Optional[str] = None):
        self.nombre = nombre
        self.descripcion = descripcion
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.modelo = "claude-3-7-sonnet-20250219"  # Modelo actual de Claude
        self.cliente = anthropic.Anthropic(api_key=self.api_key)
    
    def generar_respuesta(self, mensajes: List[Mensaje], system_prompt: str = "", max_tokens: int = 5000) -> str:
        """Genera una respuesta utilizando el API de Claude a través de la biblioteca anthropic"""
        try:
            # Convertir mensajes al formato esperado por la API
            mensajes_api = [m.to_dict() for m in mensajes]
            
            # Realizar la llamada a la API usando la biblioteca anthropic
            respuesta = self.cliente.messages.create(
                model=self.modelo,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=mensajes_api
            )
            
            # Extraer el texto de la respuesta
            return respuesta.content[0].text
        
        except Exception as e:
            print(f"Error al llamar a la API: {e}")
            return f"Error: {e}"
    
    def pensar(self, contexto: List[Mensaje]) -> str:
        """Método que debe ser implementado por cada agente específico"""
        raise NotImplementedError("Los agentes específicos deben implementar este método")


class Explorador(Agente):
    """Agente encargado de generar ideas originales de estrategias cuantitativas para hedge funds"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            nombre="Explorador",
            descripcion="Genero ideas originales de estrategias cuantitativas para hedge funds enfocadas en el S&P 500.",
            api_key=api_key
        )
    
    def pensar(self, contexto: List[Mensaje]) -> str:
        """Genera ideas creativas basadas en el contexto proporcionado"""
        # Instrucciones del sistema
        system_prompt = """
        Eres el Agente Explorador en un sistema de incubación de ideas para hedge funds cuantitativos. 
        Tu especialidad es generar estrategias algorítmicas innovadoras centradas en el S&P 500. Tu función es:
        
        1. Generar 3-5 ideas de estrategias cuantitativas innovadoras y detalladas para operar en el S&P 500
        2. Explicar los fundamentos matemáticos/estadísticos y el funcionamiento de cada estrategia
        3. Detallar las señales específicas, timeframes, factores y métricas a utilizar
        4. Destacar las ventajas potenciales de cada estrategia (alpha, Sharpe, drawdown, etc.)
        5. Responder a las preguntas o sugerencias del Curador
        
        Considera aspectos como:
        - Factores de mercado (momentum, valor, volatilidad, etc.)
        - Análisis estadísticos (cointegración, regresión, clusterización)
        - Indicadores técnicos o fundamentales innovadores
        - Optimización de ejecución y manejo de costos de transacción
        - Gestión de riesgo y diversificación de la estrategia
        
        Sé específico, técnico y detallado. Piensa en estrategias implementables y backtestables.
        """
        
        return self.generar_respuesta(contexto, system_prompt)


class Curador(Agente):
    """Agente encargado de evaluar estrategias cuantitativas y seleccionar la más prometedora"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            nombre="Curador",
            descripcion="Evalúo estrategias cuantitativas para hedge funds, selecciono la más prometedora y propongo mejoras.",
            api_key=api_key
        )
    
    def pensar(self, contexto: List[Mensaje]) -> str:
        """Evalúa las ideas del Explorador y selecciona/refina la más prometedora"""
        # Instrucciones del sistema
        system_prompt = """
        Eres el Agente Curador en un sistema de incubación de ideas para hedge funds cuantitativos.
        Tienes amplia experiencia en evaluación de estrategias algorítmicas enfocadas en el S&P 500.
        Tu función es evaluar las estrategias propuestas por el Explorador, seleccionar la más prometedora
        y proponer mejoras. Debes:
        
        1. Analizar críticamente cada estrategia considerando:
           - Ratio de Sharpe esperado y robustez estadística
           - Capacidad para generar alpha verdadero (no beta disfrazado)
           - Escalabilidad y capacidad (cuánto capital puede manejar)
           - Costos de implementación y transacción
           - Exposición a factores de riesgo conocidos
           - Riesgo de sobreoptimización o data snooping
        
        2. Seleccionar la estrategia con mayor potencial de éxito en entorno real
        
        3. Proponer mejoras específicas y técnicas como:
           - Refinamiento de parámetros o señales
           - Mejoras en la gestión de riesgo
           - Optimización de ejecución
           - Complementos con otros factores o señales
        
        4. Formular preguntas técnicas específicas para aclarar aspectos ambiguos
        
        5. Después de suficiente refinamiento, finalizar con una versión detallada y técnicamente
           sólida de la estrategia seleccionada, lista para ser implementada. Incluir:
           - Lógica exacta de entrada/salida
           - Parámetros específicos
           - Gestión de riesgo
           - Expectativas de desempeño
        
        Identifica claramente la estrategia final con el prefijo "IDEA FINAL:" cuando hayas
        llegado a una conclusión satisfactoria.
        
        Mantén un enfoque riguroso y escéptico, como lo haría un gestor de riesgos experimentado.
        """
        
        return self.generar_respuesta(contexto, system_prompt)


class Conversacion:
    """Clase para gestionar la conversación entre el Explorador y el Curador"""
    
    def __init__(self, semilla: str, max_turnos: int = 5):
        self.explorador = Explorador()
        self.curador = Curador()
        self.semilla = semilla
        self.max_turnos = max_turnos
        self.mensajes: List[Mensaje] = []
        self.idea_final: Optional[str] = None
    
    def iniciar(self) -> Tuple[List[Mensaje], Optional[str]]:
        """Inicia la conversación entre el Explorador y el Curador"""
        # Mensaje inicial con la semilla
        mensaje_inicial = Mensaje("user", f"Semilla para generación de ideas: {self.semilla}")
        self.mensajes.append(mensaje_inicial)
        
        # Primera intervención del Explorador
        respuesta_explorador = self.explorador.pensar(self.mensajes)
        self.mensajes.append(Mensaje("assistant", f"🧪 Explorador: {respuesta_explorador}"))
        
        # Comienza la conversación
        for turno in range(self.max_turnos):
            print(f"Turno {turno+1}/{self.max_turnos}")
            
            # Turno del Curador
            respuesta_curador = self.curador.pensar(self.mensajes)
            self.mensajes.append(Mensaje("user", f"🧐 Curador: {respuesta_curador}"))
            
            # Verificar si el Curador ha llegado a una decisión final
            if "IDEA FINAL:" in respuesta_curador:
                # Extraer la idea final
                idx_inicio = respuesta_curador.find("IDEA FINAL:") + len("IDEA FINAL:")
                self.idea_final = respuesta_curador[idx_inicio:].strip()
                print("¡Idea final seleccionada!")
                break
            
            # Turno del Explorador
            respuesta_explorador = self.explorador.pensar(self.mensajes)
            self.mensajes.append(Mensaje("assistant", f"🧪 Explorador: {respuesta_explorador}"))
        
        # Si llegamos al máximo de turnos sin decisión, pedimos al Curador que tome una decisión final
        if not self.idea_final:
            mensaje_forzar_decision = Mensaje("user", 
                "Hemos alcanzado el máximo de turnos. Por favor, selecciona la idea más prometedora y preséntala como IDEA FINAL.")
            self.mensajes.append(mensaje_forzar_decision)
            
            decision_final = self.curador.pensar(self.mensajes)
            self.mensajes.append(Mensaje("user", f"🧐 Curador: {decision_final}"))
            
            # Extraer la idea final
            if "IDEA FINAL:" in decision_final:
                idx_inicio = decision_final.find("IDEA FINAL:") + len("IDEA FINAL:")
                self.idea_final = decision_final[idx_inicio:].strip()
            else:
                # Si aún no hay formato claro, tomamos toda la respuesta
                self.idea_final = decision_final
        
        # Guardar la conversación y la idea final
        self.guardar_resultados()
        
        return self.mensajes, self.idea_final
    
    def guardar_resultados(self) -> None:
        """Guarda la conversación y la idea final en archivos"""
        # Crear directorio de resultados si no existe
        os.makedirs("artifacts", exist_ok=True)
        
        # Guardar conversación
        with open("artifacts/conversacion.json", "w", encoding="utf-8") as f:
            json.dump([m.to_dict() for m in self.mensajes], f, indent=2, ensure_ascii=False)
        
        # Guardar idea final
        if self.idea_final:
            with open("artifacts/idea.txt", "w", encoding="utf-8") as f:
                f.write(self.idea_final)
        
        print("Resultados guardados en la carpeta 'artifacts'")


# Ejemplo de uso
if __name__ == "__main__":
    # Semilla de ejemplo (se puede cambiar)
    semilla = "Estrategias de arbitraje en criptomonedas utilizando datos de order book"
    
    print(f"Iniciando incubadora de ideas con semilla: {semilla}")
    conversacion = Conversacion(semilla=semilla, max_turnos=3)
    mensajes, idea_final = conversacion.iniciar()
    
    print("\n--- CONVERSACIÓN COMPLETA ---")
    for m in mensajes:
        print(f"\n{m.rol}: {m.contenido}\n")
    
    print("\n--- IDEA FINAL ---")
    print(idea_final)