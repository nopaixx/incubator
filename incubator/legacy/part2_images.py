import os
import json
import base64
from typing import List, Dict, Any, Optional, Tuple
import mimetypes
from datetime import datetime
import anthropic
from dotenv import load_dotenv

# Cargar variables de entorno (para la API_KEY de Claude)
load_dotenv()

# Importar la clase Mensaje desde part1.py
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from incubator.part1 import Agente

class EvaluadorConImagenes(Agente):
    """Agente evaluador que puede procesar imágenes de visualizaciones"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            nombre="Evaluador",
            descripcion="Evalúo los resultados de las estrategias cuantitativas y sugiero ajustes.",
            api_key=api_key
        )
    
    def obtener_imagenes_recientes(self, max_imagenes: int = 3) -> List[Dict[str, Any]]:
        """Obtiene las imágenes más recientes generadas por la ejecución del código"""
        imagenes_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "imagenes")
        
        if not os.path.exists(imagenes_dir):
            return []
        
        # Listar archivos en el directorio de imágenes
        archivos = [os.path.join(imagenes_dir, f) for f in os.listdir(imagenes_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Ordenar por fecha de modificación (más recientes primero)
        archivos.sort(key=os.path.getmtime, reverse=True)
        
        # Limitar al máximo número de imágenes
        archivos = archivos[:max_imagenes]
        
        imagenes = []
        for archivo in archivos:
            # Determinar el tipo MIME
            mime_type, _ = mimetypes.guess_type(archivo)
            if not mime_type:
                mime_type = 'image/png'  # Valor predeterminado
            
            # Leer la imagen y convertirla a base64
            with open(archivo, "rb") as f:
                imagen_data = f.read()
                imagen_base64 = base64.b64encode(imagen_data).decode('utf-8')
            
            imagenes.append({
                "ruta": archivo,
                "mime_type": mime_type,
                "base64": imagen_base64,
                "nombre": os.path.basename(archivo)
            })
        
        return imagenes
    
    def evaluar_con_imagenes(self, contexto_texto: str, metricas: Dict[str, Any]) -> str:
        """Evalúa los resultados incluyendo las imágenes generadas en la llamada a la API"""
        # Instrucciones del sistema
        system_prompt = """
        Eres el Agente Evaluador en un sistema de incubación de ideas para hedge funds cuantitativos.
        Tu tarea es analizar los resultados de la ejecución de la estrategia, evaluar su desempeño y proponer
        mejoras o ajustes. Tu análisis debe ser riguroso y orientado a mejorar la estrategia.
        
        Se te proporcionarán visualizaciones de la estrategia y métricas numéricas de rendimiento.
        
        Debes enfocarte en:
        
        1. Evaluación de métricas clave:
           - Sharpe Ratio (>1 es bueno, >2 es excelente)
           - Drawdown (menor es mejor, <20% preferible)
           - Rentabilidad anualizada (comparar con benchmark)
           - Otros ratios relevantes (Sortino, Calmar, etc.)
        
        2. Análisis de las visualizaciones:
           - Interpretar las gráficas proporcionadas
           - Identificar patrones en los rendimientos
           - Analizar comportamiento en diferentes regímenes de mercado
           - Evaluar distribución de retornos y drawdowns
        
        3. Análisis de robustez:
           - Consistencia del rendimiento
           - Potenciales sesgos o problemas estadísticos
           - Sensibilidad a condiciones de mercado
        
        4. Recomendaciones concretas:
           - Ajustes específicos de parámetros (sugerir valores)
           - Mejoras en la gestión de riesgo
           - Filtros adicionales para mejorar la calidad de las señales
           - Variables o factores adicionales a considerar
           - Métricas o visualizaciones adicionales necesarias
        
        Basándote en tu evaluación, debes concluir con uno de estos veredictos:
        
        - "ESTRATEGIA APROBADA: La estrategia cumple con los criterios de desempeño y está lista para consideración seria."
        - "ESTRATEGIA CON AJUSTES: La estrategia muestra potencial pero necesita los siguientes ajustes específicos: [listar ajustes]"
        - "ESTRATEGIA RECHAZADA: La estrategia no cumple con los criterios mínimos por las siguientes razones específicas: [listar razones]"
        
        Si sugieres ajustes, sé específico y detallado para que el Ingeniero pueda implementarlos directamente.
        """
        
        # Obtener imágenes recientes
        imagenes = self.obtener_imagenes_recientes(max_imagenes=3)
        
        # Construir el mensaje con métricas
        metricas_texto = "MÉTRICAS OBTENIDAS:\n"
        for clave, valor in metricas.items():
            metricas_texto += f"- {clave}: {valor}\n"
        
        # Contenido del mensaje completo
        contenido_mensaje = [
            {"type": "text", "text": f"Evaluación de la estrategia de trading cuantitativa:\n\n{contexto_texto}\n\n{metricas_texto}"}
        ]
        
        # Añadir las imágenes al contenido
        for i, imagen in enumerate(imagenes):
            # Agregar separador de texto
            contenido_mensaje.append(
                {"type": "text", "text": f"\nVisualización {i+1} - {imagen['nombre']}:"}
            )
            
            # Agregar la imagen
            contenido_mensaje.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": imagen["mime_type"],
                    "data": imagen["base64"]
                }
            })
        
        # Agregar instrucción final
        contenido_mensaje.append({
            "type": "text", 
            "text": "\nPor favor, analiza estas visualizaciones junto con las métricas proporcionadas y ofrece tu evaluación completa de la estrategia."
        })
        
        try:
            # Realizar la llamada a la API usando la biblioteca anthropic con mensajes que incluyen imágenes
            respuesta = self.cliente.messages.create(
                model=self.modelo,
                max_tokens=1024,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": contenido_mensaje}
                ]
            )
            
            # Extraer el texto de la respuesta
            return respuesta.content[0].text
        
        except Exception as e:
            print(f"Error al llamar a la API con imágenes: {e}")
            
            # Si falla la evaluación con imágenes, intentar sin ellas
            return self.generar_respuesta(
                [{"role": "user", "content": f"Evaluación de la estrategia de trading cuantitativa:\n\n{contexto_texto}\n\n{metricas_texto}"}], 
                system_prompt
            )

def extraer_metricas_del_ejecutor(respuesta_ejecutor: str) -> Dict[str, Any]:
    """Extrae las métricas del texto de respuesta del Ejecutor"""
    metricas = {}
    
    # Buscar la sección de métricas
    if "MÉTRICAS OBTENIDAS:" in respuesta_ejecutor:
        lineas = respuesta_ejecutor.split("\n")
        capturando = False
        
        for linea in lineas:
            if "MÉTRICAS OBTENIDAS:" in linea:
                capturando = True
                continue
            
            if capturando and linea.strip() == "":
                capturando = False
                continue
            
            if capturando and "- " in linea:
                # Formato esperado: "- nombre_metrica: valor"
                partes = linea.split(":", 1)
                if len(partes) == 2:
                    nombre = partes[0].replace("- ", "").strip()
                    valor_str = partes[1].strip()
                    
                    # Intentar convertir a número si es posible
                    try:
                        # Intentar como float
                        valor = float(valor_str)
                    except ValueError:
                        # Mantener como string si no se puede convertir
                        valor = valor_str
                    
                    metricas[nombre] = valor
    
    return metricas

def integrar_evaluador_con_imagenes(mensajes_contexto: List[Dict[str, Any]], respuesta_ejecutor: str) -> str:
    """Integra el evaluador con capacidad de procesar imágenes al flujo de la conversación"""
    # Extraer métricas de la respuesta del ejecutor
    metricas = extraer_metricas_del_ejecutor(respuesta_ejecutor)
    
    # Crear instancia del evaluador
    evaluador = EvaluadorConImagenes()
    
    # Combinar todo el contexto en un solo texto
    contexto_texto = ""
    for mensaje in mensajes_contexto:
        rol = mensaje.get("role", "")
        contenido = mensaje.get("content", "")
        if isinstance(contenido, str):
            # Si es conversación normal, añadir el rol y contenido
            if "Ingeniero" in contenido or "Revisor" in contenido or "Ejecutor" in contenido:
                contexto_texto += f"{contenido}\n\n"
    
    # Añadir la respuesta del ejecutor
    contexto_texto += f"⚙️ Ejecutor: {respuesta_ejecutor}\n\n"
    
    # Obtener evaluación incluyendo imágenes
    evaluacion = evaluador.evaluar_con_imagenes(contexto_texto, metricas)
    
    return evaluacion

if __name__ == "__main__":
    # Ejemplo de uso
    evaluador = EvaluadorConImagenes()
    imagenes = evaluador.obtener_imagenes_recientes()
    print(f"Se encontraron {len(imagenes)} imágenes recientes")
    
    # Mostrar las rutas de las imágenes
    for img in imagenes:
        print(f"- {img['nombre']} ({img['mime_type']})")