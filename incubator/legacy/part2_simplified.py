import os
import json
import tempfile
import matplotlib
matplotlib.use('Agg')  # Configuración para entornos sin GUI
import matplotlib.pyplot as plt
import base64
from io import BytesIO, StringIO
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import anthropic
import importlib.util
import sys
import traceback
from datetime import datetime

# Cargar variables de entorno (para la API_KEY de Claude)
load_dotenv()

# Importar la clase Mensaje desde part1.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from incubator.part1 import Mensaje, Agente


class Ingeniero(Agente):
    """Agente encargado de implementar estrategias cuantitativas en código Python"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            nombre="Ingeniero",
            descripcion="Implemento estrategias cuantitativas en código Python para hedge funds.",
            api_key=api_key
        )
    
    def pensar(self, contexto: List[Mensaje]) -> str:
        """Implementa la idea en código Python"""
        # Instrucciones del sistema
        system_prompt = """
        Eres el Agente Ingeniero en un sistema de incubación de ideas para hedge funds cuantitativos.
        Tu tarea es transformar una estrategia cuantitativa conceptual en código Python funcional.
        
        Debes:
        
        1. Implementar la estrategia usando bibliotecas estándar y APIs públicas:
           - pandas, numpy, scipy para análisis de datos
           - yfinance para obtener datos de mercado del S&P 500
           - matplotlib, seaborn para visualizaciones
           - sklearn para modelos estadísticos si es necesario
        
        2. Estructurar el código de manera modular y bien organizada:
           - Funciones para obtener datos
           - Funciones para aplicar la lógica de trading
           - Funciones para backtesting/evaluación
           - Funciones para visualización de resultados
        
        3. Incluir documentación clara con:
           - Docstrings explicativos
           - Comentarios en secciones complejas
           - Parámetros bien definidos
        
        4. Implementar mecanismos de backtesting rigurosos:
           - División de datos en training/testing cuando sea apropiado
           - Prevención de look-ahead bias
           - Métricas de evaluación (Sharpe, Sortino, drawdown, etc.)
           
        5. IMPORTANTE: El código debe ser completamente ejecutable. Incluye una función 'main()' que:
           - Ejecute todos los pasos de la estrategia
           - Genere y muestre visualizaciones (plt.figure, etc.)
           - Calcule y MUESTRE EXPLÍCITAMENTE las métricas clave (Sharpe ratio, returns, drawdown, etc.)
           - Guarde estas métricas en variables globales con nombres claros (sharpe_ratio, max_drawdown, etc.)
           
        6. Las visualizaciones son críticas. Crea al menos:
           - Gráfico de rendimiento acumulativo vs. benchmark
           - Análisis de drawdowns
           - Distribución de retornos o métricas relevantes
           
        7. Responder a las sugerencias del Revisor o del Evaluador, adaptando el código apropiadamente
        
        El código debe ser ejecutable sin necesidad de modificaciones adicionales.
        Prioriza la claridad, robustez y evitar errores como data snooping o overfitting.
        
        Comienza tu respuesta con "```python" y termina con "```" para marcar claramente el código.
        ```python
        """
        
        return self.generar_respuesta(contexto, system_prompt)


class Revisor(Agente):
    """Agente encargado de revisar y mejorar el código de implementación, con mayor flexibilidad"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            nombre="Revisor",
            descripcion="Reviso y sugiero mejoras para el código de estrategias cuantitativas.",
            api_key=api_key
        )
        self.iteracion = 0  # Contador de iteraciones
    
    def pensar(self, contexto: List[Mensaje]) -> str:
        """Evalúa el código del Ingeniero y sugiere mejoras, siendo más flexible con el paso de iteraciones"""
        # Incrementar contador de iteraciones
        self.iteracion += 1
        
        # Instrucciones del sistema, con flexibilidad progresiva
        system_prompt = f"""
        Eres el Agente Revisor en un sistema de incubación de ideas para hedge funds cuantitativos.
        Tu tarea es revisar el código Python implementado por el Ingeniero, identificar problemas
        y sugerir mejoras. Esta es la iteración {self.iteracion}.
        
        Debes enfocarte en:
        
        1. Errores críticos:
           - Bugs o errores de sintaxis que impedirían la ejecución
           - Problemas graves de lógica que harían que la estrategia no funcione
           
        2. Aspectos importantes:
           - Implementación básica de la estrategia según la idea original
           - Generación de al menos una visualización simple
           - Cálculo de métricas básicas como Sharpe ratio o rendimiento
        
        Después de la iteración 2, debes ser más flexible y aprobar el código si:
        - No tiene errores de sintaxis graves
        - Implementa de manera básica la idea solicitada
        - Genera al menos una visualización
        - Calcula al menos una métrica básica
        
        Si consideras que el código está listo para ser ejecutado, indícalo claramente con la frase:
        "CÓDIGO APROBADO: El código está listo para ser ejecutado."
        
        Si esta es la iteración 3 o superior y el código no tiene errores críticos que
        impedirían su ejecución, DEBES aprobarlo aunque tenga áreas de mejora, para permitir
        que pase a la fase de ejecución. En este caso, incluye la frase:
        "CÓDIGO APROBADO: Aunque hay áreas de mejora, el código es funcional y está listo para ejecución."
        
        De lo contrario, detalla claramente tus sugerencias y correcciones más importantes.
        """
        
        return self.generar_respuesta(contexto, system_prompt)


class Ejecutor(Agente):
    """Agente encargado de ejecutar el código y generar resultados"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            nombre="Ejecutor",
            descripcion="Ejecuto el código de estrategias y genero resultados visuales.",
            api_key=api_key
        )
    
    def ejecutar_codigo(self, codigo: str) -> Dict[str, Any]:
        """Ejecuta el código Python y captura los resultados"""
        # Crear directorios para los resultados
        resultados_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
        imagenes_dir = os.path.join(resultados_dir, "imagenes")
        os.makedirs(resultados_dir, exist_ok=True)
        os.makedirs(imagenes_dir, exist_ok=True)
        
        # Variables para capturar métricas y resultados
        resultados = {
            "exito": False,
            "salida": "",
            "error": "",
            "figuras": [],
            "metricas": {},
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Cerrar todas las figuras existentes
        plt.close('all')
        
        # Redireccionar stdout para capturar la salida
        stdout_original = sys.stdout
        stdout_capturado = StringIO()
        sys.stdout = stdout_capturado
        
        # Crear un espacio de nombres local para la ejecución
        espacio_local = {}
        
        try:
            # Ejecutar el código
            exec(codigo, espacio_local)
            
            # Si hay una función main, ejecutarla
            if 'main' in espacio_local:
                espacio_local['main']()
            
            # Marcar como exitoso
            resultados["exito"] = True
            
        except Exception as e:
            # Capturar error detallado con traceback
            error_traceback = traceback.format_exc()
            resultados["error"] = f"{str(e)}\n\n{error_traceback}"
            resultados["exito"] = False
        finally:
            # Restaurar stdout
            sys.stdout = stdout_original
            resultados["salida"] = stdout_capturado.getvalue()
            
            # Capturar métricas del espacio local
            metricas_clave = [
                "sharpe_ratio", "sortino_ratio", "max_drawdown", "annual_return", 
                "win_rate", "profit_factor", "beta", "alpha", "volatility",
                "returns", "cumulative_returns", "cagr", "calmar_ratio"
            ]
            
            # Buscar estas métricas en el espacio local
            for clave in metricas_clave:
                if clave in espacio_local:
                    valor = espacio_local[clave]
                    # Convertir np.float a float normal si es necesario
                    if hasattr(valor, "item"):
                        valor = valor.item()
                    resultados["metricas"][clave] = valor
            
            # Capturar también variables que contienen estas palabras clave
            for clave in espacio_local:
                if isinstance(espacio_local[clave], (int, float)) or (
                    hasattr(espacio_local[clave], "item") and 
                    callable(getattr(espacio_local[clave], "item", None))
                ):
                    for metrica in ["sharpe", "sortino", "drawdown", "return", "ratio"]:
                        if metrica in clave.lower() and clave not in resultados["metricas"]:
                            valor = espacio_local[clave]
                            if hasattr(valor, "item"):
                                valor = valor.item()
                            resultados["metricas"][clave] = valor
            
            # Capturar figuras que se hayan generado
            for i, fig in enumerate(plt.get_fignums()):
                figura = plt.figure(fig)
                
                # Guardar imagen
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"figura_{timestamp}_{i}.png"
                ruta_archivo = os.path.join(imagenes_dir, nombre_archivo)
                
                # Guardar la figura con alta resolución
                figura.savefig(ruta_archivo, dpi=300, bbox_inches='tight')
                
                # También guardar en base64 para incluir en JSON
                buffer = BytesIO()
                figura.savefig(buffer, format='png', dpi=100)
                buffer.seek(0)
                imagen_b64 = base64.b64encode(buffer.read()).decode('utf-8')
                
                # Añadir a resultados
                resultados["figuras"].append({
                    "nombre": nombre_archivo,
                    "ruta": ruta_archivo,
                    "base64": imagen_b64
                })
            
            # Si no se generaron figuras, crear una de ejemplo para debug
            if not resultados["figuras"]:
                try:
                    # Crear una figura simple de debug
                    plt.figure(figsize=(8, 6))
                    plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
                    plt.title("Figura de Debug - No se generaron visualizaciones en el código")
                    plt.xlabel("Índice")
                    plt.ylabel("Valor")
                    
                    # Guardar esta figura
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    nombre_archivo = f"figura_debug_{timestamp}.png"
                    ruta_archivo = os.path.join(imagenes_dir, nombre_archivo)
                    plt.savefig(ruta_archivo, dpi=300)
                    
                    # Añadir nota a la salida
                    resultados["salida"] += "\n\nNOTA: No se generaron visualizaciones en el código. Se ha creado una figura de debug."
                    
                    # También guardar en base64
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png', dpi=100)
                    buffer.seek(0)
                    imagen_b64 = base64.b64encode(buffer.read()).decode('utf-8')
                    
                    # Añadir a resultados
                    resultados["figuras"].append({
                        "nombre": nombre_archivo,
                        "ruta": ruta_archivo,
                        "base64": imagen_b64
                    })
                except Exception as e:
                    print(f"Error al crear figura de debug: {e}")
            
            # Guardar resultados en JSON
            with open(os.path.join(resultados_dir, "resultados_ejecucion.json"), "w") as f:
                # Filtrar la clave base64 para no hacer el archivo demasiado grande
                resultado_guardar = {k: v for k, v in resultados.items() if k != "figuras"}
                resultado_guardar["figuras"] = [
                    {k: v for k, v in fig.items() if k != "base64"}
                    for fig in resultados["figuras"]
                ]
                json.dump(resultado_guardar, f, indent=2)
        
        return resultados
    
    def pensar(self, contexto: List[Mensaje]) -> str:
        """Analiza el contexto, extrae el código y lo ejecuta"""
        # Buscar el código Python en el contexto
        codigo = ""
        for mensaje in reversed(contexto):  # Comenzar desde el mensaje más reciente
            contenido = mensaje.contenido
            if "```python" in contenido and "```" in contenido:
                # Extraer el código entre los delimitadores
                inicio = contenido.find("```python") + len("```python")
                fin = contenido.find("```", inicio)
                if fin > inicio:
                    codigo = contenido[inicio:fin].strip()
                    break
        
        if not codigo:
            return "No pude encontrar código Python válido en el contexto de la conversación. Necesito que el Ingeniero proporcione código entre delimitadores ```python ```."
        
        # Ejecutar el código
        resultados = self.ejecutar_codigo(codigo)
        
        # Preparar informe de ejecución
        if resultados["exito"]:
            informe = "✅ EJECUCIÓN EXITOSA\n\n"
            
            # Incluir métricas
            if resultados["metricas"]:
                informe += "📊 MÉTRICAS OBTENIDAS:\n"
                for clave, valor in resultados["metricas"].items():
                    informe += f"- {clave}: {valor}\n"
                informe += "\n"
            else:
                informe += "⚠️ ADVERTENCIA: No se detectaron métricas en la ejecución. El código debería generar métricas como sharpe_ratio, max_drawdown, etc.\n\n"
            
            # Mencionar figuras generadas
            if resultados["figuras"]:
                informe += f"📈 FIGURAS GENERADAS: {len(resultados['figuras'])}\n"
                for i, fig in enumerate(resultados["figuras"]):
                    informe += f"- Figura {i+1}: {fig['nombre']} - guardada en {fig['ruta']}\n"
                informe += "\nLas figuras se han guardado en el directorio 'artifacts/imagenes'.\n\n"
            else:
                informe += "⚠️ ADVERTENCIA: No se generaron figuras durante la ejecución. El código debería crear visualizaciones.\n\n"
            
            # Incluir parte de la salida si es relevante
            if resultados["salida"]:
                salida_truncada = resultados["salida"][:800]
                if len(resultados["salida"]) > 800:
                    salida_truncada += "... (salida truncada)"
                informe += "🖥️ SALIDA DE CONSOLA:\n"
                informe += salida_truncada + "\n\n"
                
            informe += "La estrategia se ha ejecutado correctamente. Los resultados están listos para ser evaluados por el Evaluador."
            
        else:
            informe = "❌ ERROR EN LA EJECUCIÓN\n\n"
            informe += f"Error: {resultados['error']}\n\n"
            
            if resultados["salida"]:
                salida_truncada = resultados["salida"][:500]
                if len(resultados["salida"]) > 500:
                    salida_truncada += "... (salida truncada)"
                informe += "🖥️ SALIDA PARCIAL ANTES DEL ERROR:\n"
                informe += salida_truncada + "\n\n"
                
            informe += "Por favor, revisa el código y corrige los errores antes de volver a ejecutar."
        
        return informe


class Evaluador(Agente):
    """Agente encargado de evaluar los resultados y sugerir mejoras"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            nombre="Evaluador",
            descripcion="Evalúo los resultados de las estrategias cuantitativas y sugiero ajustes.",
            api_key=api_key
        )
    
    def pensar(self, contexto: List[Mensaje]) -> str:
        """Evalúa los resultados de la ejecución y sugiere mejoras"""
        # Instrucciones del sistema
        system_prompt = """
        Eres el Agente Evaluador en un sistema de incubación de ideas para hedge funds cuantitativos.
        Tu tarea es analizar los resultados de la ejecución de la estrategia, evaluar su desempeño y proponer
        mejoras o ajustes. Tu análisis debe ser riguroso y orientado a mejorar la estrategia.
        
        Debes enfocarte en:
        
        1. Evaluación de métricas clave:
           - Sharpe Ratio (>1 es bueno, >2 es excelente)
           - Drawdown (menor es mejor, <20% preferible)
           - Rentabilidad anualizada (comparar con benchmark)
           - Otros ratios relevantes (Sortino, Calmar, etc.)
        
        2. Análisis de los resultados:
           - Comportamiento en diferentes condiciones de mercado
           - Consistencia del rendimiento
           - Potenciales sesgos o problemas estadísticos
           - Robustez de la estrategia
        
        3. Recomendaciones concretas:
           - Ajustes de parámetros
           - Mejoras en la gestión de riesgo
           - Filtros adicionales para mejorar la calidad de las señales
           - Variables o factores adicionales a considerar
        
        Basándote en tu evaluación, debes concluir con uno de estos veredictos:
        
        - "ESTRATEGIA APROBADA: La estrategia cumple con los criterios de desempeño y está lista para consideración seria."
        - "ESTRATEGIA CON AJUSTES: La estrategia muestra potencial pero necesita los siguientes ajustes: [listar ajustes]"
        - "ESTRATEGIA RECHAZADA: La estrategia no cumple con los criterios mínimos por las siguientes razones: [listar razones]"
        
        Si sugieres ajustes, sé específico y detallado para que el Ingeniero pueda implementarlos.
        """
        
        return self.generar_respuesta(contexto, system_prompt)


# Función para extraer métricas de la respuesta del Ejecutor
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


class ImplementacionEstrategia:
    """Clase para gestionar el flujo de implementación de una estrategia cuantitativa"""
    
    def __init__(self, idea: str, max_iteraciones: int = 3):
        self.ingeniero = Ingeniero()
        self.revisor = Revisor()
        self.ejecutor = Ejecutor()
        self.evaluador = Evaluador()
        self.idea = idea
        self.max_iteraciones = max_iteraciones
        self.mensajes: List[Mensaje] = []
        self.codigo_final: Optional[str] = None
        self.resultados_finales: Optional[Dict[str, Any]] = None
        self.veredicto_final: Optional[str] = None
    
    def iniciar(self) -> Tuple[List[Mensaje], Optional[str], Optional[Dict[str, Any]], Optional[str]]:
        """Inicia el flujo de implementación de la estrategia"""
        # Mensaje inicial con la idea a implementar
        mensaje_inicial = Mensaje("user", f"Idea para implementar: {self.idea}")
        self.mensajes.append(mensaje_inicial)
        
        # Primera implementación del Ingeniero
        print("Generando implementación inicial...")
        respuesta_ingeniero = self.ingeniero.pensar(self.mensajes)
        self.mensajes.append(Mensaje("assistant", f"👨‍💻 Ingeniero: {respuesta_ingeniero}"))
        
        # Variables para el ciclo de refinamiento
        iteracion = 0
        codigo_aprobado = False
        estrategia_aprobada = False
        # Ciclo principal de refinamiento
        while iteracion < self.max_iteraciones and not estrategia_aprobada:
            iteracion += 1
            print(f"Iteración {iteracion}/{self.max_iteraciones}")
            
            # Fase 1: Revisión del código
            if not codigo_aprobado:
                print("Revisando el código...")
                respuesta_revisor = self.revisor.pensar(self.mensajes)
                self.mensajes.append(Mensaje("user", f"🔍 Revisor: {respuesta_revisor}"))
                
                # Verificar si el código está aprobado
                if "CÓDIGO APROBADO" in respuesta_revisor:
                    codigo_aprobado = True
                    print("¡Código aprobado! Procediendo a la ejecución...")
                else:
                    # Si estamos en la última iteración, forzar la aprobación
                    if iteracion >= self.max_iteraciones:
                        codigo_aprobado = True
                        print("Forzando aprobación del código para continuar con el proceso...")
                        self.mensajes.append(Mensaje("user", f"🔍 Revisor: CÓDIGO APROBADO FORZOSAMENTE: Continuamos con el proceso para obtener resultados preliminares."))
                    else:
                        # El Ingeniero mejora el código según las sugerencias del Revisor
                        print("Mejorando el código según sugerencias del Revisor...")
                        respuesta_ingeniero = self.ingeniero.pensar(self.mensajes)
                        self.mensajes.append(Mensaje("assistant", f"👨‍💻 Ingeniero: {respuesta_ingeniero}"))
                        continue  # Volver al inicio del ciclo para revisar el nuevo código

            
            # Fase 2: Ejecución del código aprobado
            print("Ejecutando el código...")
            respuesta_ejecutor = self.ejecutor.pensar(self.mensajes)
            self.mensajes.append(Mensaje("user", f"⚙️ Ejecutor: {respuesta_ejecutor}"))
            
            # Verificar si hubo error en la ejecución
            if "ERROR EN LA EJECUCIÓN" in respuesta_ejecutor:
                codigo_aprobado = False  # Volver a fase de revisión
                print("Error en la ejecución. Volviendo a fase de revisión...")
                continue
            
            # Fase 3: Evaluación de resultados
            print("Evaluando resultados...")
            respuesta_evaluador = self.evaluador.pensar(self.mensajes)
            self.mensajes.append(Mensaje("user", f"📊 Evaluador: {respuesta_evaluador}"))
            
            # Verificar el veredicto del Evaluador
            if "ESTRATEGIA APROBADA" in respuesta_evaluador:
                estrategia_aprobada = True
                self.veredicto_final = "APROBADA"
                print("¡Estrategia aprobada! Finalizando proceso.")
                break
            elif "ESTRATEGIA RECHAZADA" in respuesta_evaluador:
                self.veredicto_final = "RECHAZADA"
                print("Estrategia rechazada. Finalizando proceso.")
                break
            else:  # "ESTRATEGIA CON AJUSTES"
                # El Ingeniero ajusta el código según la evaluación
                print("Ajustando el código según evaluación...")
                codigo_aprobado = False  # Volver a fase de revisión completa
                respuesta_ingeniero = self.ingeniero.pensar(self.mensajes)
                self.mensajes.append(Mensaje("assistant", f"👨‍💻 Ingeniero: {respuesta_ingeniero}"))
        
        # Al finalizar el ciclo, extraer el código final
        if iteracion >= self.max_iteraciones:
            print(f"Se alcanzó el máximo de iteraciones ({self.max_iteraciones}). Finalizando proceso.")
            if not self.veredicto_final:
                self.veredicto_final = "PENDIENTE - Máximo de iteraciones alcanzado"
        
        # Extraer el código final
        for mensaje in reversed(self.mensajes):
            if mensaje.rol == "assistant" and "```python" in mensaje.contenido:
                contenido = mensaje.contenido
                inicio = contenido.find("```python") + len("```python")
                fin = contenido.find("```", inicio)
                if fin > inicio:
                    self.codigo_final = contenido[inicio:fin].strip()
                    break
        
        # Guardar resultados finales
        self.guardar_resultados()
        
        return self.mensajes, self.codigo_final, self.resultados_finales, self.veredicto_final
    
    def guardar_resultados(self) -> None:
        """Guarda la conversación, el código final y los resultados"""
        # Crear directorio de resultados si no existe
        resultados_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
        estrategias_dir = os.path.join(resultados_dir, "estrategias")
        
        os.makedirs(resultados_dir, exist_ok=True)
        os.makedirs(estrategias_dir, exist_ok=True)
        
        # Crear un timestamp único para los archivos
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Asegurar que siempre hay un veredicto
        if self.veredicto_final is None:
            self.veredicto_final = "PENDIENTE - No se llegó a un veredicto final"
        
        # Guardar conversación
        with open(os.path.join(resultados_dir, f"implementacion_conversacion_{timestamp}.json"), "w", encoding="utf-8") as f:
            json.dump([m.to_dict() for m in self.mensajes], f, indent=2, ensure_ascii=False)
        
        # Guardar código final
        if self.codigo_final:
            with open(os.path.join(estrategias_dir, f"estrategia_final_{timestamp}.py"), "w", encoding="utf-8") as f:
                f.write(self.codigo_final)
        
        # Guardar veredicto final
        with open(os.path.join(resultados_dir, f"veredicto_final_{timestamp}.txt"), "w", encoding="utf-8") as f:
            f.write(self.veredicto_final)
        
        print(f"Resultados guardados en la carpeta 'artifacts' con timestamp {timestamp}")


# Ejemplo de uso
if __name__ == "__main__":
    # Leer la idea generada en la parte 1
    try:
        idea_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts", "idea.txt")
        with open(idea_path, "r", encoding="utf-8") as f:
            idea = f.read()
    except FileNotFoundError:
        idea = """
        Estrategia de mean-reversion en S&P 500 basada en desviaciones estadísticas extremas.
        La estrategia identifica acciones que se han desviado significativamente de su valor intrínseco
        mediante z-scores calculados sobre ratios fundamentales y técnicos, entrando en posiciones
        cuando la desviación es extrema y saliendo cuando retorna a valores normales.
        """
    
    print(f"Iniciando implementación de estrategia: {idea[:100]}...")
    
    implementacion = ImplementacionEstrategia(idea=idea, max_iteraciones=3)
    mensajes, codigo_final, resultados_finales, veredicto = implementacion.iniciar()
    
    print("\n--- VEREDICTO FINAL ---")
    print(veredicto or "No se llegó a un veredicto final")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("\nLos detalles completos se han guardado en:")
    print(f"- artifacts/implementacion_conversacion_{timestamp}.json (toda la conversación)")
    if codigo_final:
        print(f"- artifacts/estrategias/estrategia_final_{timestamp}.py (el código final)")
    print(f"- artifacts/resultados/veredicto_final_{timestamp}.txt (el veredicto final)")
    print("- artifacts/resultados_ejecucion.json (resultados de la última ejecución)")
    print("- artifacts/imagenes/ (visualizaciones generadas)")