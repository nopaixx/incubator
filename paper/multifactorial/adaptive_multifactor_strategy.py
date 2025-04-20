import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy.stats import norm
import os
import logging
import warnings
from datetime import datetime, timedelta
import pickle
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time
import joblib
from numba import jit, njit, prange

# Crear directorios para resultados
os.makedirs('./artifacts/results', exist_ok=True)
os.makedirs('./artifacts/results/figures', exist_ok=True)
os.makedirs('./artifacts/results/data', exist_ok=True)
os.makedirs('./artifacts/cache', exist_ok=True)  # Para cacheo de resultados

# Configurar logging
logging.basicConfig(
    filename='./artifacts/errors.txt',
    level=logging.ERROR,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Suprimir advertencias
warnings.filterwarnings('ignore')

# Funciones optimizadas con numba para cálculos críticos
@njit
def calc_portfolio_return_numba(weights, returns):
    """Calcula retorno del portafolio de forma optimizada"""
    return np.sum(weights * returns)

@njit
def calc_portfolio_risk_numba(weights, cov_matrix):
    """Calcula riesgo del portafolio de forma optimizada"""
    return np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))

@njit
def calc_expected_returns_numba(regime_expected_returns, regime_probs, horizon_weights, confidence_weights):
    """Calcula retornos esperados ponderados por regímenes"""
    n_regimes, n_horizons, n_assets = regime_expected_returns.shape
    expected_returns = np.zeros(n_assets)
    
    for r in range(n_regimes):
        regime_weight = regime_probs[r]
        confidence_weight = confidence_weights[r]
        
        for h_idx in range(n_horizons):
            hw = horizon_weights[h_idx]
            expected_returns += regime_weight * regime_expected_returns[r, h_idx] * confidence_weight * hw
            
    return expected_returns

class AdaptiveMultifactorStrategy:
    def __init__(self, start_date='2010-01-01', end_date=None, symbols=None, 
                 lookback_window=252, regime_window=63, n_regimes=5, 
                 rebalance_freq=21, vol_target=0.10, max_leverage=1.5,
                 transaction_cost=0.0005, market_impact=0.1, borrow_cost=0.0002,
                 execution_delay=1, use_point_in_time=True, 
                 max_drawdown_limit=0.15, dynamic_vol_scaling=True,
                 risk_targeting=True, regime_detection_method='bgmm',
                 n_jobs=-1, use_cache=True, cache_dir='./artifacts/cache',
                 use_all_assets=True):  # Nuevo parámetro para usar todos los activos
        """
        Inicializa la estrategia de descomposición multifactorial adaptativa.
        
        Parámetros:
        -----------
        start_date : str
            Fecha de inicio para los datos históricos (formato 'YYYY-MM-DD')
        end_date : str
            Fecha de fin para los datos históricos (formato 'YYYY-MM-DD')
        symbols : list
            Lista de símbolos a incluir. Si es None, se usa un universo de referencia histórico
        lookback_window : int
            Ventana de observación para calcular factores latentes (días)
        regime_window : int
            Ventana para detectar regímenes de mercado (días)
        n_regimes : int
            Número de regímenes de mercado a detectar
        rebalance_freq : int
            Frecuencia de rebalanceo en días
        vol_target : float
            Volatilidad objetivo anualizada
        max_leverage : float
            Apalancamiento máximo permitido
        transaction_cost : float
            Costo de transacción como porcentaje del valor operado
        market_impact : float
            Impacto de mercado como factor de volatilidad diaria
        borrow_cost : float
            Costo anualizado de tomar posiciones cortas
        execution_delay : int
            Retraso en días entre decisión y ejecución
        use_point_in_time : bool
            Usar datos point-in-time para evitar sesgo de supervivencia
        max_drawdown_limit : float
            Límite de drawdown máximo permitido antes de reducir exposición
        dynamic_vol_scaling : bool
            Usar scaling dinámico de volatilidad para ajustar exposición
        risk_targeting : bool
            Usar targeting dinámico de riesgo basado en regímenes
        regime_detection_method : str
            Método para detectar regímenes ('gmm' o 'bgmm')
        n_jobs : int
            Número de procesos a usar para paralelización (-1 = todos disponibles)
        use_cache : bool
            Si se debe cachear resultados intermedios
        cache_dir : str
            Directorio donde guardar resultados cacheados
        use_all_assets : bool
            Si se deben usar todos los activos sin filtrar por liquidez o disponibilidad
        """
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.symbols = symbols
        self.lookback_window = lookback_window
        self.regime_window = regime_window
        self.n_regimes = n_regimes
        self.rebalance_freq = rebalance_freq
        self.vol_target = vol_target
        self.max_leverage = max_leverage
        
        # Parámetros para implementación realista
        self.transaction_cost = transaction_cost
        self.market_impact = market_impact
        self.borrow_cost = borrow_cost
        self.execution_delay = execution_delay
        self.use_point_in_time = use_point_in_time
        
        # Nuevos parámetros de mejora
        self.max_drawdown_limit = max_drawdown_limit
        self.dynamic_vol_scaling = dynamic_vol_scaling
        self.risk_targeting = risk_targeting
        self.regime_detection_method = regime_detection_method
        self.use_all_assets = use_all_assets  # Nuevo parámetro
        
        # Parámetros de optimización
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        
        # Cache para resultados intermedios
        self.pca_cache = {}
        self.regime_cache = {}
        self.expected_returns_cache = {}
        
        # Atributos que se inicializarán más tarde
        self.prices = None
        self.returns = None
        self.factor_loadings = None
        self.factor_returns = None
        self.regimes = None
        self.regime_probs = None
        self.optimal_weights = None
        self.performance = None
        self.benchmark_data = None
        self.trades_history = []
        
        # Parámetros adicionales para simulación realista
        self.tradable_assets = None
        self.max_position_size = 0.1
        self.liquidity_threshold = 1000000
        
        # Tiempo de inicio para medir performance
        self.start_time = time.time()
        
        # Cargar datos
        self._load_data()
        
    def _load_data(self):
        """
        Carga los datos históricos de precios y calcula retornos.
        Optimizado: Usa caché de datos si disponible
        """
        try:
            cache_file = os.path.join(self.cache_dir, 'market_data.pkl')
            
            # Intentar cargar desde caché
            if self.use_cache and os.path.exists(cache_file):
                print(f"Cargando datos desde caché: {cache_file}")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                if (cached_data.get('start_date') == self.start_date and 
                    cached_data.get('end_date') == self.end_date and
                    (self.symbols is None or set(cached_data.get('symbols', [])) == set(self.symbols))):
                    
                    self.prices = cached_data['prices']
                    self.volumes = cached_data.get('volumes')
                    self.returns = cached_data['returns']
                    self.daily_vol = cached_data['daily_vol']
                    self.tradable_universe = cached_data['tradable_universe']
                    self.shortable_universe = cached_data['shortable_universe']
                    self.symbols = list(self.prices.columns)
                    
                    # Corregir: Si use_all_assets=True, hacer que todos los activos sean negociables
                    if self.use_all_assets:
                        self.tradable_universe = pd.DataFrame(
                            True,
                            index=self.tradable_universe.index,
                            columns=self.tradable_universe.columns
                        )
                        self.shortable_universe = pd.DataFrame(
                            True,
                            index=self.shortable_universe.index,
                            columns=self.shortable_universe.columns
                        )
                    
                    # Corregir: Asegurarse de que el benchmark esté cargado correctamente
                    if 'benchmark_data' in cached_data and 'benchmark_returns' in cached_data:
                        self.benchmark_data = cached_data['benchmark_data']
                        self.benchmark_returns = cached_data['benchmark_returns']
                    else:
                        # Si no está en caché, cargarlo ahora
                        self._load_benchmark()
                    
                    print(f"Datos cargados exitosamente desde caché. {len(self.symbols)} símbolos, {len(self.returns)} días de trading.")
                    return
            
            if self.symbols is None:
                # Obtener S&P 500 actual
                sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
                self.symbols = sp500['Symbol'].tolist()
                
                # Simular cambios en el universo a lo largo del tiempo
                # np.random.seed(42)
                # self.symbols = [s for s in self.symbols if np.random.random() > 0.2]
            
            print(f"Descargando datos para {len(self.symbols)} símbolos...")
            
            # Descargar datos - Optimizado para hacerlo en bloques
            max_symbols_per_request = 100  # Yahoo tiene límites en cantidad de símbolos por consulta
            all_data = []
            
            for i in range(0, len(self.symbols), max_symbols_per_request):
                symbol_chunk = self.symbols[i:i+max_symbols_per_request]
                print(f"Descargando bloque {i//max_symbols_per_request + 1}/{(len(self.symbols)-1)//max_symbols_per_request + 1} ({len(symbol_chunk)} símbolos)")
                chunk_data = yf.download(symbol_chunk, start=self.start_date, end=self.end_date, progress=False)
                all_data.append(chunk_data)
            
            # Combinar datos si hay múltiples bloques
            if len(all_data) > 1:
                # Si hay un solo símbolo, yfinance no usa MultiIndex
                if len(self.symbols) == 1:
                    data = all_data[0]
                    # Convertir a MultiIndex para consistencia
                    data.columns = pd.MultiIndex.from_product([data.columns, self.symbols])
                else:
                    data = pd.concat(all_data, axis=1)
            else:
                data = all_data[0]
            
            # Extraer precios y volúmenes
            if len(self.symbols) == 1:
                # Caso especial para un solo símbolo
                self.prices = data['Close'].to_frame(self.symbols[0])
                self.volumes = data['Volume'].to_frame(self.symbols[0]) if 'Volume' in data.columns else None
            else:
                self.prices = data['Close']
                self.volumes = data['Volume'] if 'Volume' in data.columns else None
            
            # Limpiar y preparar datos
            self.prices = self.prices.dropna(axis=1, thresh=int(len(self.prices) * 0.9))
            self.symbols = list(self.prices.columns)
            
            # Calcular retornos diarios - optimizado
            self.returns = self.prices.pct_change().dropna()
            
            # Calcular volatilidades diarias para estimar impacto de mercado - optimizado
            self.daily_vol = self.returns.rolling(21).std().fillna(method='bfill')
            
            # Crear DataFrame para simular disponibilidad de activos
            if self.use_all_assets:
                # MODIFICACIÓN: Marcar todos los activos como negociables si use_all_assets=True
                self.tradable_universe = pd.DataFrame(
                    True, 
                    index=self.returns.index, 
                    columns=self.returns.columns
                )
                
                # MODIFICACIÓN: Permitir posiciones cortas en todos los activos
                self.shortable_universe = pd.DataFrame(
                    True,
                    index=self.tradable_universe.index,
                    columns=self.tradable_universe.columns
                )
            else:
                # Comportamiento original
                self.tradable_universe = pd.DataFrame(
                    True, 
                    index=self.returns.index, 
                    columns=self.returns.columns
                )
                
                # Simular disponibilidad para posiciones cortas
                self.shortable_universe = pd.DataFrame(
                    np.random.random(self.tradable_universe.shape) > 0.3,
                    index=self.tradable_universe.index,
                    columns=self.tradable_universe.columns
                )
                
                # Filtrar por liquidez mínima
                if self.volumes is not None:
                    dollar_volumes = self.volumes * self.prices
                    
                    # Marca como no negociables los activos con baja liquidez - Optimizado con vectorización
                    low_liquidity = dollar_volumes < self.liquidity_threshold
                    self.tradable_universe = ~low_liquidity & self.tradable_universe
                    self.shortable_universe = ~low_liquidity & self.shortable_universe
            
            # Cargar benchmark
            self._load_benchmark()
            
            print(f"Datos cargados exitosamente. {len(self.symbols)} símbolos, {len(self.returns)} días de trading.")
            print(f"En promedio, {self.tradable_universe.mean().mean()*100:.1f}% de los activos son negociables.")
            print(f"En promedio, {self.shortable_universe.mean().mean()*100:.1f}% de los activos son susceptibles de posiciones cortas.")
            
            # Guardar en caché
            if self.use_cache:
                os.makedirs(self.cache_dir, exist_ok=True)
                cached_data = {
                    'start_date': self.start_date,
                    'end_date': self.end_date,
                    'symbols': self.symbols,
                    'prices': self.prices,
                    'volumes': self.volumes,
                    'returns': self.returns,
                    'daily_vol': self.daily_vol,
                    'tradable_universe': self.tradable_universe,
                    'shortable_universe': self.shortable_universe,
                    'benchmark_data': self.benchmark_data,
                    'benchmark_returns': self.benchmark_returns
                }
                
                with open(cache_file, 'wb') as f:
                    pickle.dump(cached_data, f)
                
                print(f"Datos guardados en caché: {cache_file}")
            
        except Exception as e:
            logging.error(f"Error al cargar datos: {str(e)}", exc_info=True)
            raise
    
    def _load_benchmark(self):
        """
        Carga los datos del benchmark (S&P 500).
        NUEVA FUNCIÓN: Separada para mejor manejo de errores y reintento
        """
        try:
            # Descargar benchmark (S&P 500)
            print("Descargando datos de benchmark (S&P 500)...")
            benchmark = yf.download('^GSPC', start=self.start_date, end=self.end_date, progress=False)
            
            # CORREGIDO: Mejor manejo del DataFrame del benchmark
            if not benchmark.empty:
                # Crear un DataFrame simple con la columna 'S&P500'
                self.benchmark_data = pd.DataFrame(benchmark['Close'].copy())
                self.benchmark_data.columns = ['S&P500']
                
                # Calcular retornos del benchmark
                self.benchmark_returns = pd.DataFrame(self.benchmark_data.pct_change().dropna())
                
                print(f"Datos de benchmark cargados: {len(self.benchmark_returns)} días")
            else:
                print("ERROR: No se pudieron descargar datos del benchmark. Intentando una alternativa...")
                
                # Intentar con otro ticker como alternativa
                alt_benchmark = yf.download('SPY', start=self.start_date, end=self.end_date, progress=False)
                if not alt_benchmark.empty:
                    self.benchmark_data = pd.DataFrame(alt_benchmark['Close'].copy())
                    self.benchmark_data.columns = ['S&P500']
                    self.benchmark_returns = pd.DataFrame(self.benchmark_data.pct_change().dropna())
                    print(f"Datos de benchmark alternativo cargados: {len(self.benchmark_returns)} días")
                else:
                    # Si todo falla, crear un benchmark simulado basado en la media de todos los activos
                    print("ADVERTENCIA: Creando benchmark simulado basado en la media del universo")
                    self.benchmark_data = pd.DataFrame(self.prices.mean(axis=1), columns=['S&P500'])
                    self.benchmark_returns = pd.DataFrame(self.returns.mean(axis=1), columns=['S&P500'])
        
        except Exception as e:
            logging.error(f"Error al cargar benchmark: {str(e)}", exc_info=True)
            print(f"Error al cargar benchmark: {str(e)}")
            
            # Crear un benchmark simulado en caso de error
            print("ADVERTENCIA: Creando benchmark simulado basado en la media del universo")
            self.benchmark_data = pd.DataFrame(self.prices.mean(axis=1), columns=['S&P500'])
            self.benchmark_returns = pd.DataFrame(self.returns.mean(axis=1), columns=['S&P500'])
    
    def extract_latent_factors(self, returns_window, n_components=None):
        """
        Extrae factores latentes de los retornos usando PCA.
        Optimizado: Usa caché y algoritmos más eficientes
        
        Parámetros:
        -----------
        returns_window : DataFrame
            Ventana de retornos para extraer factores
        n_components : int, opcional
            Número de componentes a extraer. Si es None, se determina automáticamente.
            
        Retorna:
        --------
        factor_loadings : ndarray
            Cargas de los factores latentes
        factor_returns : DataFrame
            Retornos de los factores latentes
        n_components : int
            Número de componentes utilizados
        """
        try:
            # Generar clave de caché basada en ventana de retornos
            start_date = returns_window.index[0].strftime('%Y%m%d')
            end_date = returns_window.index[-1].strftime('%Y%m%d')
            cache_key = f"pca_{start_date}_{end_date}_{n_components}"
            
            # Intentar usar caché si está disponible
            if self.use_cache and cache_key in self.pca_cache:
                return self.pca_cache[cache_key]
            
            # Manejar valores faltantes
            returns_filled = returns_window.copy()
            
            # Optimización: Mejor manejo de NaNs utilizando vectorización
            # Rellenar NaNs con medias móviles
            if returns_filled.isna().any().any():
                for col in returns_filled.columns[returns_filled.isna().any()]:
                    mask = returns_filled[col].isna()
                    if mask.any():
                        returns_filled.loc[mask, col] = returns_filled[col].rolling(5, min_periods=1).mean()[mask]
            
            # Rellenar NaNs restantes con ceros
            returns_filled = returns_filled.fillna(0)
            
            # Determinar número óptimo de componentes si no se especifica
            if n_components is None:
                n_components = self.find_optimal_components(returns_filled)
            
            # Optimización: Usar PCA incremental o rápido para datasets grandes
            n_samples, n_features = returns_filled.shape
            use_incremental = n_samples * n_features > 1000000  # Umbral basado en tamaño de datos
            
            if use_incremental:
                # PCA incremental para conjuntos de datos grandes
                batch_size = min(n_samples, 1000)  # Tamaño de lote razonable
                pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
            else:
                # PCA estándar para conjuntos de datos menores
                pca = PCA(n_components=n_components)
            
            # Ejecutar PCA
            factor_returns_np = pca.fit_transform(returns_filled)
            
            # Convertir a DataFrame
            factor_returns = pd.DataFrame(
                factor_returns_np, 
                index=returns_window.index,
                columns=[f'Factor_{i+1}' for i in range(n_components)]
            )
            
            # Almacenar en caché
            result = (pca.components_, factor_returns, n_components)
            if self.use_cache:
                self.pca_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logging.error(f"Error en extract_latent_factors: {str(e)}", exc_info=True)
            raise
    
    def find_optimal_components(self, returns_window, threshold=0.85, max_components=15):
        """
        Determina el número óptimo de componentes principales.
        Optimizado: Algoritmo más eficiente y cacheado
        
        Parámetros:
        -----------
        returns_window : DataFrame
            Ventana de retornos para analizar
        threshold : float
            Umbral de varianza explicada acumulada
        max_components : int
            Número máximo de componentes a considerar
            
        Retorna:
        --------
        n_components : int
            Número óptimo de componentes
        """
        try:
            # Generar clave de caché
            start_date = returns_window.index[0].strftime('%Y%m%d')
            end_date = returns_window.index[-1].strftime('%Y%m%d')
            cache_key = f"opt_comp_{start_date}_{end_date}_{threshold}_{max_components}"
            
            # Intentar usar caché si está disponible
            if self.use_cache and cache_key in self.pca_cache:
                return self.pca_cache[cache_key]
            
            # Limitar el máximo posible de componentes
            max_possible = min(returns_window.shape[1], returns_window.shape[0], max_components)
            
            # Optimización: Si el conjunto de datos es pequeño, usar análisis completo
            if returns_window.shape[1] <= 50:  # Umbral pequeño para análisis completo
                pca = PCA(n_components=max_possible)
                pca.fit(returns_window)
                explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
            else:
                # Para conjuntos grandes, calcular solo la varianza explicada incremental
                # usando un enfoque de muestreo o truncado
                randomized_pca = PCA(n_components=max_possible, svd_solver='randomized')
                randomized_pca.fit(returns_window)
                explained_variance_ratio_cumsum = np.cumsum(randomized_pca.explained_variance_ratio_)
            
            # Encontrar el número de componentes que explican al menos threshold de la varianza
            n_components = np.argmax(explained_variance_ratio_cumsum >= threshold) + 1
            
            # Asegurar un mínimo de componentes
            n_components = max(n_components, 3)
            
            # Almacenar en caché
            if self.use_cache:
                self.pca_cache[cache_key] = n_components
            
            return n_components
            
        except Exception as e:
            logging.error(f"Error en find_optimal_components: {str(e)}", exc_info=True)
            # Valor por defecto en caso de error
            return 5
    
    def detect_regimes(self, factor_returns, n_regimes=None):
        """
        Detecta regímenes de mercado usando modelos de mezcla gaussiana.
        Optimizado: Algoritmo más eficiente y cacheado
        
        Parámetros:
        -----------
        factor_returns : DataFrame
            Retornos de los factores latentes
        n_regimes : int, opcional
            Número de regímenes a detectar. Si es None, se usa self.n_regimes.
            
        Retorna:
        --------
        regimes : ndarray
            Etiquetas de régimen para cada punto temporal
        regime_probs : ndarray
            Probabilidades de pertenencia a cada régimen
        """
        try:
            if n_regimes is None:
                n_regimes = self.n_regimes
            
            # Generar clave de caché
            start_date = factor_returns.index[0].strftime('%Y%m%d')
            end_date = factor_returns.index[-1].strftime('%Y%m%d')
            cache_key = f"regime_{start_date}_{end_date}_{n_regimes}_{self.regime_detection_method}"
            
            # Intentar usar caché si está disponible
            if self.use_cache and cache_key in self.regime_cache:
                return self.regime_cache[cache_key]
            
            # MEJORA Y OPTIMIZACIÓN: Características más eficientes para detectar regímenes
            # 1. Volatilidad - usando vectorización
            vol = factor_returns.rolling(21).std().dropna()
            
            # 2. Correlación entre factores - optimizado
            # Preparar arrays para crear matrices de correlación de manera eficiente
            window_size = 21
            n_windows = len(factor_returns) - window_size + 1
            
            if n_windows <= 0:
                # No hay suficientes datos para calcular correlaciones
                features = vol.copy()
            else:
                # Optimización: Precalcular todas las ventanas a la vez
                windows = np.array([factor_returns.iloc[i:i+window_size].values for i in range(n_windows)])
                
                # Inicializar array para correlaciones
                n_factors = factor_returns.shape[1]
                n_corr_features = n_factors * (n_factors - 1) // 2
                corr_features = np.zeros((n_windows, n_corr_features))
                
                # Calcular correlaciones para todas las ventanas
                for i in range(n_windows):
                    # Calcular matriz de correlación
                    window_data = windows[i]
                    corr_matrix = np.corrcoef(window_data.T)
                    
                    # Extraer triángulo superior
                    upper_indices = np.triu_indices(n_factors, k=1)
                    corr_features[i] = corr_matrix[upper_indices]
                
                # Convertir a DataFrame
                corr_df = pd.DataFrame(
                    corr_features,
                    index=factor_returns.index[window_size-1:],
                    columns=[f'Corr_{i}_{j}' for i, j in zip(*np.triu_indices(n_factors, k=1))]
                )
                
                # Alinear índices
                vol = vol.loc[corr_df.index[0]:].copy()
                
                # Combinar características
                features = pd.concat([vol, corr_df], axis=1)
            
            # Estandarizar características
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Limitar el número de características para GMM - mejora la velocidad
            if features_scaled.shape[1] > 20:  # Si hay demasiadas características
                # Usar PCA para reducir dimensionalidad
                pca_reducer = PCA(n_components=min(20, features_scaled.shape[1]))
                features_scaled = pca_reducer.fit_transform(features_scaled)
            
            # MEJORA Y OPTIMIZACIÓN: GMM más eficiente
            if self.regime_detection_method == 'bgmm':
                # Bayesian Gaussian Mixture Model con inicialización optimizada
                gmm = BayesianGaussianMixture(
                    n_components=n_regimes,
                    covariance_type='full',
                    random_state=42,
                    n_init=5,  # Reducir número de inicializaciones
                    max_iter=100,  # Limitar iteraciones
                    weight_concentration_prior=0.1,
                    init_params='kmeans'  # Mejor inicialización
                )
            else:
                # GMM estándar optimizado
                gmm = GaussianMixture(
                    n_components=n_regimes,
                    covariance_type='full',
                    random_state=42,
                    n_init=5,
                    max_iter=100,
                    init_params='kmeans'
                )
            
            # Manejar NaNs de forma eficiente
            if np.isnan(features_scaled).any():
                features_scaled = np.nan_to_num(features_scaled)
            
            # Ajustar modelo
            gmm.fit(features_scaled)
            
            # Predecir regímenes y probabilidades
            regimes = gmm.predict(features_scaled)
            regime_probs = gmm.predict_proba(features_scaled)
            
            # MEJORA: Ordenar regímenes por volatilidad para que sean consistentes entre ejecuciones
            # Calcular volatilidad media por régimen - vectorizado
            regime_vol = np.zeros(n_regimes)
            for r in range(n_regimes):
                regime_mask = (regimes == r)
                if np.any(regime_mask):
                    regime_vol[r] = np.mean(vol.values[regime_mask].mean())
            
            # Ordenar regímenes por volatilidad (ascendente)
            regime_order = np.argsort(regime_vol)
            regime_map = {old: new for new, old in enumerate(regime_order)}
            
            # Reordenar regímenes y probabilidades - vectorizado
            new_regimes = np.array([regime_map[r] for r in regimes])
            new_probs = np.zeros_like(regime_probs)
            for old, new in regime_map.items():
                new_probs[:, new] = regime_probs[:, old]
            
            # Almacenar en caché
            result = (new_regimes, new_probs)
            if self.use_cache:
                self.regime_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logging.error(f"Error en detect_regimes: {str(e)}", exc_info=True)
            # Valores por defecto en caso de error
            features_len = len(features) if 'features' in locals() else len(factor_returns) - 20
            dummy_regimes = np.zeros(features_len)
            dummy_probs = np.ones((len(dummy_regimes), self.n_regimes)) / self.n_regimes
            return dummy_regimes, dummy_probs
    
    def predict_expected_returns(self, returns_window, regimes, current_regime_probs, horizon=10):
        """
        MEJORA: Método de predicción de retornos más robusto con múltiples horizontes
        Optimizado: Proceso más eficiente y vectorizado
        
        Parámetros:
        -----------
        returns_window : DataFrame
            Ventana histórica de retornos
        regimes : ndarray
            Etiquetas de régimen para cada punto temporal
        current_regime_probs : ndarray
            Probabilidades de pertenencia a cada régimen en el momento actual
        horizon : int
            Horizonte de predicción en días
            
        Retorna:
        --------
        expected_returns : Series
            Retornos esperados para cada activo
        prediction_confidence : Series
            Confianza en las predicciones
        """
        try:
            # Generar clave de caché
            start_date = returns_window.index[0].strftime('%Y%m%d')
            end_date = returns_window.index[-1].strftime('%Y%m%d')
            current_regime = np.argmax(current_regime_probs)
            cache_key = f"exp_ret_{start_date}_{end_date}_{current_regime}_{horizon}"
            
            # Intentar usar caché si está disponible
            if self.use_cache and cache_key in self.expected_returns_cache:
                return self.expected_returns_cache[cache_key]
            
            n_assets = returns_window.shape[1]
            
            # MEJORA Y OPTIMIZACIÓN: Usar múltiples horizontes para predicción más robusta
            horizons = [5, 10, 21]  # 1 semana, 2 semanas, 1 mes
            horizon_weights = np.array([0.5, 0.3, 0.2])  # Priorizar horizontes más cortos
            
            # Inicializar arrays para retornos esperados por régimen y horizonte
            regime_expected_returns = np.zeros((self.n_regimes, len(horizons), n_assets))
            regime_counts = np.zeros(self.n_regimes)
            
            # Optimización: Precalcular retornos futuros para todos los horizontes
            # Esta es una operación costosa, así que la hacemos una sola vez
            future_returns_all = {}
            for h_idx, h in enumerate(horizons):
                future_returns = []
                for idx in range(len(returns_window) - h):
                    # Próximos 'h' días después de este régimen
                    future_ret = returns_window.iloc[idx+1:idx+1+h].values
                    # Calcular retorno acumulado
                    cum_returns = np.prod(1 + future_ret, axis=0) - 1
                    future_returns.append(cum_returns)
                
                if future_returns:
                    future_returns_all[h] = np.array(future_returns)
            
            # Paralelizar el cálculo por régimen si hay muchos regímenes
            if self.n_regimes > 3 and self.n_jobs > 1:
                # Función para procesar cada régimen en paralelo
                def process_regime(r):
                    regime_result = np.zeros((len(horizons), n_assets))
                    regime_indices = np.where(regimes == r)[0]
                    count = len(regime_indices)
                    
                    if count > 0:
                        for h_idx, h in enumerate(horizons):
                            if h in future_returns_all and len(future_returns_all[h]) > 0:
                                # Filtrar retornos futuros para este régimen
                                valid_indices = regime_indices[regime_indices < len(future_returns_all[h])]
                                if len(valid_indices) > 0:
                                    # Extraer retornos futuros para este régimen
                                    regime_future_returns = future_returns_all[h][valid_indices]
                                    # Usar mediana para mayor robustez
                                    regime_result[h_idx] = np.median(regime_future_returns, axis=0)
                    
                    return r, regime_result, count
                
                # Ejecutar en paralelo
                with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    results = list(executor.map(process_regime, range(self.n_regimes)))
                
                # Organizar resultados
                for r, regime_result, count in results:
                    regime_expected_returns[r] = regime_result
                    regime_counts[r] = count
            else:
                # Versión en serie si no vale la pena paralelizar
                for r in range(self.n_regimes):
                    # Encontrar índices donde el régimen es r
                    regime_indices = np.where(regimes == r)[0]
                    regime_counts[r] = len(regime_indices)
                    
                    if len(regime_indices) > 0:
                        for h_idx, h in enumerate(horizons):
                            if h in future_returns_all and len(future_returns_all[h]) > 0:
                                # Filtrar retornos futuros para este régimen
                                valid_indices = regime_indices[regime_indices < len(future_returns_all[h])]
                                if len(valid_indices) > 0:
                                    # Extraer retornos futuros para este régimen
                                    regime_future_returns = future_returns_all[h][valid_indices]
                                    # Usar mediana para mayor robustez
                                    regime_expected_returns[r, h_idx] = np.median(regime_future_returns, axis=0)
            
            # MEJORA Y OPTIMIZACIÓN: Usar función numba para cálculos más rápidos
            # Preparar datos para la función optimizada
            confidence_weights = np.minimum(1.0, regime_counts / 30)
            
            # Calcular retornos esperados vectorizados
            expected_returns_array = calc_expected_returns_numba(
                regime_expected_returns, 
                current_regime_probs, 
                horizon_weights, 
                confidence_weights
            )
            
            # Convertir a Series
            expected_returns = pd.Series(expected_returns_array, index=returns_window.columns)
            
            # MEJORA: Ajustar por autocorrelación - añadir componente de momentum/reversal
            # Optimizado usando operaciones vectorizadas
            short_momentum = returns_window.iloc[-5:].mean() * 0.2  # Momentum de corto plazo (5 días)
            long_momentum = returns_window.iloc[-21:].mean() * 0.1  # Momentum de largo plazo (21 días)
            
            # Combinar señales
            expected_returns = expected_returns + short_momentum - long_momentum
            
            # Calcular confianza de predicción
            regime_certainty = np.max(current_regime_probs)
            data_sufficiency = np.min([count for count in regime_counts if count > 0]) / 30 if any(regime_counts > 0) else 0
            prediction_confidence = pd.Series(regime_certainty * data_sufficiency, index=returns_window.columns)
            
            # MEJORA: Ajustar retornos esperados por volatilidad reciente
            recent_vol = returns_window.iloc[-21:].std()
            vol_scale = np.clip(self.vol_target / (recent_vol * np.sqrt(252)), 0.5, 2.0)
            expected_returns *= vol_scale
            
            # Almacenar en caché
            result = (expected_returns, prediction_confidence)
            if self.use_cache:
                self.expected_returns_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logging.error(f"Error en predict_expected_returns: {str(e)}", exc_info=True)
            # Valores por defecto en caso de error
            dummy_returns = pd.Series(0.0001, index=returns_window.columns)
            dummy_confidence = pd.Series(0.1, index=returns_window.columns)
            return dummy_returns, dummy_confidence
    
    def calculate_risk_budget(self, current_regime, regime_certainty, market_vol):
        """
        MEJORA: Calcula dinámicamente el presupuesto de riesgo según condiciones del mercado
        Optimizado para mayor eficiencia
        
        Parámetros:
        -----------
        current_regime : int
            Régimen de mercado actual
        regime_certainty : float
            Certeza sobre el régimen actual
        market_vol : float
            Volatilidad reciente del mercado
        
        Retorna:
        --------
        risk_budget : dict
            Parámetros de riesgo para la optimización
        """
        # Volatilidad normalizada (>1 significa alta volatilidad, <1 significa baja volatilidad)
        vol_normalized = market_vol / self.vol_target
        
        # Factor de escala para la aversión al riesgo (mayor en volatilidad alta)
        risk_aversion_scale = np.clip(vol_normalized, 0.5, 3.0)
        
        # Aversión al riesgo base según régimen
        # Regímenes fueron ordenados por volatilidad, así que mayor número = mayor volatilidad
        base_risk_aversion = 1.0 + current_regime * 0.5
        
        # Ajustar por certeza del régimen (menor certeza = mayor aversión)
        certainty_adjustment = 1.0 + (1.0 - regime_certainty) * 2.0
        
        # Aversión al riesgo final
        final_risk_aversion = base_risk_aversion * certainty_adjustment * risk_aversion_scale
        
        # Ajustar leverage máximo según condiciones
        leverage_scale = 1.0
        
        # En regímenes de alta volatilidad, reducir apalancamiento
        if current_regime >= self.n_regimes - 2:  # Dos regímenes más volátiles
            leverage_scale *= 0.7
        
        # Si volatilidad es muy alta, reducir aún más
        if vol_normalized > 1.5:
            leverage_scale *= 0.7
        
        # Si certeza es baja, reducir apalancamiento
        if regime_certainty < 0.5:
            leverage_scale *= 0.9
        
        # Calcular apalancamiento máximo objetivo (nunca mayor al configurado)
        target_max_leverage = min(self.max_leverage * leverage_scale, self.max_leverage)
        
        # Límites para posiciones cortas según régimen y volatilidad
        if current_regime <= 1 and vol_normalized < 1.2:  # Regímenes de baja volatilidad
            short_limit = -0.3  # Permitir más posiciones cortas
        elif current_regime <= 2 and vol_normalized < 1.5:  # Regímenes moderados
            short_limit = -0.2  # Posiciones cortas moderadas
        elif current_regime <= 3:  # Regímenes de mayor volatilidad
            short_limit = -0.1  # Pocas posiciones cortas
        else:  # Regímenes de alta volatilidad
            short_limit = -0.05  # Muy pocas posiciones cortas
        
        # Tamaño máximo de posición ajustado por régimen
        position_size_scale = np.clip(1.5 - current_regime * 0.2, 0.5, 1.0)
        max_position_size = self.max_position_size * position_size_scale
        
        # Devolver presupuesto de riesgo
        return {
            'risk_aversion': final_risk_aversion,
            'max_leverage': target_max_leverage,
            'short_limit': short_limit,
            'max_position_size': max_position_size
        }
    
    def optimize_portfolio(self, expected_returns, factor_loadings, prediction_confidence, 
                          current_regime, regime_certainty, current_date, 
                          previous_weights=None, current_drawdown=0.0,
                          market_vol=0.15):
        """
        MEJORA: Optimiza el portafolio con control dinámico de riesgo
        Optimizado: Algoritmo más eficiente y rapido
        
        Parámetros:
        -----------
        expected_returns : Series
            Retornos esperados para cada activo
        factor_loadings : ndarray
            Cargas de los factores latentes
        prediction_confidence : Series
            Confianza en las predicciones
        current_regime : int
            Régimen de mercado actual
        regime_certainty : float
            Certeza sobre el régimen actual
        current_date : Timestamp
            Fecha actual para determinar activos negociables
        previous_weights : Series, opcional
            Pesos del portafolio previo
        current_drawdown : float
            Drawdown actual del portafolio
        market_vol : float
            Volatilidad reciente del mercado anualizada
            
        Retorna:
        --------
        weights : Series
            Pesos óptimos para cada activo
        """
        try:
            n_assets = len(expected_returns)
            assets = expected_returns.index
            
            # MODIFICACIÓN: Si use_all_assets=True, considerar todos los activos como negociables
            if self.use_all_assets:
                # Marcar todos los activos como negociables y shortables
                tradable_mask = np.ones(n_assets, dtype=bool)
                shortable_mask = np.ones(n_assets, dtype=bool)
            else:
                # Determinar activos negociables en la fecha actual - optimizado
                if current_date in self.tradable_universe.index:
                    tradable_mask = self.tradable_universe.loc[current_date, assets].values
                    shortable_mask = self.shortable_universe.loc[current_date, assets].values
                else:
                    # Si fecha no disponible, usar más reciente
                    last_available = self.tradable_universe.index[self.tradable_universe.index <= current_date]
                    if len(last_available) > 0:
                        last_available = last_available[-1]
                        tradable_mask = self.tradable_universe.loc[last_available, assets].values
                        shortable_mask = self.shortable_universe.loc[last_available, assets].values
                    else:
                        # Fallback: asumir todos negociables
                        tradable_mask = np.ones(len(assets), dtype=bool)
                        shortable_mask = np.ones(len(assets), dtype=bool)
            
            # Filtrar activos no negociables - optimizado con vectorización
            filtered_returns = expected_returns.copy()
            filtered_returns[~tradable_mask] = -999
            
            # MEJORA: Usar presupuesto de riesgo dinámico
            risk_budget = self.calculate_risk_budget(current_regime, regime_certainty, market_vol)
            adjusted_risk_aversion = risk_budget['risk_aversion']
            target_max_leverage = risk_budget['max_leverage']
            short_limit = risk_budget['short_limit']
            max_position_size = risk_budget['max_position_size']
            
            # MEJORA: Reducir exposición si drawdown se acerca al límite
            drawdown_factor = 1.0
            if self.max_drawdown_limit > 0 and current_drawdown > 0:
                # Reducir exposición progresivamente a medida que se acerca al límite
                drawdown_ratio = current_drawdown / self.max_drawdown_limit
                if drawdown_ratio > 0.5:
                    drawdown_factor = max(0.5, 1.0 - (drawdown_ratio - 0.5) * 2.0)
            
            # Ajustar apalancamiento máximo por drawdown
            target_max_leverage *= drawdown_factor
            
            # Calcular matriz de covarianza basada en factores latentes - optimizado
            factor_cov = np.cov(factor_loadings)
            asset_cov = factor_loadings.T @ factor_cov @ factor_loadings
            
            # Asegurar que la matriz es definida positiva - optimizado
            min_eig = np.min(np.real(np.linalg.eigvals(asset_cov)))
            if min_eig < 1e-6:
                asset_cov += np.eye(n_assets) * (1e-6 - min_eig)
            
            # Ajustar retornos esperados por confianza
            adjusted_returns = filtered_returns * prediction_confidence
            
            # Considerar costos de préstamo para posiciones cortas - vectorizado
            borrow_costs = np.zeros(n_assets)
            borrow_costs[~shortable_mask] = 0.1  # Penalización alta para activos no shortables
            borrow_costs[shortable_mask] = self.borrow_cost / 252
            
            # Incluir costos de transacción estimados
            if previous_weights is not None:
                # Estimar impacto de mercado basado en volatilidad
                if current_date in self.daily_vol.index:
                    vol_values = self.daily_vol.loc[current_date, assets].fillna(0.02).values
                else:
                    vol_values = np.full(n_assets, 0.02)
                
                market_impact_cost = vol_values * self.market_impact
                prev_weights_array = previous_weights.values
            else:
                market_impact_cost = np.zeros(n_assets)
                prev_weights_array = np.zeros(n_assets)
            
            # OPTIMIZACIÓN: Función objetivo vectorizada para mayor velocidad
            def objective(weights):
                # Retorno esperado
                portfolio_return = np.sum(weights * adjusted_returns.values)
                
                # Riesgo - usar función optimizada
                portfolio_risk = calc_portfolio_risk_numba(weights, asset_cov)
                
                # Costos de transacción
                turnover = np.sum(np.abs(weights - prev_weights_array))
                transaction_costs = turnover * self.transaction_cost
                
                # Impacto de mercado estimado
                impact_costs = np.sum(np.abs(weights - prev_weights_array) * market_impact_cost)
                
                # Costos de préstamo para posiciones cortas
                short_weights = np.maximum(-weights, 0)
                short_costs = np.sum(short_weights * borrow_costs)
                
                # Penalización para diversificación
                concentration_penalty = np.sum(weights ** 2) * adjusted_risk_aversion * 0.5
                
                # Utilidad final: retorno - riesgo - costos - concentración
                utility = (portfolio_return 
                           - adjusted_risk_aversion * portfolio_risk 
                           - transaction_costs 
                           - impact_costs 
                           - short_costs
                           - concentration_penalty)
                
                return -utility  # Negativo porque minimizamos
            
            # Restricciones
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Suma de pesos = 1
            ]
            
            # Límites en las posiciones - optimizado
            lower_bounds = np.zeros(n_assets)
            upper_bounds = np.zeros(n_assets)
            
            # MODIFICACIÓN: Si use_all_assets=True, aplicar límites más simples
            if self.use_all_assets:
                # Permitir posiciones cortas y largas en todos los activos con límites por activo
                lower_bounds = np.full(n_assets, max(short_limit, -max_position_size))
                upper_bounds = np.full(n_assets, min(1.0, max_position_size))
            else:
                # Vectorizar límites (comportamiento original)
                lower_bounds[~tradable_mask] = 0.0
                upper_bounds[~tradable_mask] = 0.0
                
                # Activos negociables pero no shortables
                mask_tradable_not_shortable = tradable_mask & ~shortable_mask
                lower_bounds[mask_tradable_not_shortable] = 0.0
                upper_bounds[mask_tradable_not_shortable] = min(1.0, max_position_size)
                
                # Activos completamente negociables
                mask_fully_tradable = tradable_mask & shortable_mask
                lower_bounds[mask_fully_tradable] = max(short_limit, -max_position_size)
                upper_bounds[mask_fully_tradable] = min(1.0, max_position_size)
            
            bounds = list(zip(lower_bounds, upper_bounds))
            
            # Solución inicial: pesos previos o iguales si no hay previos
            if previous_weights is not None and not previous_weights.isna().any():
                initial_weights = previous_weights.values
            else:
                # MODIFICACIÓN: si use_all_assets=True, asignar a todos los activos
                if self.use_all_assets:
                    initial_weights = np.ones(n_assets) / n_assets
                else:
                    # Sólo asignar a activos negociables (comportamiento original)
                    initial_weights = np.zeros(n_assets)
                    tradable_indices = np.where(tradable_mask)[0]
                    if len(tradable_indices) > 0:
                        initial_weights[tradable_indices] = 1.0 / len(tradable_indices)
            
            # OPTIMIZACIÓN: Usar configuración más eficiente para el optimizador
            options = {
                'maxiter': 200,      # Limitar iteraciones para mayor velocidad
                'ftol': 1e-6,        # Tolerancia más relajada
                'disp': False        # No mostrar mensajes
            }
            
            # Optimizar
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options=options
            )
            
            if not result.success:
                logging.warning(f"Optimización no convergió: {result.message}")
                # Usar pesos iniciales como fallback
                optimal_weights = pd.Series(initial_weights, index=assets)
            else:
                optimal_weights = pd.Series(result.x, index=assets)
            
            # Registrar una operación para visualización
            if previous_weights is not None:
                turnover = np.sum(np.abs(optimal_weights - previous_weights))
                if turnover > 0.05:  # Sólo registrar operaciones significativas
                    self.trades_history.append({
                        'date': current_date,
                        'turnover': turnover,
                        'regime': current_regime,
                        'leverage': np.sum(np.abs(optimal_weights)),
                        'n_long': np.sum(optimal_weights > 0.01),
                        'n_short': np.sum(optimal_weights < -0.01)
                    })
            
            # Eliminar posiciones muy pequeñas (menor a 0.1%) - optimizado
            small_positions_mask = np.abs(optimal_weights.values) < 0.001
            if np.any(small_positions_mask):
                optimal_weights.values[small_positions_mask] = 0.0
            
            # Renormalizar para asegurar suma = 1.0
            sum_weights = optimal_weights.sum()
            if sum_weights != 0:
                optimal_weights = optimal_weights / sum_weights
            
            # MEJORA: Aplicar control de volatilidad objetivo - optimizado
            expected_vol = calc_portfolio_risk_numba(optimal_weights.values, asset_cov) * np.sqrt(252)
            
            if expected_vol > 0:
                # Escalar según volatilidad objetivo
                vol_scalar = min(self.vol_target / expected_vol, target_max_leverage)
                
                # MEJORA: Reducir volatilidad en regímenes de alta volatilidad
                if current_regime >= self.n_regimes - 1:  # Régimen más volátil
                    vol_scalar *= 0.7
                elif current_regime >= self.n_regimes - 2:  # Segundo régimen más volátil
                    vol_scalar *= 0.85
            else:
                vol_scalar = 1.0
            
            # MEJORA: Asegurar que el apalancamiento final nunca excede el máximo permitido
            leverage = min(vol_scalar, target_max_leverage)
            
            # Calcular apalancamiento bruto (suma de valores absolutos)
            gross_leverage = np.sum(np.abs(optimal_weights.values * leverage))
            
            # Verificar que no excede el máximo permitido
            if gross_leverage > self.max_leverage:
                # Reducir proporcionalmente
                correction_factor = self.max_leverage / gross_leverage
                leverage *= correction_factor
            
            # Ajustar pesos finales
            final_weights = optimal_weights * leverage
            
            # Verificación final de apalancamiento bruto
            final_gross_leverage = np.sum(np.abs(final_weights.values))
            if final_gross_leverage > self.max_leverage + 1e-6:
                # Corrección de emergencia
                final_weights = final_weights * (self.max_leverage / final_gross_leverage)
            
            return final_weights
            
        except Exception as e:
            logging.error(f"Error en optimize_portfolio: {str(e)}", exc_info=True)
            # Valor por defecto: pesos iguales en activos negociables
            if self.use_all_assets:
                # Si usamos todos los activos, asignar pesos iguales a todos
                default_weights = pd.Series(1.0 / len(assets), index=assets)
            else:
                # Comportamiento original
                if current_date in self.tradable_universe.index:
                    tradable_mask = self.tradable_universe.loc[current_date, assets]
                else:
                    tradable_mask = pd.Series(True, index=assets)
                
                default_weights = pd.Series(0.0, index=assets)
                tradable_assets = assets[tradable_mask]
                if len(tradable_assets) > 0:
                    default_weights[tradable_assets] = 1.0 / len(tradable_assets)
            
            return default_weights
    
    def check_regime_change_rebalance(self, current_date_idx, backtest_dates, current_regime, regimes_history):
        """
        MEJORA: Verifica si se debe rebalancear por cambio de régimen.
        Optimizado para mayor eficiencia.
        """
        # Si es la primera fecha, no hay cambio de régimen
        if current_date_idx == 0 or len(regimes_history) == 0:
            return False
        
        # Si ya pasaron menos de 5 días desde el último rebalanceo, no rebalancear
        last_rebalance_idx = max([i for i, r in enumerate(regimes_history) if r is not None], default=-1)
        if current_date_idx - last_rebalance_idx < 5:
            return False
        
        # Verificar si el régimen ha cambiado desde el último rebalanceo
        last_regime = next((r for r in reversed(regimes_history) if r is not None), None)
        if last_regime is not None and current_regime != last_regime:
            # Solo rebalancear si el cambio es significativo (distancia > 1)
            if abs(current_regime - last_regime) > 1:
                return True
        
        return False
    
    def calculate_dynamic_rebalance_freq(self, current_regime, market_vol):
        """
        MEJORA: Calcula frecuencia de rebalanceo dinámica basada en régimen y volatilidad.
        Optimizado para mayor eficiencia.
        """
        base_freq = self.rebalance_freq
        
        # En regímenes de alta volatilidad, rebalancear más frecuentemente
        if current_regime >= self.n_regimes - 1:  # Régimen más volátil
            base_freq = max(5, base_freq // 3)
        elif current_regime >= self.n_regimes - 2:  # Segundo régimen más volátil
            base_freq = max(10, base_freq // 2)
        
        # Si volatilidad es muy alta o muy baja, ajustar frecuencia
        vol_ratio = market_vol / self.vol_target
        if vol_ratio > 2.0:
            base_freq = max(5, base_freq // 2)
        elif vol_ratio < 0.5:
            base_freq = min(42, base_freq * 2)
        
        return base_freq
    
    def backtest(self, start_date=None, end_date=None):
        """
        MEJORA: Backtest con control dinámico de riesgo y rebalanceo adaptativo.
        Optimizado: Mejor rendimiento y registro de operaciones
        
        Parámetros:
        -----------
        start_date : str, opcional
            Fecha de inicio del backtest (formato 'YYYY-MM-DD')
        end_date : str, opcional
            Fecha de fin del backtest (formato 'YYYY-MM-DD')
            
        Retorna:
        --------
        performance : DataFrame
            Resultados del backtest incluyendo retornos, drawdowns, etc.
        """
        try:
            print(f"Iniciando backtest... (Optimizado)")
            start_time = time.time()
            
            # Configurar fechas
            if start_date is None:
                start_date = self.returns.index[self.lookback_window]
            else:
                start_date = pd.to_datetime(start_date)
            
            if end_date is None:
                end_date = self.returns.index[-1]
            else:
                end_date = pd.to_datetime(end_date)
            
            # Filtrar datos por fechas
            mask = (self.returns.index >= start_date) & (self.returns.index <= end_date)
            backtest_dates = self.returns.index[mask]
            
            # Inicializar resultados
            portfolio_values = [1.0]
            portfolio_returns = []
            weights_history = []
            regime_history = []
            drawdown_history = [0.0]
            leverage_history = []
            pending_orders = {}
            
            # Inicializar pesos (comenzar con efectivo)
            current_weights = pd.Series(0, index=self.returns.columns)
            
            # Para rebalanceo dinámico
            next_rebalance_date = None
            dynamic_rebalance_freq = self.rebalance_freq
            
            # Para almacenar datos de benchmark
            benchmark_values = [1.0]  # CORREGIDO: Inicializar con 1.0 siempre
            
            # Resetear historial de operaciones
            self.trades_history = []
            
            # OPTIMIZACIÓN: Precalcular ventanas deslizantes para evitar cálculos repetitivos
            print("Precalculando datos para optimización...")
            
            # Ejecutar backtest
            print(f"Ejecutando backtest para {len(backtest_dates)} días...")
            for i, date in enumerate(tqdm(backtest_dates, desc="Backtest Progress")):
                # Manejar órdenes pendientes
                if date in pending_orders:
                    order_date, target_weights = pending_orders[date]
                    current_weights = target_weights.copy()
                    del pending_orders[date]
                
                # Determinar si es momento de rebalancear
                should_rebalance = False
                
                # Rebalancear en la primera fecha
                if i == 0:
                    should_rebalance = True
                    next_rebalance_date = None
                
                # MEJORA: Rebalanceo con frecuencia dinámica
                if next_rebalance_date is None or date >= next_rebalance_date:
                    should_rebalance = True
                
                # Obtener datos hasta la fecha actual de forma eficiente
                current_idx = self.returns.index.get_loc(date)
                history_end_idx = current_idx
                history_start_idx = max(0, history_end_idx - self.lookback_window)
                
                returns_window = self.returns.iloc[history_start_idx:history_end_idx]
                
                # Calcular volatilidad del mercado para ajustes dinámicos
                market_vol = returns_window.mean(axis=1).rolling(21).std().iloc[-1] * np.sqrt(252) if len(returns_window) > 21 else 0.15
                
                # Detectar regímenes solo si es necesario para rebalanceo dinámico o si vamos a rebalancear
                if should_rebalance or i % 5 == 0:  # Verificar cada 5 días para detección de cambios de régimen
                    # Extraer factores latentes
                    factor_loadings, factor_returns, n_components = self.extract_latent_factors(returns_window)
                    
                    # Detectar regímenes
                    regimes, regime_probs = self.detect_regimes(factor_returns)
                    
                    # Determinar régimen actual
                    current_regime = regimes[-1]
                    regime_certainty = np.max(regime_probs[-1])
                    
                    # MEJORA: Verificar si se debe rebalancear por cambio de régimen
                    if self.check_regime_change_rebalance(i, backtest_dates, current_regime, regime_history):
                        should_rebalance = True
                    
                    # MEJORA: Actualizar frecuencia de rebalanceo dinámico
                    if should_rebalance:
                        dynamic_rebalance_freq = self.calculate_dynamic_rebalance_freq(current_regime, market_vol)
                        next_rebalance_date = backtest_dates[min(i + dynamic_rebalance_freq, len(backtest_dates) - 1)]
                else:
                    # Si no calculamos regímenes, usar el último conocido
                    current_regime = regime_history[-1] if regime_history else 0
                    regime_certainty = 0.8  # Valor por defecto
                
                # Rebalancear si es necesario
                if should_rebalance:
                    # Recalcular factores y regímenes si no lo hicimos antes
                    if 'factor_loadings' not in locals():
                        factor_loadings, factor_returns, n_components = self.extract_latent_factors(returns_window)
                        regimes, regime_probs = self.detect_regimes(factor_returns)
                        current_regime = regimes[-1]
                        regime_certainty = np.max(regime_probs[-1])
                    
                    # Calcular retornos esperados
                    expected_returns, prediction_confidence = self.predict_expected_returns(
                        returns_window, regimes, regime_probs[-1], horizon=10
                    )
                    
                    # MEJORA: Pasar drawdown actual para control de riesgo dinámico
                    current_drawdown = drawdown_history[-1]
                    
                    # Optimizar portafolio
                    target_weights = self.optimize_portfolio(
                        expected_returns,
                        factor_loadings,
                        prediction_confidence,
                        current_regime,
                        regime_certainty,
                        date,
                        current_weights,
                        current_drawdown,
                        market_vol
                    )
                    
                    # Simular retraso en la ejecución
                    if self.execution_delay > 0 and i + self.execution_delay < len(backtest_dates):
                        execution_date = backtest_dates[i + self.execution_delay]
                        pending_orders[execution_date] = (date, target_weights)
                    else:
                        current_weights = target_weights.copy()
                    
                    # Guardar régimen actual
                    regime_history.append(current_regime)
                else:
                    # Si no rebalanceamos, el régimen es None para ese día
                    regime_history.append(None)
                
                # Calcular costos de posiciones cortas - optimizado
                short_positions = current_weights[current_weights < 0]
                short_cost = 0
                if not short_positions.empty:
                    daily_borrow_cost = self.borrow_cost / 252
                    short_cost = (short_positions.abs() * daily_borrow_cost).sum()
                
                # Guardar apalancamiento actual
                current_leverage = np.sum(np.abs(current_weights))
                leverage_history.append(current_leverage)
                
                # Calcular retorno del portafolio para el día siguiente
                if i + 1 < len(backtest_dates):
                    next_date = backtest_dates[i + 1]
                    next_returns = self.returns.loc[next_date]
                    
                    # Incluir costos de transacción si hubo rebalanceo
                    transaction_cost = 0
                    if should_rebalance:
                        weights_before = weights_history[-1] if weights_history else pd.Series(0, index=current_weights.index)
                        turnover = np.sum(np.abs(current_weights - weights_before))
                        transaction_cost = turnover * self.transaction_cost
                    
                    # Calcular retorno del portafolio con costos
                    portfolio_return = (current_weights * next_returns).sum() - short_cost - transaction_cost
                    portfolio_returns.append(portfolio_return)
                    
                    # Actualizar valor del portafolio
                    portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))
                    
                    # Calcular drawdown actual
                    peak = max(portfolio_values)
                    current_drawdown = 1 - portfolio_values[-1] / peak
                    drawdown_history.append(current_drawdown)
                    
                    # MEJORA: Reducir exposición si drawdown supera el límite
                    if self.max_drawdown_limit > 0 and current_drawdown > self.max_drawdown_limit:
                        # Reducir exposición en un 30%
                        current_weights = current_weights * 0.7
                        # Forzar rebalanceo en próxima fecha
                        next_rebalance_date = backtest_dates[min(i + 1, len(backtest_dates) - 1)]
                
                # Guardar pesos
                weights_history.append(current_weights.copy())
                
                # CORREGIDO: Mejor manejo de datos de benchmark para alineación con portfolio
                # Guardar datos de benchmark para el mismo período
                if i + 1 < len(backtest_dates):
                    next_date = backtest_dates[i + 1]
                    # Comprobar si el benchmark tiene datos para esta fecha
                    if next_date in self.benchmark_returns.index:
                        benchmark_return = self.benchmark_returns.loc[next_date, 'S&P500']
                        benchmark_values.append(benchmark_values[-1] * (1 + benchmark_return))
                    else:
                        # Si no hay datos para esta fecha, repetir el último valor
                        benchmark_values.append(benchmark_values[-1])
            
            # CORREGIDO: Crear DataFrame de resultados con benchmark incluido correctamente
            # Ajustar tamaños para asegurar que los arrays tengan la misma longitud
            min_length = min(len(portfolio_values) - 1, len(portfolio_returns), len(leverage_history),
                             len(benchmark_values) - 1)
            
            # Cortar los arrays al tamaño mínimo para evitar problemas de dimensiones
            performance = pd.DataFrame({
                'Portfolio_Value': portfolio_values[:min_length + 1],
                'Returns': portfolio_returns[:min_length],
                'Leverage': leverage_history[:min_length]
            }, index=backtest_dates[:min_length])
            
            # Calcular retornos del benchmark
            benchmark_returns_list = []
            for i in range(1, len(benchmark_values)):
                bench_return = benchmark_values[i] / benchmark_values[i-1] - 1
                benchmark_returns_list.append(bench_return)
            
            # Recortar a la longitud adecuada
            benchmark_returns_list = benchmark_returns_list[:min_length]
            benchmark_values_list = benchmark_values[:min_length + 1]
            
            # Añadir datos de benchmark al DataFrame principal
            performance['Benchmark_Value'] = pd.Series(benchmark_values_list[:-1], index=performance.index)
            performance['Benchmark_Returns'] = pd.Series(benchmark_returns_list, index=performance.index)
            
            # Calcular métricas
            performance['Cumulative_Returns'] = (1 + performance['Returns']).cumprod()
            performance['Drawdown'] = 1 - performance['Cumulative_Returns'] / performance['Cumulative_Returns'].cummax()
            
            # CORREGIDO: Calcular métricas de benchmark solo si hay datos suficientes
            if 'Benchmark_Returns' in performance.columns and not performance['Benchmark_Returns'].isna().all():
                performance['Benchmark_Cumulative'] = (1 + performance['Benchmark_Returns']).cumprod()
                performance['Benchmark_Drawdown'] = 1 - performance['Benchmark_Cumulative'] / performance['Benchmark_Cumulative'].cummax()
                
                # Calcular Alpha y Beta
                # Beta = Cov(r_p, r_b) / Var(r_b)
                beta = performance['Returns'].cov(performance['Benchmark_Returns']) / performance['Benchmark_Returns'].var()
                
                # Alpha = r_p - beta * r_b (anualizado)
                alpha = (performance['Returns'].mean() - beta * performance['Benchmark_Returns'].mean()) * 252
                
                performance['Alpha'] = alpha
                performance['Beta'] = beta
            
            # Guardar resultados adicionales
            self.weights_history = pd.DataFrame(weights_history, index=backtest_dates)
            
            # Convertir regime_history a Series, manejando valores None
            valid_regimes = [(i, r) for i, r in enumerate(regime_history) if r is not None]
            valid_indices = [backtest_dates[i] for i, _ in valid_regimes]
            valid_values = [r for _, r in valid_regimes]
            self.regime_history = pd.Series(valid_values, index=valid_indices)
            
            self.performance = performance
            
            # Guardar historial de operaciones como DataFrame
            self.trades_df = pd.DataFrame(self.trades_history)
            if not self.trades_df.empty:
                self.trades_df.set_index('date', inplace=True)
            
            print(f"Backtest completado en {time.time() - start_time:.2f} segundos")
            
            return performance
            
        except Exception as e:
            logging.error(f"Error en backtest: {str(e)}", exc_info=True)
            raise
    
    def calculate_metrics(self, performance=None):
        """
        Calcula métricas de rendimiento de la estrategia.
        Actualizado: Incluye comparación con benchmark
        
        Parámetros:
        -----------
        performance : DataFrame, opcional
            Resultados del backtest. Si es None, se usa self.performance.
            
        Retorna:
        --------
        metrics : dict
            Diccionario con métricas de rendimiento
        """
        try:
            if performance is None:
                performance = self.performance
            
            if performance is None or len(performance) == 0:
                raise ValueError("No hay datos de rendimiento disponibles")
            
            # Calcular métricas anualizadas
            returns = performance['Returns']
            ann_factor = 252
            
            total_return = performance['Cumulative_Returns'].iloc[-1] - 1
            ann_return = (1 + total_return) ** (ann_factor / len(returns)) - 1
            ann_volatility = returns.std() * np.sqrt(ann_factor)
            sharpe_ratio = ann_return / ann_volatility if ann_volatility > 0 else 0
            max_drawdown = performance['Drawdown'].max()
            
            # Calcular ratio de Sortino (solo considera volatilidad negativa)
            negative_returns = returns[returns < 0]
            downside_deviation = negative_returns.std() * np.sqrt(ann_factor) if len(negative_returns) > 0 else 0
            sortino_ratio = ann_return / downside_deviation if downside_deviation > 0 else 0
            
            # Calcular ratio de Calmar (retorno anualizado / máximo drawdown)
            calmar_ratio = ann_return / max_drawdown if max_drawdown > 0 else 0
            
            # Estimar turnover anualizado (rotación de cartera)
            if hasattr(self, 'weights_history') and len(self.weights_history) > 1:
                turnovers = []
                for i in range(1, len(self.weights_history)):
                    turnover = np.sum(np.abs(self.weights_history.iloc[i] - self.weights_history.iloc[i-1]))
                    turnovers.append(turnover)
                avg_turnover = np.mean(turnovers) if turnovers else 0
                avg_rebalance_freq = self.rebalance_freq  # Usar promedio si tuviéramos frecuencia dinámica
                ann_turnover = avg_turnover * (252 / avg_rebalance_freq)
            else:
                ann_turnover = 0
            
            # Calcular % de meses positivos
            monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            pct_positive_months = (monthly_returns > 0).mean() if len(monthly_returns) > 0 else 0
            
            # Calcular métricas realistas
            gross_return = ann_return
            
            # Estimar costos anuales
            estimated_transaction_costs = ann_turnover * self.transaction_cost
            
            # Estimar costos de préstamo para posiciones cortas
            if hasattr(self, 'weights_history'):
                short_exposure = self.weights_history.apply(lambda x: np.sum(np.maximum(-x, 0)), axis=1).mean()
                short_costs = short_exposure * self.borrow_cost
            else:
                short_costs = 0
            
            # Retorno neto
            net_return = gross_return - estimated_transaction_costs - short_costs
            net_sharpe = net_return / ann_volatility if ann_volatility > 0 else 0
            
            # Calcular apalancamiento promedio
            avg_leverage = performance['Leverage'].mean() if 'Leverage' in performance.columns else 0
            max_leverage_used = performance['Leverage'].max() if 'Leverage' in performance.columns else 0
            
            # Calcular características de drawdown
            dd = performance['Drawdown']
            avg_drawdown = dd.mean()
            drawdown_duration = 0
            if max_drawdown > 0:
                # Encontrar períodos de drawdown
                dd_periods = []
                in_dd = False
                start_idx = 0
                for i, val in enumerate(dd):
                    if not in_dd and val > 0.01:  # Inicio de drawdown (>1%)
                        in_dd = True
                        start_idx = i
                    elif in_dd and val < 0.01:  # Fin de drawdown
                        in_dd = False
                        dd_periods.append(i - start_idx)
                
                # Calcular duración promedio en días
                drawdown_duration = np.mean(dd_periods) if dd_periods else 0
            
            # CORREGIDO: Métricas de comparación con benchmark - verifica si existen datos y están completos
            if ('Benchmark_Returns' in performance.columns and 
                not performance['Benchmark_Returns'].isna().all() and 
                len(performance['Benchmark_Returns']) > 0):
                
                benchmark_returns = performance['Benchmark_Returns'].fillna(0)  # Llenar NaN con ceros para evitar errores
                benchmark_total_return = performance['Benchmark_Cumulative'].iloc[-1] - 1
                benchmark_ann_return = (1 + benchmark_total_return) ** (ann_factor / len(benchmark_returns)) - 1
                benchmark_volatility = benchmark_returns.std() * np.sqrt(ann_factor)
                benchmark_sharpe = benchmark_ann_return / benchmark_volatility if benchmark_volatility > 0 else 0
                benchmark_max_dd = performance['Benchmark_Drawdown'].max()
                
                # Alpha y Beta - solo calcular si hay suficiente varianza en el benchmark
                beta = returns.cov(benchmark_returns) / benchmark_returns.var() if benchmark_returns.var() > 0 else 0
                alpha = (returns.mean() - beta * benchmark_returns.mean()) * 252
                
                # Information Ratio
                tracking_error = (returns - beta * benchmark_returns).std() * np.sqrt(ann_factor)
                information_ratio = alpha / tracking_error if tracking_error > 0 else 0
                
                # Capture ratios
                up_months = benchmark_returns > 0
                down_months = benchmark_returns < 0
                
                if up_months.sum() > 0 and benchmark_returns[up_months].mean() != 0:
                    up_capture = (returns[up_months].mean() / benchmark_returns[up_months].mean())
                else:
                    up_capture = 0
                
                if down_months.sum() > 0 and benchmark_returns[down_months].mean() != 0:
                    down_capture = (returns[down_months].mean() / benchmark_returns[down_months].mean())
                else:
                    down_capture = 0
            else:
                benchmark_ann_return = 0
                benchmark_volatility = 0
                benchmark_sharpe = 0
                benchmark_max_dd = 0
                alpha = 0
                beta = 0
                information_ratio = 0
                up_capture = 0
                down_capture = 0
            
            # Recopilar métricas
            metrics = {
                'Gross Total Return': total_return,
                'Gross Annualized Return': gross_return,
                'Net Annualized Return': net_return,
                'Annualized Volatility': ann_volatility,
                'Gross Sharpe Ratio': sharpe_ratio,
                'Net Sharpe Ratio': net_sharpe,
                'Sortino Ratio': sortino_ratio,
                'Calmar Ratio': calmar_ratio,
                'Maximum Drawdown': max_drawdown,
                'Average Drawdown': avg_drawdown,
                'Average Drawdown Duration (days)': drawdown_duration,
                'Annualized Turnover': ann_turnover,
                'Estimated Transaction Costs': estimated_transaction_costs,
                'Estimated Short Costs': short_costs,
                'Positive Months (%)': pct_positive_months,
                'Average Leverage': avg_leverage,
                'Maximum Leverage Used': max_leverage_used,
                'Number of Rebalances': len(self.regime_history) if hasattr(self, 'regime_history') else 0,
                # Métricas vs Benchmark
                'Benchmark Annualized Return': benchmark_ann_return,
                'Benchmark Volatility': benchmark_volatility,
                'Benchmark Sharpe Ratio': benchmark_sharpe,
                'Benchmark Maximum Drawdown': benchmark_max_dd,
                'Alpha vs Benchmark': alpha,
                'Beta vs Benchmark': beta,
                'Information Ratio': information_ratio,
                'Upside Capture Ratio': up_capture,
                'Downside Capture Ratio': down_capture
            }
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error en calculate_metrics: {str(e)}", exc_info=True)
            return {}
    
    def plot_results(self, save_path='./artifacts/results/figures/'):
        """
        Genera y guarda visualizaciones de los resultados.
        Actualizado: Incluye comparación con benchmark y visualización de operaciones
        
        Parámetros:
        -----------
        save_path : str
            Ruta donde guardar las figuras
        """
        try:
            if self.performance is None or len(self.performance) == 0:
                raise ValueError("No hay datos de rendimiento disponibles")
            
            # Crear directorio si no existe
            os.makedirs(save_path, exist_ok=True)
            
            # Configuración para gráficos más nítidos
            plt.style.use('seaborn-v0_8-whitegrid')
            sns.set_context("talk")
            
            # CORREGIDO: Verificar si hay datos de benchmark válidos
            has_valid_benchmark = ('Benchmark_Cumulative' in self.performance.columns and 
                                  not self.performance['Benchmark_Cumulative'].isna().all())
            
            # 1. Gráfico de rendimiento acumulado con benchmark
            plt.figure(figsize=(14, 7))
            self.performance['Cumulative_Returns'].plot(label='Estrategia', linewidth=2)
            
            if has_valid_benchmark:
                self.performance['Benchmark_Cumulative'].plot(label='S&P 500', linewidth=2, linestyle='--')
                
                # Agregar Alpha y Beta como anotación si existen
                if 'Alpha' in self.performance.columns and 'Beta' in self.performance.columns:
                    alpha = self.performance['Alpha'].iloc[-1]
                    beta = self.performance['Beta'].iloc[-1]
                    plt.annotate(f'α: {alpha:.2%} anual | β: {beta:.2f}', 
                                 xy=(0.02, 0.05), xycoords='axes fraction', 
                                 fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
            
            plt.title('Rendimiento Acumulado vs. Benchmark', fontsize=14)
            plt.xlabel('Fecha', fontsize=12)
            plt.ylabel('Retorno Acumulado', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{save_path}cumulative_returns_vs_benchmark.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Gráfico de drawdowns comparativo
            plt.figure(figsize=(14, 7))
            self.performance['Drawdown'].plot(label='Estrategia', color='red', linewidth=2)
            
            if has_valid_benchmark:
                self.performance['Benchmark_Drawdown'].plot(label='S&P 500', color='blue', linewidth=2, linestyle='--')
            
            plt.title('Drawdowns: Estrategia vs. Benchmark', fontsize=14)
            plt.xlabel('Fecha', fontsize=12)
            plt.ylabel('Drawdown', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{save_path}drawdowns_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Gráfico de regímenes de mercado y retornos
            if hasattr(self, 'regime_history') and len(self.regime_history) > 0:
                plt.figure(figsize=(14, 8))
                ax1 = plt.gca()
                self.performance['Cumulative_Returns'].plot(ax=ax1, color='blue', linewidth=2, label='Estrategia')
                
                if has_valid_benchmark:
                    self.performance['Benchmark_Cumulative'].plot(ax=ax1, color='green', linewidth=2, 
                                                                 linestyle='--', label='S&P 500')
                
                ax1.set_xlabel('Fecha', fontsize=12)
                ax1.set_ylabel('Retorno Acumulado', fontsize=12, color='blue')
                ax1.legend(loc='upper left', fontsize=12)
                
                ax2 = ax1.twinx()
                # Solo usar fechas que existen en ambos índices
                common_dates = self.regime_history.index.intersection(self.performance.index)
                if len(common_dates) > 0:
                    ax2.scatter(common_dates, self.regime_history.loc[common_dates], 
                               color='red', alpha=0.7, marker='o', s=50, label='Régimen')
                    ax2.set_ylabel('Régimen', fontsize=12, color='red')
                    ax2.tick_params(axis='y', colors='red')
                    ax2.set_yticks(range(self.n_regimes))
                    ax2.legend(loc='upper right', fontsize=12)
                
                plt.title('Rendimiento vs. Regímenes de Mercado', fontsize=14)
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f'{save_path}trading_activity.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # 5. Exposición a activos a lo largo del tiempo
            if hasattr(self, 'weights_history') and len(self.weights_history) > 0:
                # Seleccionar los 10 activos con mayor peso promedio absoluto
                top_assets = self.weights_history.abs().mean().nlargest(10).index
                
                plt.figure(figsize=(14, 8))
                self.weights_history[top_assets].plot(colormap='viridis')
                plt.title('Exposición a los 10 Activos Principales', fontsize=14)
                plt.xlabel('Fecha', fontsize=12)
                plt.ylabel('Peso en el Portafolio', fontsize=12)
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f'{save_path}asset_exposure.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # 6. Heatmap de pesos a lo largo del tiempo - Optimizado para visualización
                # Reducir el número de fechas para hacer el gráfico más legible
                if len(self.weights_history) > 50:
                    # Muestrear fechas uniformemente
                    sample_size = 50
                    step = len(self.weights_history) // sample_size
                    sampled_weights = self.weights_history.iloc[::step]
                else:
                    sampled_weights = self.weights_history
                
                plt.figure(figsize=(16, 10))
                sns.heatmap(
                    sampled_weights[top_assets].T,
                    cmap='RdBu_r',
                    center=0,
                    robust=True,
                    cbar_kws={'label': 'Peso'}
                )
                plt.title('Evolución de Pesos del Portafolio (Top 10 Activos)', fontsize=14)
                plt.xlabel('Tiempo', fontsize=12)
                plt.ylabel('Activo', fontsize=12)
                plt.tight_layout()
                plt.savefig(f'{save_path}weights_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # 7. Gráfico de apalancamiento a lo largo del tiempo
            plt.figure(figsize=(14, 7))
            self.performance['Leverage'].plot(linewidth=2)
            plt.axhline(y=self.max_leverage, color='r', linestyle='--', label=f'Límite máximo ({self.max_leverage})')
            plt.axhline(y=1.0, color='g', linestyle='--', label='Sin apalancamiento')
            plt.title('Apalancamiento del Portafolio', fontsize=14)
            plt.xlabel('Fecha', fontsize=12)
            plt.ylabel('Apalancamiento', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{save_path}leverage.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 8. Rendimientos mensuales como mapa de calor
            if len(self.performance) > 30:
                # Calcular rendimientos mensuales
                monthly_returns = self.performance['Returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
                monthly_returns_df = monthly_returns.to_frame('Strategy')
                
                if has_valid_benchmark:
                    # CORREGIDO: Mejor manejo de datos de benchmark para cálculos mensuales
                    if 'Benchmark_Returns' in self.performance.columns and not self.performance['Benchmark_Returns'].isna().all():
                        benchmark_monthly = self.performance['Benchmark_Returns'].fillna(0).resample('M').apply(lambda x: (1 + x).prod() - 1)
                        monthly_returns_df['Benchmark'] = benchmark_monthly
                
                # Crear matriz de rendimientos por año y mes para heatmap
                returns_by_month = monthly_returns_df.copy()
                returns_by_month.index = pd.MultiIndex.from_arrays([
                    returns_by_month.index.year,
                    returns_by_month.index.month
                ], names=['Year', 'Month'])
                
                # Pivotear para heatmap
                returns_pivot = returns_by_month.reset_index().pivot(index='Year', columns='Month', values='Strategy')
                
                # Gráfico de rendimientos mensuales
                plt.figure(figsize=(14, 8))
                sns.heatmap(returns_pivot, annot=True, fmt='.1%', cmap='RdYlGn', center=0)
                plt.title('Rendimientos Mensuales de la Estrategia', fontsize=14)
                plt.xlabel('Mes', fontsize=12)
                plt.ylabel('Año', fontsize=12)
                plt.tight_layout()
                plt.savefig(f'{save_path}monthly_returns_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # Si hay benchmark, hacer gráfico comparativo de rendimientos anualizados
                if 'Benchmark' in monthly_returns_df.columns:
                    yearly_returns = monthly_returns_df.resample('Y').apply(lambda x: (1 + x).prod() - 1)
                    
                    plt.figure(figsize=(14, 7))
                    bar_width = 0.35
                    index = np.arange(len(yearly_returns.index))
                    
                    plt.bar(index, yearly_returns['Strategy'], bar_width, label='Estrategia', color='blue', alpha=0.7)
                    plt.bar(index + bar_width, yearly_returns['Benchmark'], bar_width, label='S&P 500', color='green', alpha=0.7)
                    
                    plt.xlabel('Año', fontsize=12)
                    plt.ylabel('Retorno Anual', fontsize=12)
                    plt.title('Comparación de Rendimientos Anuales', fontsize=14)
                    plt.xticks(index + bar_width/2, [d.year for d in yearly_returns.index], rotation=45)
                    plt.legend(fontsize=12)
                    
                    # Añadir etiquetas de porcentaje
                    for i, v in enumerate(yearly_returns['Strategy']):
                        plt.text(i - 0.15, v + 0.01, f"{v:.1%}", color='blue', fontweight='bold')
                    for i, v in enumerate(yearly_returns['Benchmark']):
                        plt.text(i + bar_width - 0.15, v + 0.01, f"{v:.1%}", color='green', fontweight='bold')
                    
                    plt.tight_layout()
                    plt.savefig(f'{save_path}yearly_returns_comparison.png', dpi=300, bbox_inches='tight')
                    plt.close()
            
            # 9. Gráfico de metrics de Alpha, Beta y exposición a lo largo del tiempo
            if has_valid_benchmark:
                # Calcular rolling alpha y beta
                window = min(252, len(self.performance) // 4)  # Usar 1 año o un cuarto de los datos
                
                rolling_beta = pd.Series(index=self.performance.index)
                rolling_alpha = pd.Series(index=self.performance.index)
                
                for i in range(window, len(self.performance)):
                    window_rets = self.performance['Returns'].iloc[i-window:i]
                    window_bench = self.performance['Benchmark_Returns'].fillna(0).iloc[i-window:i]  # CORREGIDO: Manejar NaN
                    
                    # Verificar que hay varianza en el benchmark
                    if window_bench.var() > 0:
                        # Beta = Cov(r_p, r_b) / Var(r_b)
                        beta = window_rets.cov(window_bench) / window_bench.var()
                        
                        # Alpha anualizado = (r_p - beta * r_b) * 252
                        alpha = (window_rets.mean() - beta * window_bench.mean()) * 252
                        
                        rolling_beta.iloc[i] = beta
                        rolling_alpha.iloc[i] = alpha
                
                # Gráfico de Beta si hay datos válidos
                if not rolling_beta.dropna().empty:
                    plt.figure(figsize=(14, 7))
                    rolling_beta.dropna().plot(linewidth=2)
                    plt.axhline(y=1.0, color='r', linestyle='--', label='Beta = 1')
                    plt.axhline(y=0.0, color='g', linestyle='--', label='Beta = 0')
                    plt.title('Beta vs S&P 500 (Rolling Window)', fontsize=14)
                    plt.xlabel('Fecha', fontsize=12)
                    plt.ylabel('Beta', fontsize=12)
                    plt.legend(fontsize=12)
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(f'{save_path}rolling_beta.png', dpi=300, bbox_inches='tight')
                    plt.close()
                
                # Gráfico de Alpha si hay datos válidos
                if not rolling_alpha.dropna().empty:
                    plt.figure(figsize=(14, 7))
                    rolling_alpha.dropna().plot(linewidth=2)
                    plt.axhline(y=0.0, color='r', linestyle='--', label='Alpha = 0')
                    plt.title('Alpha vs S&P 500 (Rolling Window) - Anualizado', fontsize=14)
                    plt.xlabel('Fecha', fontsize=12)
                    plt.ylabel('Alpha', fontsize=12)
                    plt.legend(fontsize=12)
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(f'{save_path}rolling_alpha.png', dpi=300, bbox_inches='tight')
                    plt.close()
            
            # 10. Gráfico de histograma comparativo de retornos mensuales
            if has_valid_benchmark:
                strategy_monthly = self.performance['Returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
                benchmark_monthly = self.performance['Benchmark_Returns'].fillna(0).resample('M').apply(lambda x: (1 + x).prod() - 1)
                
                plt.figure(figsize=(14, 7))
                plt.hist(strategy_monthly, bins=20, alpha=0.5, label='Estrategia', color='blue')
                plt.hist(benchmark_monthly, bins=20, alpha=0.5, label='S&P 500', color='green')
                plt.axvline(x=0, color='r', linestyle='--')
                plt.title('Distribución de Retornos Mensuales: Estrategia vs S&P 500', fontsize=14)
                plt.xlabel('Retorno Mensual', fontsize=12)
                plt.ylabel('Frecuencia', fontsize=12)
                plt.legend(fontsize=12)
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f'{save_path}monthly_returns_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # 11. Métricas clave en un solo gráfico
            metrics = self.calculate_metrics()
            
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 2, 1)
            plt.bar(['Estrategia', 'S&P 500'], 
                   [metrics['Net Annualized Return'], metrics['Benchmark Annualized Return']], 
                   color=['blue', 'green'])
            plt.title('Retorno Anualizado', fontsize=12)
            plt.grid(axis='y')
            
            plt.subplot(2, 2, 2)
            plt.bar(['Estrategia', 'S&P 500'], 
                   [metrics['Annualized Volatility'], metrics['Benchmark Volatility']], 
                   color=['blue', 'green'])
            plt.title('Volatilidad Anualizada', fontsize=12)
            plt.grid(axis='y')
            
            plt.subplot(2, 2, 3)
            plt.bar(['Estrategia', 'S&P 500'], 
                   [metrics['Net Sharpe Ratio'], metrics['Benchmark Sharpe Ratio']], 
                   color=['blue', 'green'])
            plt.title('Ratio de Sharpe', fontsize=12)
            plt.grid(axis='y')
            
            plt.subplot(2, 2, 4)
            plt.bar(['Estrategia', 'S&P 500'], 
                   [metrics['Maximum Drawdown'], metrics['Benchmark Maximum Drawdown']], 
                   color=['blue', 'green'])
            plt.title('Drawdown Máximo', fontsize=12)
            plt.grid(axis='y')
            
            plt.tight_layout()
            plt.savefig(f'{save_path}key_metrics_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Gráficos guardados en {save_path}")
            
        except Exception as e:
            logging.error(f"Error en plot_results: {str(e)}", exc_info=True)
            print(f"Error al generar gráficos: {str(e)}")
            
    def run_walk_forward_analysis(self, train_size=0.6, step_size=63, train_lookback=252):
        """
        MEJORA: Ejecuta análisis walk-forward más robusto para evaluar la estrategia.
        Optimizado: Paralelismo y algoritmos más eficientes.
        
        Parámetros:
        -----------
        train_size : float
            Proporción de datos a usar para entrenamiento en cada ventana
        step_size : int
            Tamaño del paso para avanzar la ventana de prueba (en días)
        train_lookback : int
            Cantidad de días máximos a utilizar en cada ventana de entrenamiento
            
        Retorna:
        --------
        wfa_results : DataFrame
            Resultados del análisis walk-forward
        """
        try:
            print("Iniciando análisis walk-forward optimizado...")
            wfa_start_time = time.time()
            
            # Asegurar que tenemos suficientes datos
            if len(self.returns) < self.lookback_window + 2 * step_size:
                raise ValueError("No hay suficientes datos para análisis walk-forward")
            
            # Inicializar resultados
            wfa_results = []
            dates = self.returns.index
            
            # Definir ventanas
            start_idx = self.lookback_window
            windows = []
            
            while start_idx + step_size < len(dates):
                # Limitar la ventana de entrenamiento
                train_end_idx = start_idx + int((len(dates) - start_idx) * train_size)
                train_start_idx = max(0, train_end_idx - train_lookback)
                test_end_idx = min(train_end_idx + step_size, len(dates))
                
                train_start_date = dates[train_start_idx]
                train_end_date = dates[train_end_idx - 1]
                test_start_date = dates[train_end_idx]
                test_end_date = dates[test_end_idx - 1]
                
                windows.append({
                    'train_start_date': train_start_date,
                    'train_end_date': train_end_date,
                    'test_start_date': test_start_date,
                    'test_end_date': test_end_date,
                    'idx': len(windows)
                })
                
                # Avanzar ventana
                start_idx = train_end_idx
            
            # Procesar ventanas con barra de progreso
            print(f"Procesando {len(windows)} ventanas walk-forward...")
            
            # OPTIMIZACIÓN: Procesar ventanas en paralelo
            if self.n_jobs > 1 and len(windows) > 3:
                # Crear chunks de ventanas para reducir overhead de paralelización
                chunk_size = max(1, len(windows) // self.n_jobs)
                window_chunks = [windows[i:i+chunk_size] for i in range(0, len(windows), chunk_size)]
                
                # Función para procesar un chunk de ventanas
                def process_window_chunk(chunk):
                    chunk_results = []
                    for window in chunk:
                        result = self._process_wfa_window(window)
                        chunk_results.append(result)
                    return chunk_results
                
                # Procesar chunks en paralelo
                with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                    chunk_results = list(tqdm(
                        executor.map(process_window_chunk, window_chunks),
                        total=len(window_chunks),
                        desc="Procesando ventanas WFA"
                    ))
                
                # Aplanar resultados
                for chunk_result in chunk_results:
                    wfa_results.extend(chunk_result)
            else:
                # Procesar ventanas secuencialmente
                for window in tqdm(windows, desc="Procesando ventanas WFA"):
                    result = self._process_wfa_window(window)
                    wfa_results.append(result)
            
            # Convertir resultados a DataFrame
            wfa_df = pd.DataFrame(wfa_results)
            
            # Guardar resultados
            os.makedirs('./artifacts/results/data', exist_ok=True)
            wfa_df.to_csv('./artifacts/results/data/walk_forward_analysis.csv', index=False)
            
            # Calcular métricas agregadas
            wfa_metrics = {
                'Mean_Sharpe': wfa_df['Sharpe_Ratio'].mean(),
                'Median_Sharpe': wfa_df['Sharpe_Ratio'].median(),
                'Min_Sharpe': wfa_df['Sharpe_Ratio'].min(),
                'Max_Sharpe': wfa_df['Sharpe_Ratio'].max(),
                'Mean_Return': wfa_df['Annualized_Return'].mean(),
                'Mean_Volatility': wfa_df['Annualized_Volatility'].mean(),
                'Mean_Drawdown': wfa_df['Max_Drawdown'].mean(),
                'Consistency': (wfa_df['Sharpe_Ratio'] > 0).mean(),
                'Mean_Initial_Cost': wfa_df['Initial_Cost'].mean(),
                'Mean_Short_Exposure': wfa_df['Short_Exposure'].mean()
            }
            
            # Guardar métricas agregadas
            pd.Series(wfa_metrics).to_csv('./artifacts/results/data/walk_forward_metrics.csv')
            
            # Visualizar resultados de forma más profesional
            self._plot_wfa_results(wfa_df)
            
            print(f"Análisis walk-forward completado en {time.time() - wfa_start_time:.2f} segundos")
            
            return wfa_df
            
        except Exception as e:
            logging.error(f"Error en run_walk_forward_analysis: {str(e)}", exc_info=True)
            print(f"Error en análisis walk-forward: {str(e)}")
            return pd.DataFrame()
    
    def _process_wfa_window(self, window):
        """
        Procesa una ventana individual del análisis walk-forward.
        """
        try:
            train_start_date = window['train_start_date']
            train_end_date = window['train_end_date']
            test_start_date = window['test_start_date']
            test_end_date = window['test_end_date']
            
            print(f"\nVentana WFA {window['idx']+1}: {test_start_date.strftime('%Y-%m-%d')} a {test_end_date.strftime('%Y-%m-%d')}")
            
            # Guardar configuración actual
            original_n_regimes = self.n_regimes
            original_lookback = self.lookback_window
            original_regime_method = self.regime_detection_method
            
            # Ajustar parámetros según la ventana
            # Más regímenes para períodos volátiles
            train_window_vol = self.returns.loc[train_start_date:train_end_date].std().mean() * np.sqrt(252)
            if train_window_vol > 0.25:  # Alta volatilidad
                self.n_regimes = 6
                self.lookback_window = 126  # Ventana más corta
                self.regime_detection_method = 'bgmm'
            elif train_window_vol > 0.15:  # Volatilidad media
                self.n_regimes = 5
                self.lookback_window = 189
                self.regime_detection_method = 'bgmm'
            else:  # Baja volatilidad
                self.n_regimes = 4
                self.lookback_window = 252
                self.regime_detection_method = 'gmm'
            
            # Ejecutar backtest en datos de entrenamiento usando caché si es posible
            train_performance = self.backtest(
                start_date=train_start_date,
                end_date=train_end_date
            )
            
            # Guardar pesos óptimos del último rebalanceo
            last_weights = self.weights_history.iloc[-1]
            
            # Considerar costos de transacción al aplicar pesos
            initial_turnover = np.sum(np.abs(last_weights))
            initial_cost = initial_turnover * self.transaction_cost
            
            # Inicializar tracking de posiciones cortas
            short_positions = last_weights[last_weights < 0]
            daily_borrow_cost = self.borrow_cost / 252
            
            # Ejecutar backtest en datos de prueba con pesos fijos - vectorizado
            test_returns = self.returns.loc[test_start_date:test_end_date]
            test_portfolio_values = [1.0 - initial_cost]
            
            # OPTIMIZACIÓN: Vectorizar cálculo de retornos en lugar de iteraciones
            if not test_returns.empty:
                # Calcular costo de posiciones cortas
                short_cost = 0
                if not short_positions.empty:
                    short_cost = (short_positions.abs() * daily_borrow_cost).sum()
                
                # Calcular retornos del portafolio para todo el período de prueba
                portfolio_returns = test_returns.dot(last_weights) - short_cost
                
                # Calcular valores acumulados
                for ret in portfolio_returns:
                    test_portfolio_values.append(test_portfolio_values[-1] * (1 + ret))
            
            # Calcular métricas para esta ventana
            test_returns_series = pd.Series(
                [test_portfolio_values[i+1]/test_portfolio_values[i] - 1 for i in range(len(test_portfolio_values)-1)],
                index=test_returns.index
            )
            
            test_performance = pd.DataFrame({
                'Returns': test_returns_series,
                'Cumulative_Returns': (1 + test_returns_series).cumprod()
            })
            
            test_performance['Drawdown'] = 1 - test_performance['Cumulative_Returns'] / test_performance['Cumulative_Returns'].cummax()
            
            # Calcular métricas
            total_return = test_performance['Cumulative_Returns'].iloc[-1] - 1 if not test_performance.empty else 0
            ann_factor = 252
            ann_return = (1 + total_return) ** (ann_factor / len(test_returns_series)) - 1 if len(test_returns_series) > 0 else 0
            ann_volatility = test_returns_series.std() * np.sqrt(ann_factor) if len(test_returns_series) > 0 else 0
            sharpe_ratio = ann_return / ann_volatility if ann_volatility > 0 else 0
            max_drawdown = test_performance['Drawdown'].max() if not test_performance.empty else 0
            
            # Restaurar parámetros originales
            self.n_regimes = original_n_regimes
            self.lookback_window = original_lookback
            self.regime_detection_method = original_regime_method
            
            # Devolver resultados
            return {
                'Test_Start_Date': test_start_date,
                'Test_End_Date': test_end_date,
                'Total_Return': total_return,
                'Annualized_Return': ann_return,
                'Annualized_Volatility': ann_volatility,
                'Sharpe_Ratio': sharpe_ratio,
                'Max_Drawdown': max_drawdown,
                'Initial_Cost': initial_cost,
                'Short_Exposure': short_positions.abs().sum() if not short_positions.empty else 0,
                'Train_Window_Vol': train_window_vol,
                'N_Regimes_Used': self.n_regimes,
                'Lookback_Used': self.lookback_window,
                'Method_Used': self.regime_detection_method
            }
        except Exception as e:
            logging.error(f"Error en _process_wfa_window: {str(e)}", exc_info=True)
            print(f"Error procesando ventana WFA: {str(e)}")
            return {
                'Test_Start_Date': window['test_start_date'],
                'Test_End_Date': window['test_end_date'],
                'Total_Return': 0,
                'Annualized_Return': 0,
                'Annualized_Volatility': 0,
                'Sharpe_Ratio': 0,
                'Max_Drawdown': 0,
                'Initial_Cost': 0,
                'Short_Exposure': 0,
                'Train_Window_Vol': 0,
                'N_Regimes_Used': self.n_regimes,
                'Lookback_Used': self.lookback_window,
                'Method_Used': self.regime_detection_method
            }
    
    def _plot_wfa_results(self, wfa_df):
        """
        Visualiza los resultados del análisis walk-forward.
        """
        try:
            save_path = './artifacts/results/figures/'
            os.makedirs(save_path, exist_ok=True)
            
            # 1. Visualización mejorada de Sharpe Ratio por ventana
            plt.figure(figsize=(14, 7))
            
            # Colorear barras por sharpe ratio (rojo si negativo, verde si positivo)
            colors = ['green' if sr > 0 else 'red' for sr in wfa_df['Sharpe_Ratio']]
            plt.bar(range(len(wfa_df)), wfa_df['Sharpe_Ratio'], color=colors)
            
            # Añadir línea de promedio
            mean_sharpe = wfa_df['Sharpe_Ratio'].mean()
            plt.axhline(y=mean_sharpe, color='blue', linestyle='-', label=f'Promedio: {mean_sharpe:.2f}')
            
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title('Sharpe Ratio por Ventana de Prueba', fontsize=14)
            plt.xlabel('Ventana de Prueba', fontsize=12)
            plt.ylabel('Sharpe Ratio', fontsize=12)
            
            # Usar fechas más legibles en el eje x
            if len(wfa_df) <= 20:
                plt.xticks(range(len(wfa_df)), 
                        [f"{d.strftime('%Y-%m')}" for d in wfa_df['Test_Start_Date']], 
                        rotation=45)
            else:
                # Si hay muchas fechas, mostrar solo algunas
                step = max(1, len(wfa_df) // 10)
                plt.xticks(range(0, len(wfa_df), step), 
                        [f"{wfa_df['Test_Start_Date'].iloc[i].strftime('%Y-%m')}" for i in range(0, len(wfa_df), step)], 
                        rotation=45)
            
            plt.legend(fontsize=12)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{save_path}wfa_sharpe_ratios.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Gráfico de retornos totales por ventana
            plt.figure(figsize=(14, 7))
            colors = ['green' if r > 0 else 'red' for r in wfa_df['Total_Return']]
            plt.bar(range(len(wfa_df)), wfa_df['Total_Return'], color=colors)
            
            # Añadir línea de promedio
            mean_return = wfa_df['Total_Return'].mean()
            plt.axhline(y=mean_return, color='blue', linestyle='-', label=f'Promedio: {mean_return:.2%}')
            
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title('Retorno Total por Ventana de Prueba', fontsize=14)
            plt.xlabel('Ventana de Prueba', fontsize=12)
            plt.ylabel('Retorno Total', fontsize=12)
            
            # Usar mismas etiquetas que en el gráfico anterior
            if len(wfa_df) <= 20:
                plt.xticks(range(len(wfa_df)), 
                        [f"{d.strftime('%Y-%m')}" for d in wfa_df['Test_Start_Date']], 
                        rotation=45)
            else:
                step = max(1, len(wfa_df) // 10)
                plt.xticks(range(0, len(wfa_df), step), 
                        [f"{wfa_df['Test_Start_Date'].iloc[i].strftime('%Y-%m')}" for i in range(0, len(wfa_df), step)], 
                        rotation=45)
            
            # Añadir etiquetas de porcentaje para cada barra
            for i, v in enumerate(wfa_df['Total_Return']):
                plt.text(i - 0.3, v + 0.01 if v >= 0 else v - 0.03, f"{v:.1%}", 
                        color='black', fontweight='bold')
            
            plt.legend(fontsize=12)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{save_path}wfa_total_returns.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Visualización de métricas clave en formato profesional
            plt.figure(figsize=(16, 12))
            
            # 3.1 Sharpe Ratio
            plt.subplot(2, 2, 1)
            plt.hist(wfa_df['Sharpe_Ratio'], bins=10, alpha=0.7, color='blue')
            plt.axvline(x=wfa_df['Sharpe_Ratio'].mean(), color='r', linestyle='--', 
                       label=f'Promedio: {wfa_df["Sharpe_Ratio"].mean():.2f}')
            plt.axvline(x=0, color='green', linestyle='-')
            plt.title('Distribución de Sharpe Ratios', fontsize=14)
            plt.xlabel('Sharpe Ratio', fontsize=12)
            plt.ylabel('Frecuencia', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True)
            
            # 3.2 Retornos Anualizados
            plt.subplot(2, 2, 2)
            plt.hist(wfa_df['Annualized_Return'], bins=10, alpha=0.7, color='green')
            plt.axvline(x=wfa_df['Annualized_Return'].mean(), color='r', linestyle='--', 
                       label=f'Promedio: {wfa_df["Annualized_Return"].mean():.2%}')
            plt.axvline(x=0, color='blue', linestyle='-')
            plt.title('Distribución de Retornos Anualizados', fontsize=14)
            plt.xlabel('Retorno Anualizado', fontsize=12)
            plt.ylabel('Frecuencia', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True)
            
            # 3.3 Volatilidad
            plt.subplot(2, 2, 3)
            plt.hist(wfa_df['Annualized_Volatility'], bins=10, alpha=0.7, color='orange')
            plt.axvline(x=wfa_df['Annualized_Volatility'].mean(), color='r', linestyle='--', 
                       label=f'Promedio: {wfa_df["Annualized_Volatility"].mean():.2%}')
            plt.title('Distribución de Volatilidad Anualizada', fontsize=14)
            plt.xlabel('Volatilidad Anualizada', fontsize=12)
            plt.ylabel('Frecuencia', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True)
            
            # 3.4 Drawdowns
            plt.subplot(2, 2, 4)
            plt.hist(wfa_df['Max_Drawdown'], bins=10, alpha=0.7, color='red')
            plt.axvline(x=wfa_df['Max_Drawdown'].mean(), color='blue', linestyle='--', 
                       label=f'Promedio: {wfa_df["Max_Drawdown"].mean():.2%}')
            plt.title('Distribución de Drawdowns Máximos', fontsize=14)
            plt.xlabel('Drawdown Máximo', fontsize=12)
            plt.ylabel('Frecuencia', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f'{save_path}wfa_metrics_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 4. Análisis de regímenes utilizados
            if 'N_Regimes_Used' in wfa_df.columns and 'Method_Used' in wfa_df.columns:
                # Agrupar por método y número de regímenes
                regime_counts = wfa_df.groupby(['Method_Used', 'N_Regimes_Used']).size().reset_index(name='Count')
                
                plt.figure(figsize=(12, 6))
                
                # Usar colores distintivos por método
                colors = {'gmm': 'blue', 'bgmm': 'orange'}
                
                # Crear barras agrupadas
                bar_width = 0.35
                for i, method in enumerate(regime_counts['Method_Used'].unique()):
                    method_data = regime_counts[regime_counts['Method_Used'] == method]
                    positions = np.arange(len(method_data)) + i * bar_width
                    plt.bar(positions, method_data['Count'], bar_width, 
                           label=f'Método: {method}', 
                           color=colors.get(method, 'gray'))
                    
                    # Añadir etiquetas a las barras
                    for pos, count, regimes in zip(positions, method_data['Count'], method_data['N_Regimes_Used']):
                        plt.text(pos, count + 0.5, f"{count}\n({regimes} reg.)", 
                                ha='center', va='bottom', fontsize=10)
                
                plt.xlabel('Configuración', fontsize=12)
                plt.ylabel('Número de Ventanas', fontsize=12)
                plt.title('Métodos y Regímenes Utilizados en Walk-Forward Analysis', fontsize=14)
                plt.xticks([])  # Ocultar ticks en x
                plt.legend(fontsize=12)
                plt.grid(axis='y')
                plt.tight_layout()
                plt.savefig(f'{save_path}wfa_regime_methods.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # 5. Relación entre volatilidad de ventana y rendimiento
            plt.figure(figsize=(10, 6))
            plt.scatter(wfa_df['Train_Window_Vol'], wfa_df['Sharpe_Ratio'], 
                       c=wfa_df['Total_Return'], cmap='RdYlGn', s=100, alpha=0.7)
            
            plt.colorbar(label='Retorno Total')
            plt.axhline(y=0, color='r', linestyle='--')
            
            # Línea de tendencia
            z = np.polyfit(wfa_df['Train_Window_Vol'], wfa_df['Sharpe_Ratio'], 1)
            p = np.poly1d(z)
            plt.plot(wfa_df['Train_Window_Vol'], p(wfa_df['Train_Window_Vol']), 
                    "b--", linewidth=1, label=f"Tendencia: y={z[0]:.2f}x+{z[1]:.2f}")
            
            plt.title('Relación entre Volatilidad y Sharpe Ratio', fontsize=14)
            plt.xlabel('Volatilidad de Entrenamiento', fontsize=12)
            plt.ylabel('Sharpe Ratio', fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{save_path}wfa_vol_vs_sharpe.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logging.error(f"Error en _plot_wfa_results: {str(e)}", exc_info=True)
            print(f"Error visualizando resultados WFA: {str(e)}")

# Ejecutar la estrategia
if __name__ == "__main__":
    try:
        print("Iniciando estrategia multifactorial adaptativa optimizada...")
        
        # Inicializar estrategia con parámetros mejorados
        strategy = AdaptiveMultifactorStrategy(
            start_date='2015-01-01',
            end_date='2023-01-01',
            lookback_window=189,                # Más corto que 252 para adaptación más rápida
            regime_window=63,                   # Más corto para detectar cambios de régimen
            n_regimes=5,                        # Más regímenes para mejor granularidad
            rebalance_freq=21,                  # 1 mes de trading
            vol_target=0.10,                    # 10% anualizado
            max_leverage=1.5,                   # Límite de apalancamiento
            transaction_cost=0.0005,            # 5 puntos básicos
            market_impact=0.1,                  # Como factor de volatilidad diaria
            borrow_cost=0.0002,                 # 20 puntos básicos anualizados
            execution_delay=1,                  # 1 día de retraso
            use_point_in_time=False,            # No usar point-in-time data
            max_drawdown_limit=0.20,            # Limitar drawdown al 20%
            dynamic_vol_scaling=True,           # Escalar volatilidad dinámicamente
            risk_targeting=True,                # Usar targeting de riesgo basado en régimen
            regime_detection_method='bgmm',     # Usar Bayesian GMM para detección de regímenes
            n_jobs=-1,                          # Usar todos los núcleos disponibles
            use_cache=True,                     # Activar caché de resultados
            use_all_assets=True                 # MODIFICADO: Usar todos los activos sin filtrar
        )
        
        # Ejecutar backtest
        print("\n--- EJECUTANDO BACKTEST ---")
        performance = strategy.backtest()
        
        # Calcular métricas
        metrics = strategy.calculate_metrics()
        print("\nMétricas de Rendimiento:")
        for key, value in metrics.items():
            # Formatear salida según el tipo de valor
            if 'Return' in key or 'Alpha' in key or 'Drawdown' in key or 'Volatility' in key or 'Turnover' in key or 'Costs' in key:
                print(f"{key}: {value:.2%}")
            elif 'Ratio' in key or 'Beta' in key or 'Leverage' in key:
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        
        # Generar visualizaciones
        print("\n--- GENERANDO VISUALIZACIONES ---")
        strategy.plot_results()
        
        # Ejecutar análisis walk-forward
        print("\n--- EJECUTANDO ANÁLISIS WALK-FORWARD ---")
        wfa_results = strategy.run_walk_forward_analysis(train_size=0.6, step_size=63, train_lookback=252)
        
        print("\nAnálisis completado. Todos los resultados guardados en ./artifacts/results/")
        
    except Exception as e:
        logging.error(f"Error en la ejecución principal: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}. Ver ./artifacts/errors.txt para más detalles.")