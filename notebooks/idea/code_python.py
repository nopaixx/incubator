
import os
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal, optimize
from scipy.fft import fft
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor, LinearRegression
from statsmodels.nonparametric.kernel_regression import KernelReg
from statsmodels.tsa.stattools import acf
import warnings
from datetime import datetime, timedelta
from tqdm import tqdm
import sqlite3
from joblib import Parallel, delayed
from functools import partial
import matplotlib.dates as mdates
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import information_coefficient

# Crear directorios para resultados
os.makedirs('./artifacts/results', exist_ok=True)
os.makedirs('./artifacts/results/figures', exist_ok=True)
os.makedirs('./artifacts/results/data', exist_ok=True)

# Configurar logging
logging.basicConfig(
    filename='./artifacts/errors.txt',
    level=logging.ERROR,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Ignorar advertencias
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

class AdaptiveMultifactorRankingSystem:
    def __init__(self, start_date='2010-01-01', end_date=None, db_path='./artifacts/results/data/market_data.db'):
        """
        Inicializa el sistema de ranking adaptativo multifactorial
        
        Args:
            start_date: Fecha de inicio para datos históricos
            end_date: Fecha de fin para datos históricos (None = hoy)
            db_path: Ruta a la base de datos SQLite
        """
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.db_path = db_path
        self.tickers = self._get_sp500_tickers()
        self.market_data = {}
        self.features = {}
        self.rankings = {}
        self.portfolio = {}
        self.performance = {}
        
        # Parámetros de la estrategia
        self.rebalance_frequency = 5  # días (semanal)
        self.recalibration_frequency = 21  # días (mensual)
        self.position_limit = 0.03  # 3% máximo por posición
        self.min_volume = 500000  # Volumen mínimo diario
        self.min_market_cap = 2e9  # Capitalización mínima ($2B)
        self.sector_limit = 0.25  # 25% máximo por sector
        
        # Inicializar base de datos
        self._init_database()
    
    def _get_sp500_tickers(self):
        """Obtiene los tickers actuales del S&P 500"""
        try:
            # En una implementación real, se usaría una fuente point-in-time
            # Para simplificar, usamos los componentes actuales
            sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(sp500_url)
            sp500_table = tables[0]
            tickers = sp500_table['Symbol'].str.replace('.', '-').tolist()
            
            # Para este ejemplo, limitamos a 50 tickers para reducir tiempo de ejecución
            return tickers[:50]
        except Exception as e:
            logging.error(f"Error obteniendo tickers del S&P 500: {str(e)}")
            # Fallback a algunos tickers principales
            return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V']
    
    def _init_database(self):
        """Inicializa la base de datos SQLite para almacenar datos históricos"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Crear tabla para datos de precios
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_data (
                ticker TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (ticker, date)
            )
            ''')
            
            # Crear tabla para features calculadas
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS features (
                ticker TEXT,
                date TEXT,
                feature_name TEXT,
                value REAL,
                PRIMARY KEY (ticker, date, feature_name)
            )
            ''')
            
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"Error inicializando base de datos: {str(e)}")
    
    def fetch_market_data(self, force_update=False):
        """
        Obtiene datos de mercado para todos los tickers
        
        Args:
            force_update: Si es True, fuerza la actualización desde la fuente
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Verificar datos existentes en la base de datos
            if not force_update:
                for ticker in tqdm(self.tickers, desc="Verificando datos existentes"):
                    query = f"SELECT date, open, high, low, close, volume FROM price_data WHERE ticker = '{ticker}' ORDER BY date"
                    df = pd.read_sql_query(query, conn)
                    
                    if not df.empty:
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                        self.market_data[ticker] = df
            
            # Obtener datos faltantes
            missing_tickers = [t for t in self.tickers if t not in self.market_data]
            if missing_tickers or force_update:
                tickers_to_fetch = self.tickers if force_update else missing_tickers
                
                for ticker in tqdm(tickers_to_fetch, desc="Descargando datos de mercado"):
                    try:
                        # Obtener datos de yfinance
                        data = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
                        
                        if data.empty:
                            continue
                        
                        # Guardar en la base de datos
                        data_to_save = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                        data_to_save.columns = ['open', 'high', 'low', 'close', 'volume']
                        data_to_save.index.name = 'date'
                        data_to_save.reset_index(inplace=True)
                        data_to_save['ticker'] = ticker
                        
                        data_to_save.to_sql('price_data', conn, if_exists='replace', index=False)
                        
                        # Guardar en memoria
                        self.market_data[ticker] = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                        self.market_data[ticker].columns = ['open', 'high', 'low', 'close', 'volume']
                    except Exception as e:
                        logging.error(f"Error descargando datos para {ticker}: {str(e)}")
            
            conn.close()
            
            # Obtener datos del índice S&P 500
            if 'SPY' not in self.market_data:
                spy_data = yf.download('SPY', start=self.start_date, end=self.end_date, progress=False)
                self.market_data['SPY'] = spy_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                self.market_data['SPY'].columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Obtener datos del VIX
            if '^VIX' not in self.market_data:
                vix_data = yf.download('^VIX', start=self.start_date, end=self.end_date, progress=False)
                self.market_data['^VIX'] = vix_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                self.market_data['^VIX'].columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Calcular retornos diarios para todos los tickers
            for ticker in self.market_data:
                self.market_data[ticker]['returns'] = self.market_data[ticker]['close'].pct_change()
                self.market_data[ticker]['log_returns'] = np.log(self.market_data[ticker]['close']).diff()
                self.market_data[ticker]['true_range'] = np.maximum(
                    self.market_data[ticker]['high'] - self.market_data[ticker]['low'],
                    np.maximum(
                        abs(self.market_data[ticker]['high'] - self.market_data[ticker]['close'].shift(1)),
                        abs(self.market_data[ticker]['low'] - self.market_data[ticker]['close'].shift(1))
                    )
                )
            
            print(f"Datos de mercado obtenidos para {len(self.market_data)} tickers")
        except Exception as e:
            logging.error(f"Error en fetch_market_data: {str(e)}")
            raise
    
    def _calculate_dominant_cycle(self, ticker, window=252):
        """
        Calcula el ciclo dominante específico para un ticker
        
        Args:
            ticker: Símbolo del ticker
            window: Ventana para análisis espectral
            
        Returns:
            Período del ciclo dominante en días
        """
        try:
            if ticker not in self.market_data:
                return 21  # valor por defecto
            
            prices = self.market_data[ticker]['close'].dropna()
            if len(prices) < window:
                return 21
            
            # Aplicar ventana deslizante
            cycles = []
            
            for i in range(len(prices) - window + 1):
                segment = prices.iloc[i:i+window].values
                segment = segment - np.mean(segment)  # Eliminar tendencia
                
                # Aplicar ventana Hamming para reducir leakage espectral
                windowed = segment * np.hamming(len(segment))
                
                # Calcular FFT
                spectrum = np.abs(fft(windowed))
                
                # Solo considerar la primera mitad (frecuencias positivas)
                half_spectrum = spectrum[1:window//2]
                frequencies = np.fft.fftfreq(window, 1)[1:window//2]
                
                # Encontrar frecuencia dominante
                if len(half_spectrum) > 0:
                    dominant_idx = np.argmax(half_spectrum)
                    dominant_freq = frequencies[dominant_idx]
                    if dominant_freq > 0:
                        dominant_period = abs(1.0 / dominant_freq)
                        cycles.append(min(max(dominant_period, 5), 252))  # Limitar entre 5 y 252 días
            
            if not cycles:
                return 21
            
            # Usar mediana para robustez
            dominant_cycle = int(np.median(cycles))
            return dominant_cycle
        except Exception as e:
            logging.error(f"Error calculando ciclo dominante para {ticker}: {str(e)}")
            return 21  # valor por defecto
    
    def _calculate_hurst_exponent(self, ticker, min_scale=5, max_scale=100):
        """
        Calcula el exponente de Hurst adaptativo para un ticker
        
        Args:
            ticker: Símbolo del ticker
            min_scale: Escala mínima para análisis R/S
            max_scale: Escala máxima para análisis R/S
            
        Returns:
            Exponente de Hurst
        """
        try:
            if ticker not in self.market_data:
                return 0.5  # valor por defecto (paseo aleatorio)
            
            returns = self.market_data[ticker]['log_returns'].dropna()
            if len(returns) < max_scale:
                return 0.5
            
            # Escalas para análisis R/S
            scales = np.logspace(np.log10(min_scale), np.log10(max_scale), num=4).astype(int)
            scales = np.unique(scales)
            
            rs_values = []
            
            for scale in scales:
                # Dividir la serie en segmentos
                segments = len(returns) // scale
                if segments == 0:
                    continue
                
                rs_segment = []
                
                for i in range(segments):
                    segment = returns.iloc[i*scale:(i+1)*scale].values
                    
                    # Calcular perfil acumulativo
                    profile = segment.cumsum() - segment.mean()
                    
                    # Rango rescalado
                    r = np.max(profile) - np.min(profile)
                    s = np.std(segment)
                    
                    if s > 0:
                        rs_segment.append(r/s)
                
                if rs_segment:
                    rs_values.append((scale, np.mean(rs_segment)))
            
            if len(rs_values) < 2:
                return 0.5
            
            # Regresión log-log
            x = np.log10([t[0] for t in rs_values])
            y = np.log10([t[1] for t in rs_values])
            
            slope, _, _, _, _ = stats.linregress(x, y)
            
            return slope
        except Exception as e:
            logging.error(f"Error calculando exponente de Hurst para {ticker}: {str(e)}")
            return 0.5
    
    def _calculate_lempel_ziv_complexity(self, ticker, window=126):
        """
        Calcula la complejidad de Lempel-Ziv para un ticker
        
        Args:
            ticker: Símbolo del ticker
            window: Ventana para el cálculo
            
        Returns:
            Valor de complejidad normalizado
        """
        try:
            if ticker not in self.market_data:
                return 0.5  # valor por defecto
            
            returns = self.market_data[ticker]['returns'].dropna().iloc[-window:]
            if len(returns) < window:
                return 0.5
            
            # Calcular volatilidad para umbral adaptativo
            volatility = returns.std()
            threshold = 0.5 * volatility
            
            # Convertir retornos a secuencia binaria
            binary_seq = ''.join(['1' if r > threshold else '0' if r < -threshold else '1' if np.random.random() > 0.5 else '0' for r in returns])
            
            # Calcular complejidad de Lempel-Ziv
            sub_strings = set()
            i, complexity = 0, 1
            
            while i < len(binary_seq):
                sub_str = binary_seq[i]
                i += 1
                while i < len(binary_seq) and sub_str + binary_seq[i] in sub_strings:
                    sub_str += binary_seq[i]
                    i += 1
                
                if i < len(binary_seq):
                    sub_str += binary_seq[i]
                    sub_strings.add(sub_str)
                    complexity += 1
            
            # Normalizar por la complejidad máxima teórica
            max_complexity = len(binary_seq) / np.log2(len(binary_seq))
            normalized_complexity = complexity / max_complexity
            
            return normalized_complexity
        except Exception as e:
            logging.error(f"Error calculando complejidad LZ para {ticker}: {str(e)}")
            return 0.5
    
    def _calculate_sample_entropy(self, ticker, scales=[5, 10, 20], m=2, r=0.2):
        """
        Calcula la entropía de muestra multiescala para un ticker
        
        Args:
            ticker: Símbolo del ticker
            scales: Escalas temporales para el cálculo
            m: Dimensión de embedding
            r: Tolerancia
            
        Returns:
            Entropía de muestra promedio ponderada
        """
        try:
            if ticker not in self.market_data:
                return 0.5  # valor por defecto
            
            returns = self.market_data[ticker]['returns'].dropna().iloc[-252:]
            if len(returns) < 50:
                return 0.5
            
            # Función para calcular entropía de muestra en una escala
            def sample_entropy(data, m, r):
                N = len(data)
                if N < 2*m+1:
                    return 0
                
                # Normalizar r por la desviación estándar
                r = r * np.std(data)
                
                # Contar coincidencias para dimensiones m y m+1
                count_m = 0
                count_m1 = 0
                
                for i in range(N-m):
                    template_m = data[i:i+m]
                    template_m1 = data[i:i+m+1]
                    
                    # Contar coincidencias para dimensión m
                    for j in range(i+1, N-m+1):
                        if np.max(np.abs(template_m - data[j:j+m])) < r:
                            count_m += 1
                            
                            # Verificar si también coincide para dimensión m+1
                            if j <= N-m-1 and np.max(np.abs(template_m1 - data[j:j+m+1])) < r:
                                count_m1 += 1
                
                # Calcular probabilidades
                if count_m == 0 or count_m1 == 0:
                    return 0
                
                return -np.log(count_m1 / count_m)
            
            # Calcular entropía para cada escala
            entropies = []
            weights = [0.2, 0.3, 0.5]  # Pesos para cada escala (mayor peso a escalas mayores)
            
            for i, scale in enumerate(scales):
                # Coarse-graining (promedio en ventanas)
                coarse_data = np.array([np.mean(returns.iloc[j:j+scale]) for j in range(0, len(returns)-scale+1, scale)])
                
                if len(coarse_data) > 2*m+1:
                    entropy = sample_entropy(coarse_data, m, r)
                    entropies.append(entropy * weights[i])
            
            if not entropies:
                return 0.5
            
            return sum(entropies) / sum(weights[:len(entropies)])
        except Exception as e:
            logging.error(f"Error calculando entropía de muestra para {ticker}: {str(e)}")
            return 0.5
    
    def generate_features(self):
        """Genera todas las características adaptativas para cada ticker"""
        try:
            # Inicializar diccionario de características
            self.features = {ticker: {} for ticker in self.market_data}
            
            # Procesar cada ticker en paralelo
            results = Parallel(n_jobs=-1)(
                delayed(self._generate_ticker_features)(ticker) 
                for ticker in tqdm(self.market_data.keys(), desc="Generando características")
            )
            
            # Consolidar resultados
            for ticker, features_df in results:
                if features_df is not None:
                    self.features[ticker] = features_df
            
            print(f"Características generadas para {len(self.features)} tickers")
        except Exception as e:
            logging.error(f"Error en generate_features: {str(e)}")
            raise
    
    def _generate_ticker_features(self, ticker):
        """
        Genera todas las características para un ticker específico
        
        Args:
            ticker: Símbolo del ticker
            
        Returns:
            Tuple (ticker, DataFrame con características)
        """
        try:
            if ticker not in self.market_data or self.market_data[ticker].empty:
                return ticker, None
            
            # Obtener datos del ticker
            data = self.market_data[ticker].copy()
            
            # Inicializar DataFrame para características
            features = pd.DataFrame(index=data.index)
            
            # 1. Características de Dinámica Temporal (CDT)
            
            # 1.1 Ciclo Dominante Específico (CDE)
            dominant_cycle = self._calculate_dominant_cycle(ticker)
            features['CDE'] = dominant_cycle
            
            # 1.2 Oscilador de Momentum Adaptativo (OMA)
            if len(data) > dominant_cycle:
                features['OMA'] = (data['close'] - data['close'].shift(dominant_cycle)) / data['close'].rolling(dominant_cycle).std()
                
                # 1.3 Aceleración del Precio Adaptativa (APA)
                features['APA'] = (features['OMA'] - features['OMA'].shift(5)) / features['OMA'].rolling(63).std()
                features['APA'] = features['APA'].ewm(span=max(5, dominant_cycle//10)).mean()  # Filtrado
            
            # 2. Características de Eficiencia y Persistencia (CEP)
            
            # 2.1 Exponente de Hurst Adaptativo (EHA)
            # Calculamos en ventanas rodantes
            rolling_hurst = []
            for i in range(max(252, len(data)-504), len(data), 5):  # Actualización semanal
                window = data.iloc[max(0, i-504):i]  # 2 años de datos
                if len(window) > 100:
                    h = self._calculate_hurst_exponent(ticker)
                    rolling_hurst.extend([h] * min(5, len(data)-i))
            
            if rolling_hurst:
                features['EHA'] = pd.Series(rolling_hurst, index=data.index[-len(rolling_hurst):])
            
            # 2.2 Ratio de Eficiencia de Movimiento (REM)
            n = max(21, dominant_cycle//2)
            if len(data) > n:
                features['REM'] = abs(np.log(data['close'] / data['close'].shift(n))) / data['true_range'].rolling(n).sum()
            
            # 2.3 Índice de Persistencia Direccional (IPD)
            if len(data) > dominant_cycle:
                signed_returns = np.sign(data['returns']) * np.abs(data['returns'])**0.5
                features['IPD'] = signed_returns.ewm(span=max(5, dominant_cycle//4)).mean()
                
                # Normalización a percentiles
                if len(features['IPD'].dropna()) > 252:
                    rolling_percentile = []
                    for i in range(252, len(features['IPD'])):
                        window = features['IPD'].iloc[max(0, i-504):i]
                        if not window.empty:
                            perc = stats.percentileofscore(window.dropna(), features['IPD'].iloc[i]) / 100
                            rolling_percentile.append(perc)
                    
                    if rolling_percentile:
                        features['IPD_norm'] = pd.Series(rolling_percentile, index=features.index[-len(rolling_percentile):])
            
            # 3. Características de Microestructura (CME)
            
            # 3.1 Asimetría de Volumen-Precio (AVP)
            if len(data) > 63:
                abs_returns = abs(data['returns'])
                norm_volume = np.log(data['volume'] / data['volume'].rolling(63).mean())
                
                # Correlación de Spearman en ventanas rodantes
                rolling_corr = []
                for i in range(21, len(data)):
                    window_returns = abs_returns.iloc[i-21:i]
                    window_volume = norm_volume.iloc[i-21:i]
                    valid_data = ~(window_returns.isna() | window_volume.isna())
                    if valid_data.sum() > 10:
                        corr = stats.spearmanr(window_returns[valid_data], window_volume[valid_data])[0]
                        rolling_corr.append(corr if not np.isnan(corr) else 0)
                    else:
                        rolling_corr.append(0)
                
                if rolling_corr:
                    features['AVP'] = pd.Series(rolling_corr, index=data.index[-len(rolling_corr):])
                    features['AVP'] = features['AVP'].ewm(span=5).mean()
            
            # 3.2 Elasticidad de Liquidez Estimada (ELE)
            window_size = min(63, 3*dominant_cycle)
            if len(data) > window_size:
                # Usar regresión Huber para robustez
                rolling_elasticity = []
                for i in range(window_size, len(data)):
                    window_returns = abs(data['returns'].iloc[i-window_size:i]).values.reshape(-1, 1)
                    window_volume = norm_volume.iloc[i-window_size:i].values.reshape(-1, 1)
                    valid_data = ~(np.isnan(window_returns).any(axis=1) | np.isnan(window_volume).any(axis=1))
                    
                    if valid_data.sum() > window_size//2:
                        try:
                            huber = HuberRegressor(epsilon=1.35)
                            huber.fit(window_volume[valid_data], window_returns[valid_data])
                            elasticity = huber.coef_[0]
                            rolling_elasticity.append(elasticity)
                        except:
                            rolling_elasticity.append(0)
                    else:
                        rolling_elasticity.append(0)
                
                if rolling_elasticity:
                    features['ELE'] = pd.Series(rolling_elasticity, index=data.index[-len(rolling_elasticity):])
            
            # 3.3 Indicador de Presión de Compra-Venta (IPCV)
            window_size = min(21, dominant_cycle//2)
            if len(data) > window_size:
                price_direction = np.sign(data['close'] - data['open'])
                volume_pressure = data['volume'] * price_direction
                features['IPCV'] = volume_pressure.rolling(window_size).sum() / data['volume'].rolling(window_size).sum()
                
                # Normalización Z-score
                if len(features['IPCV'].dropna()) > 252:
                    features['IPCV_norm'] = (features['IPCV'] - features['IPCV'].rolling(252).mean()) / features['IPCV'].rolling(252).std()
            
            # 4. Características de Complejidad y Entropía (CCE)
            
            # 4.1 Entropía de Muestra Multiescala (EMM)
            # Calculamos en ventanas rodantes para reducir carga computacional
            rolling_entropy = []
            for i in range(max(252, len(data)-504), len(data), 5):  # Actualización semanal
                window = data.iloc[max(0, i-252):i]  # 1 año de datos
                if len(window) > 50:
                    entropy = self._calculate_sample_entropy(ticker)
                    rolling_entropy.extend([entropy] * min(5, len(data)-i))
            
            if rolling_entropy:
                features['EMM'] = pd.Series(rolling_entropy, index=data.index[-len(rolling_entropy):])
            
            # 4.2 Complejidad de Lempel-Ziv (CLZ)
            rolling_complexity = []
            for i in range(max(252, len(data)-504), len(data), 5):  # Actualización semanal
                window = data.iloc[max(0, i-252):i]  # 1 año de datos
                if len(window) > 126:
                    complexity = self._calculate_lempel_ziv_complexity(ticker)
                    rolling_complexity.extend([complexity] * min(5, len(data)-i))
            
            if rolling_complexity:
                features['CLZ'] = pd.Series(rolling_complexity, index=data.index[-len(rolling_complexity):])
            
            # 4.3 Índice de Transferencia de Información (ITI)
            # Simplificamos usando correlación con lag como proxy
            if 'SPY' in self.market_data and len(data) > 21:
                spy_returns = self.market_data['SPY']['returns']
                
                # Alinear índices
                common_index = data.index.intersection(spy_returns.index)
                if len(common_index) > 21:
                    ticker_returns = data.loc[common_index, 'returns']
                    spy_returns = spy_returns.loc[common_index]
                    
                    # Calcular correlaciones con diferentes lags
                    lags = [1, 5, 10, 21]
                    weights = [0.1, 0.2, 0.3, 0.4]  # Mayor peso a horizontes más largos
                    
                    iti_values = []
                    for i in range(max(lags), len(common_index)):
                        iti_lag = 0
                        for lag, weight in zip(lags, weights):
                            corr_lag0 = np.corrcoef(ticker_returns.iloc[i-lag:i], spy_returns.iloc[i-lag:i])[0, 1]
                            corr_lag1 = np.corrcoef(ticker_returns.iloc[i-lag:i], spy_returns.iloc[i-lag-1:i-1])[0, 1]
                            iti_lag += weight * (corr_lag1 - corr_lag0)
                        
                        iti_values.append(iti_lag)
                    
                    if iti_values:
                        features['ITI'] = pd.Series(iti_values, index=common_index[-len(iti_values):])
            
            # 5. Características de Correlación Dinámica (CCD)
            
            # 5.1 Beta Condicional Multirégimen (BCM)
            if 'SPY' in self.market_data and '^VIX' in self.market_data and len(data) > 63:
                # Alinear índices
                common_index = data.index.intersection(self.market_data['SPY'].index).intersection(self.market_data['^VIX'].index)
                
                if len(common_index) > 63:
                    ticker_returns = data.loc[common_index, 'returns']
                    spy_returns = self.market_data['SPY'].loc[common_index, 'returns']
                    vix_values = self.market_data['^VIX'].loc[common_index, 'close']
                    
                    # Definir regímenes basados en VIX
                    vix_quantiles = vix_values.rolling(252).quantile([0.33, 0.67])
                    
                    # Calcular beta condicional
                    rolling_beta = []
                    for i in range(63, len(common_index)):
                        window_ticker = ticker_returns.iloc[i-63:i]
                        window_spy = spy_returns.iloc[i-63:i]
                        current_vix = vix_values.iloc[i]
                        
                        # Determinar régimen
                        if pd.notna(vix_quantiles.iloc[i-1, 0]) and pd.notna(vix_quantiles.iloc[i-1, 1]):
                            if current_vix <= vix_quantiles.iloc[i-1, 0]:
                                regime = 'low'
                                weights = np.ones(63) * 0.8  # Menor peso en régimen de baja volatilidad
                            elif current_vix <= vix_quantiles.iloc[i-1, 1]:
                                regime = 'medium'
                                weights = np.ones(63)  # Peso normal en régimen medio
                            else:
                                regime = 'high'
                                weights = np.ones(63) * 1.2  # Mayor peso en régimen de alta volatilidad
                        else:
                            regime = 'medium'
                            weights = np.ones(63)
                        
                        # Regresión ponderada
                        valid_data = ~(window_ticker.isna() | window_spy.isna())
                        if valid_data.sum() > 30:
                            X = sm.add_constant(window_spy[valid_data].values)
                            model = sm.WLS(window_ticker[valid_data].values, X, weights=weights[valid_data])
                            results = model.fit()
                            beta = results.params[1]
                            rolling_beta.append(beta)
                        else:
                            rolling_beta.append(np.nan)
                    
                    if rolling_beta:
                        features['BCM'] = pd.Series(rolling_beta, index=common_index[-len(rolling_beta):])
            
            # 5.2 Divergencia de Correlación Sectorial (DCS)
            # Simplificamos usando correlación con SPY como proxy del sector
            if 'SPY' in self.market_data and len(data) > 252:
                # Alinear índices
                common_index = data.index.intersection(self.market_data['SPY'].index)
                
                if len(common_index) > 252:
                    ticker_returns = data.loc[common_index, 'returns']
                    spy_returns = self.market_data['SPY'].loc[common_index, 'returns']
                    
                    # Calcular correlación rodante
                    rolling_corr = ticker_returns.rolling(21).corr(spy_returns)
                    rolling_corr_mean = rolling_corr.rolling(252).mean()
                    rolling_corr_std = rolling_corr.rolling(252).std()
                    
                    # Divergencia normalizada
                    features['DCS'] = (rolling_corr - rolling_corr_mean) / rolling_corr_std
            
            # 6. Características de Régimen Adaptativo (CRA)
            
            # 6.1 Indicador de Régimen de Volatilidad Específico (IRVE)
            if len(data) > 252:
                vol_21d = data['returns'].rolling(21).std() * np.sqrt(252)  # Anualizada
                vol_25p = vol_21d.rolling(252).quantile(0.25)
                vol_75p = vol_21d.rolling(252).quantile(0.75)
                
                features['IRVE'] = (vol_21d - vol_25p) / (vol_75p - vol_25p)
            
            # 6.2 Indicador de Cambio de Tendencia (ICT)
            if len(data) > dominant_cycle:
                # Osciladores en múltiples escalas
                scales = [max(5, dominant_cycle//4), max(10, dominant_cycle//2), dominant_cycle]
                oscillators = {}
                
                for scale in scales:
                    if len(data) > scale:
                        oscillators[scale] = (data['close'] - data['close'].rolling(scale).mean()) / data['close'].rolling(scale).std()
                
                # Calcular divergencias
                if len(oscillators) > 1:
                    scales_list = sorted(oscillators.keys())
                    divergence = pd.Series(0, index=data.index)
                    
                    for i in range(1, len(scales_list)):
                        scale_current = scales_list[i]
                        scale_prev = scales_list[i-1]
                        
                        if scale_current in oscillators and scale_prev in oscillators:
                            sign_diff = np.sign(oscillators[scale_current] - oscillators[scale_prev])
                            divergence += sign_diff
                    
                    features['ICT'] = divergence / (len(scales_list) - 1)
            
            # Limpiar NaN y valores infinitos
            features = features.replace([np.inf, -np.inf], np.nan)
            
            return ticker, features
        except Exception as e:
            logging.error(f"Error generando características para {ticker}: {str(e)}")
            return ticker, None
    
    def normalize_features(self):
        """Normaliza todas las características para cada ticker"""
        try:
            for ticker in tqdm(self.features.keys(), desc="Normalizando características"):
                if ticker not in self.features or self.features[ticker].empty:
                    continue
                
                features_df = self.features[ticker].copy()
                
                # Identificar columnas numéricas
                numeric_cols = features_df.select_dtypes(include=[np.number]).columns
                
                for col in numeric_cols:
                    # Winsorización adaptativa
                    if features_df[col].count() > 100:
                        lower = features_df[col].quantile(0.01)
                        upper = features_df[col].quantile(0.99)
                        features_df[col] = features_df[col].clip(lower, upper)
                    
                    # Normalización a rango [0,1] usando CDF empírica
                    if features_df[col].count() > 10:
                        sorted_values = features_df[col].dropna().sort_values()
                        ranks = sorted_values.rank(method='average')
                        normalized = (ranks - 1) / (len(ranks) - 1) if len(ranks) > 1 else ranks
                        
                        # Mapear valores originales a normalizados
                        value_to_norm = dict(zip(sorted_values, normalized))
                        features_df[col] = features_df[col].map(value_to_norm)
                
                self.features[ticker] = features_df
            
            print("Características normalizadas")
        except Exception as e:
            logging.error(f"Error en normalize_features: {str(e)}")
            raise
    
    def reduce_dimensions(self):
        """Reduce dimensionalidad de características usando ICA"""
        try:
            # Crear DataFrame consolidado con todas las características
            all_features = {}
            
            for ticker in self.features:
                if ticker in self.features and not self.features[ticker].empty:
                    # Usar las últimas 252 observaciones para cada ticker
                    ticker_features = self.features[ticker].iloc[-252:].copy()
                    
                    # Añadir identificador de ticker
                    ticker_features['ticker'] = ticker
                    
                    all_features[ticker] = ticker_features
            
            if not all_features:
                print("No hay suficientes datos para reducción de dimensionalidad")
                return
            
            # Concatenar todos los DataFrames
            combined_features = pd.concat(all_features.values())
            
            # Identificar columnas numéricas con suficientes datos
            numeric_cols = combined_features.select_dtypes(include=[np.number]).columns
            valid_cols = [col for col in numeric_cols if combined_features[col].count() > combined_features.shape[0] * 0.5]
            
            if len(valid_cols) < 3:
                print("No hay suficientes características válidas para reducción de dimensionalidad")
                return
            
            # Imputar valores faltantes con la media
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(combined_features[valid_cols])
            
            # Estandarizar datos
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Aplicar ICA
            n_components = min(len(valid_cols), 10)  # Máximo 10 componentes
            ica = FastICA(n_components=n_components, random_state=42)
            X_ica = ica.fit_transform(X_scaled)
            
            # Crear DataFrame con componentes
            ica_cols = [f'ICA_{i+1}' for i in range(n_components)]
            ica_df = pd.DataFrame(X_ica, columns=ica_cols, index=combined_features.index)
            
            # Añadir ticker
            ica_df['ticker'] = combined_features['ticker']
            
            # Separar por ticker y guardar
            for ticker in self.features:
                if ticker in ica_df['ticker'].values:
                    ticker_ica = ica_df[ica_df['ticker'] == ticker].drop(columns=['ticker'])
                    
                    # Añadir componentes ICA a características originales
                    self.features[ticker] = pd.concat([self.features[ticker], ticker_ica], axis=1)
            
            print(f"Dimensionalidad reducida a {n_components} componentes")
        except Exception as e:
            logging.error(f"Error en reduce_dimensions: {str(e)}")
            print(f"Error en reducción de dimensionalidad: {str(e)}")
    
    def calculate_rankings(self):
        """Calcula rankings para todos los tickers en diferentes horizontes temporales"""
        try:
            # Definir horizontes temporales
            horizons = {
                'short': 10,   # 10 días (2 semanas)
                'medium': 21,  # 21 días (1 mes)
                'long': 63     # 63 días (3 meses)
            }
            
            # Inicializar rankings
            self.rankings = {
                horizon: pd.DataFrame() 
                for horizon in horizons.keys()
            }
            
            # Para cada horizonte, calcular IC (Information Coefficient) de cada característica
            for horizon_name, horizon_days in horizons.items():
                print(f"Calculando rankings para horizonte {horizon_name} ({horizon_days} días)")
                
                # Recopilar datos de rendimiento futuro para todos los tickers
                future_returns = {}
                
                for ticker in self.features:
                    if ticker in self.market_data and not self.market_data[ticker].empty:
                        # Calcular rendimiento futuro
                        returns = self.market_data[ticker]['returns']
                        future_return = returns.shift(-horizon_days).rolling(horizon_days).apply(
                            lambda x: (1 + x).prod() - 1, raw=True
                        )
                        
                        future_returns[ticker] = future_return
                
                # Calcular IC para cada característica
                feature_ic = {}
                common_features = set()
                
                # Identificar características comunes
                for ticker in self.features:
                    if ticker in self.features and not self.features[ticker].empty:
                        common_features.update(self.features[ticker].columns)
                
                common_features = list(common_features)
                
                for feature in common_features:
                    feature_values = {}
                    feature_returns = {}
                    
                    # Recopilar valores de característica y rendimientos futuros
                    for ticker in self.features:
                        if (ticker in self.features and not self.features[ticker].empty and 
                            feature in self.features[ticker].columns and 
                            ticker in future_returns):
                            
                            feature_values[ticker] = self.features[ticker][feature]
                            feature_returns[ticker] = future_returns[ticker]
                    
                    # Calcular IC en ventanas rodantes
                    ic_values = []
                    dates = []
                    
                    # Obtener fechas comunes
                    all_dates = set()
                    for ticker in feature_values:
                        all_dates.update(feature_values[ticker].index)
                    
                    all_dates = sorted(all_dates)
                    
                    # Calcular IC para cada fecha
                    for date in all_dates[-252:]:  # Últimos 252 días
                        date_values = []
                        date_returns = []
                        
                        for ticker in feature_values:
                            if (date in feature_values[ticker].index and 
                                date in feature_returns[ticker].index and
                                pd.notna(feature_values[ticker].loc[date]) and
                                pd.notna(feature_returns[ticker].loc[date])):
                                
                                date_values.append(feature_values[ticker].loc[date])
                                date_returns.append(feature_returns[ticker].loc[date])
                        
                        if len(date_values) > 10:  # Mínimo 10 tickers
                            ic = np.corrcoef(date_values, date_returns)[0, 1]
                            if not np.isnan(ic):
                                ic_values.append(ic)
                                dates.append(date)
                    
                    if ic_values:
                        # Calcular IC promedio
                        avg_ic = np.mean(ic_values)
                        feature_ic[feature] = avg_ic
                
                # Filtrar características con IC significativo
                significant_features = {f: ic for f, ic in feature_ic.items() if abs(ic) > 0.05}
                
                if not significant_features:
                    print(f"No se encontraron características significativas para horizonte {horizon_name}")
                    continue
                
                # Normalizar pesos por magnitud de IC
                total_abs_ic = sum(abs(ic) for ic in significant_features.values())
                feature_weights = {f: abs(ic)/total_abs_ic for f, ic in significant_features.items()}
                
                # Calcular score combinado para cada ticker
                scores = {}
                
                for ticker in self.features:
                    if ticker in self.features and not self.features[ticker].empty:
                        ticker_scores = pd.Series(0, index=self.features[ticker].index)
                        
                        for feature, weight in feature_weights.items():
                            if feature in self.features[ticker].columns:
                                # Ajustar signo según correlación con rendimientos
                                sign = np.sign(feature_ic.get(feature, 0))
                                if sign != 0:
                                    ticker_scores += sign * self.features[ticker][feature] * weight
                        
                        scores[ticker] = ticker_scores
                
                # Crear DataFrame con todos los scores
                all_scores = pd.DataFrame(scores)
                
                # Calcular ranking percentil para cada fecha
                ranking_percentile = pd.DataFrame(index=all_scores.index)
                
                for date in all_scores.index:
                    date_scores = all_scores.loc[date].dropna()
                    
                    if len(date_scores) > 0:
                        # Calcular percentiles (0-100)
                        percentiles = date_scores.rank(pct=True) * 100
                        ranking_percentile.loc[date, percentiles.index] = percentiles
                
                # Guardar ranking
                self.rankings[horizon_name] = ranking_percentile
                
                print(f"Ranking calculado para horizonte {horizon_name} con {len(significant_features)} características")
                print(f"Top 5 características por IC: {sorted(significant_features.items(), key=lambda x: abs(x[1]), reverse=True)[:5]}")
            
            # Combinar rankings de diferentes horizontes
            self.combined_ranking = pd.DataFrame()
            
            # Pesos para cada horizonte
            horizon_weights = {
                'short': 0.3,
                'medium': 0.4,
                'long': 0.3
            }
            
            # Obtener todas las fechas y tickers
            all_dates = set()
            all_tickers = set()
            
            for horizon in self.rankings:
                all_dates.update(self.rankings[horizon].index)
                all_tickers.update(self.rankings[horizon].columns)
            
            all_dates = sorted(all_dates)
            
            # Inicializar DataFrame combinado
            self.combined_ranking = pd.DataFrame(index=all_dates, columns=list(all_tickers))
            
            # Combinar rankings con pesos
            for date in all_dates:
                for ticker in all_tickers:
                    weighted_rank = 0
                    total_weight = 0
                    
                    for horizon, weight in horizon_weights.items():
                        if (horizon in self.rankings and 
                            date in self.rankings[horizon].index and 
                            ticker in self.rankings[horizon].columns and
                            pd.notna(self.rankings[horizon].loc[date, ticker])):
                            
                            weighted_rank += self.rankings[horizon].loc[date, ticker] * weight
                            total_weight += weight
                    
                    if total_weight > 0:
                        self.combined_ranking.loc[date, ticker] = weighted_rank / total_weight
            
            print("Rankings combinados calculados")
        except Exception as e:
            logging.error(f"Error en calculate_rankings: {str(e)}")
            raise
    
    def generate_portfolio(self, strategy_type='long_only'):
        """
        Genera portafolio basado en rankings
        
        Args:
            strategy_type: 'long_only' o 'market_neutral'
        """
        try:
            if self.combined_ranking.empty:
                print("No hay rankings disponibles para generar portafolio")
                return
            
            # Inicializar DataFrame de portafolio
            self.portfolio = pd.DataFrame(index=self.combined_ranking.index)
            self.portfolio['cash'] = 1.0  # Iniciar con 100% en efectivo
            
            # Parámetros de portafolio
            n_positions = 20  # Número de posiciones
            rebalance_days = list(range(0, len(self.portfolio), self.rebalance_frequency))
            
            if not rebalance_days:
                print("No hay suficientes días para rebalanceo")
                return
            
            # Añadir último día si no está incluido
            if rebalance_days[-1] != len(self.portfolio) - 1:
                rebalance_days.append(len(self.portfolio) - 1)
            
            # Inicializar posiciones
            for ticker in self.combined_ranking.columns:
                self.portfolio[ticker] = 0.0
            
            # Generar portafolio para cada día de rebalanceo
            for i, rebalance_idx in enumerate(rebalance_days[:-1]):
                rebalance_date = self.portfolio.index[rebalance_idx]
                next_rebalance_idx = rebalance_days[i+1]
                
                # Obtener ranking en fecha de rebalanceo
                ranking = self.combined_ranking.loc[rebalance_date].dropna()
                
                if len(ranking) < n_positions:
                    print(f"No hay suficientes tickers con ranking en {rebalance_date}")
                    continue
                
                # Filtrar por volumen y capitalización
                valid_tickers = []
                for ticker in ranking.index:
                    if ticker in self.market_data and not self.market_data[ticker].empty:
                        # Verificar volumen promedio
                        avg_volume = self.market_data[ticker]['volume'].iloc[-63:].mean()
                        
                        if avg_volume >= self.min_volume:
                            valid_tickers.append(ticker)
                
                if len(valid_tickers) < n_positions:
                    print(f"No hay suficientes tickers válidos en {rebalance_date}")
                    continue
                
                # Filtrar ranking por tickers válidos
                valid_ranking = ranking[ranking.index.isin(valid_tickers)]
                
                # Seleccionar tickers según estrategia
                if strategy_type == 'long_only':
                    # Seleccionar top n_positions
                    selected_tickers = valid_ranking.nlargest(n_positions).index
                    weights = np.ones(len(selected_tickers)) / len(selected_tickers)
                    
                    # Asignar pesos
                    self.portfolio.loc[rebalance_date, 'cash'] = 0.0
                    for ticker, weight in zip(selected_tickers, weights):
                        self.portfolio.loc[rebalance_date, ticker] = weight
                
                elif strategy_type == 'market_neutral':
                    # Seleccionar top y bottom n_positions/2
                    long_tickers = valid_ranking.nlargest(n_positions//2).index
                    short_tickers = valid_ranking.nsmallest(n_positions//2).index
                    
                    # Pesos iguales para long y short
                    long_weights = np.ones(len(long_tickers)) / len(long_tickers)
                    short_weights = -np.ones(len(short_tickers)) / len(short_tickers)
                    
                    # Asignar pesos
                    self.portfolio.loc[rebalance_date, 'cash'] = 0.0
                    for ticker, weight in zip(long_tickers, long_weights):
                        self.portfolio.loc[rebalance_date, ticker] = weight
                    
                    for ticker, weight in zip(short_tickers, short_weights):
                        self.portfolio.loc[rebalance_date, ticker] = weight
                
                # Mantener pesos hasta próximo rebalanceo
                for date_idx in range(rebalance_idx+1, next_rebalance_idx):
                    if date_idx < len(self.portfolio):
                        date = self.portfolio.index[date_idx]
                        self.portfolio.loc[date] = self.portfolio.loc[rebalance_date]
            
            # Forward fill para días restantes
            self.portfolio.fillna(method='ffill', inplace=True)
            
            print(f"Portafolio generado con estrategia {strategy_type}")
        except Exception as e:
            logging.error(f"Error en generate_portfolio: {str(e)}")
            raise
    
    def backtest_strategy(self):
        """Realiza backtest de la estrategia"""
        try:
            if self.portfolio.empty:
                print("No hay portafolio para realizar backtest")
                return
            
            # Inicializar DataFrame de rendimientos
            self.performance = pd.DataFrame(index=self.portfolio.index)
            self.performance['daily_return'] = 0.0
            self.performance['equity_curve'] = 1.0
            
            # Calcular rendimientos diarios para cada ticker
            ticker_returns = {}
            for ticker in self.portfolio.columns:
                if ticker != 'cash' and ticker in self.market_data:
                    # Alinear fechas
                    common_dates = self.portfolio.index.intersection(self.market_data[ticker].index)
                    if len(common_dates) > 0:
                        ticker_returns[ticker] = self.market_data[ticker].loc[common_dates, 'returns']
            
            # Calcular rendimiento del portafolio
            for i in range(1, len(self.portfolio)):
                date = self.portfolio.index[i]
                prev_date = self.portfolio.index[i-1]
                
                # Pesos del día anterior
                weights = self.portfolio.loc[prev_date]
                
                # Rendimiento del día
                daily_return = 0.0
                
                for ticker, weight in weights.items():
                    if ticker == 'cash':
                        # Rendimiento de efectivo (0)
                        continue
                    elif ticker in ticker_returns and date in ticker_returns[ticker].index:
                        # Rendimiento ponderado
                        ticker_return = ticker_returns[ticker].loc[date]
                        if pd.notna(ticker_return):
                            daily_return += weight * ticker_return
                
                # Guardar rendimiento diario
                self.performance.loc[date, 'daily_return'] = daily_return
                
                # Actualizar equity curve
                self.performance.loc[date, 'equity_curve'] = self.performance.loc[prev_date, 'equity_curve'] * (1 + daily_return)
            
            # Calcular métricas de rendimiento
            self._calculate_performance_metrics()
            
            print("Backtest completado")
        except Exception as e:
            logging.error(f"Error en backtest_strategy: {str(e)}")
            raise
    
    def _calculate_performance_metrics(self):
        """Calcula métricas de rendimiento de la estrategia"""
        try:
            # Rendimientos diarios
            returns = self.performance['daily_return']
            
            # Rendimiento acumulado
            cumulative_return = self.performance['equity_curve'].iloc[-1] / self.performance['equity_curve'].iloc[0] - 1
            
            # Rendimiento anualizado
            years = (self.performance.index[-1] - self.performance.index[0]).days / 365.25
            annual_return = (1 + cumulative_return) ** (1 / years) - 1
            
            # Volatilidad anualizada
            annual_volatility = returns.std() * np.sqrt(252)
            
            # Sharpe Ratio
            sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
            
            # Sortino Ratio (solo considera volatilidad negativa)
            negative_returns = returns[returns < 0]
            downside_deviation = negative_returns.std() * np.sqrt(252)
            sortino_ratio = annual_return / downside_deviation if downside_deviation > 0 else 0
            
            # Maximum Drawdown
            equity_curve = self.performance['equity_curve']
            rolling_max = equity_curve.cummax()
            drawdown = (equity_curve / rolling_max - 1)
            max_drawdown = drawdown.min()
            
            # Calcular rendimientos del S&P 500 (SPY) para comparación
            if 'SPY' in self.market_data:
                spy_returns = self.market_data['SPY']['returns']
                common_dates = self.performance.index.intersection(spy_returns.index)
                
                if len(common_dates) > 0:
                    spy_returns = spy_returns.loc[common_dates]
                    
                    # Equity curve del S&P 500
                    spy_equity = (1 + spy_returns).cumprod()
                    spy_equity = spy_equity / spy_equity.iloc[0]
                    
                    # Rendimiento acumulado del S&P 500
                    spy_cumulative_return = spy_equity.iloc[-1] / spy_equity.iloc[0] - 1
                    
                    # Rendimiento anualizado del S&P 500
                    spy_annual_return = (1 + spy_cumulative_return) ** (1 / years) - 1
                    
                    # Volatilidad anualizada del S&P 500
                    spy_annual_volatility = spy_returns.std() * np.sqrt(252)
                    
                    # Sharpe Ratio del S&P 500
                    spy_sharpe_ratio = spy_annual_return / spy_annual_volatility if spy_annual_volatility > 0 else 0
                    
                    # Maximum Drawdown del S&P 500
                    spy_rolling_max = spy_equity.cummax()
                    spy_drawdown = (spy_equity / spy_rolling_max - 1)
                    spy_max_drawdown = spy_drawdown.min()
                    
                    # Information Ratio
                    excess_returns = returns - spy_returns
                    tracking_error = excess_returns.std() * np.sqrt(252)
                    information_ratio = (annual_return - spy_annual_return) / tracking_error if tracking_error > 0 else 0
                    
                    # Beta
                    covariance = returns.cov(spy_returns)
                    variance = spy_returns.var()
                    beta = covariance / variance if variance > 0 else 0
                    
                    # Alpha anualizado
                    alpha = annual_return - (0.02 + beta * (spy_annual_return - 0.02))  # Asumiendo tasa libre de riesgo de 2%
                    
                    # Captura alcista/bajista
                    up_markets = spy_returns[spy_returns > 0]
                    down_markets = spy_returns[spy_returns < 0]
                    
                    strategy_up = returns.loc[up_markets.index]
                    strategy_down = returns.loc[down_markets.index]
                    
                    up_capture = strategy_up.mean() / up_markets.mean() if up_markets.mean() > 0 else 0
                    down_capture = strategy_down.mean() / down_markets.mean() if down_markets.mean() < 0 else 0
                    
                    # Guardar métricas de comparación
                    self.performance_metrics = {
                        'Cumulative Return': cumulative_return,
                        'Annual Return': annual_return,
                        'Annual Volatility': annual_volatility,
                        'Sharpe Ratio': sharpe_ratio,
                        'Sortino Ratio': sortino_ratio,
                        'Maximum Drawdown': max_drawdown,
                        'SPY Cumulative Return': spy_cumulative_return,
                        'SPY Annual Return': spy_annual_return,
                        'SPY Annual Volatility': spy_annual_volatility,
                        'SPY Sharpe Ratio': spy_sharpe_ratio,
                        'SPY Maximum Drawdown': spy_max_drawdown,
                        'Information Ratio': information_ratio,
                        'Beta': beta,
                        'Alpha': alpha,
                        'Up Capture': up_capture,
                        'Down Capture': down_capture
                    }
                else:
                    # Métricas sin comparación con SPY
                    self.performance_metrics = {
                        'Cumulative Return': cumulative_return,
                        'Annual Return': annual_return,
                        'Annual Volatility': annual_volatility,
                        'Sharpe Ratio': sharpe_ratio,
                        'Sortino Ratio': sortino_ratio,
                        'Maximum Drawdown': max_drawdown
                    }
            else:
                # Métricas sin comparación con SPY
                self.performance_metrics = {
                    'Cumulative Return': cumulative_return,
                    'Annual Return': annual_return,
                    'Annual Volatility': annual_volatility,
                    'Sharpe Ratio': sharpe_ratio,
                    'Sortino Ratio': sortino_ratio,
                    'Maximum Drawdown': max_drawdown
                }
            
            # Guardar métricas en CSV
            metrics_df = pd.DataFrame(list(self.performance_metrics.items()), columns=['Metric', 'Value'])
            metrics_df.to_csv('./artifacts/results/performance_metrics.csv', index=False)
            
            print("Métricas de rendimiento calculadas y guardadas")
        except Exception as e:
            logging.error(f"Error en _calculate_performance_metrics: {str(e)}")
            raise
    
    def generate_reports(self):
        """Genera reportes y visualizaciones de la estrategia"""
        try:
            if not hasattr(self, 'performance_metrics'):
                print("No hay métricas de rendimiento para generar reportes")
                return
            
            # 1. Gráfico de equity curve
            plt.figure(figsize=(12, 6))
            plt.plot(self.performance.index, self.performance['equity_curve'], label='Estrategia')
            
            # Añadir SPY para comparación
            if 'SPY' in self.market_data:
                spy_returns = self.market_data['SPY']['returns']
                common_dates = self.performance.index.intersection(spy_returns.index)
                
                if len(common_dates) > 0:
                    spy_returns = spy_returns.loc[common_dates]
                    spy_equity = (1 + spy_returns).cumprod()
                    spy_equity = spy_equity / spy_equity.iloc[0]
                    
                    plt.plot(common_dates, spy_equity, label='S&P 500', alpha=0.7)
            
            plt.title('Equity Curve')
            plt.xlabel('Fecha')
            plt.ylabel('Valor')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('./artifacts/results/figures/equity_curve.png')
            plt.close()
            
            # 2. Gráfico de drawdown
            equity_curve = self.performance['equity_curve']
            rolling_max = equity_curve.cummax()
            drawdown = (equity_curve / rolling_max - 1) * 100  # En porcentaje
            
            plt.figure(figsize=(12, 6))
            plt.plot(self.performance.index, drawdown)
            plt.fill_between(self.performance.index, drawdown, 0, alpha=0.3, color='red')
            plt.title('Drawdown')
            plt.xlabel('Fecha')
            plt.ylabel('Drawdown (%)')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('./artifacts/results/figures/drawdown.png')
            plt.close()
            
            # 3. Gráfico de rendimientos mensuales
            if len(self.performance) > 30:
                # Convertir a rendimientos mensuales
                monthly_returns = self.performance['daily_return'].resample('M').apply(
                    lambda x: (1 + x).prod() - 1
                ) * 100  # En porcentaje
                
                plt.figure(figsize=(12, 6))
                monthly_returns.plot(kind='bar', color=np.where(monthly_returns >= 0, 'green', 'red'))
                plt.title('Rendimientos Mensuales')
                plt.xlabel('Fecha')
                plt.ylabel('Rendimiento (%)')
                plt.grid(True, axis='y')
                plt.tight_layout()
                plt.savefig('./artifacts/results/figures/monthly_returns.png')
                plt.close()
            
            # 4. Gráfico de distribución de rendimientos diarios
            plt.figure(figsize=(12, 6))
            sns.histplot(self.performance['daily_return'] * 100, kde=True, bins=50)
            plt.axvline(0, color='red', linestyle='--')
            plt.title('Distribución de Rendimientos Diarios')
            plt.xlabel('Rendimiento Diario (%)')
            plt.ylabel('Frecuencia')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('./artifacts/results/figures/return_distribution.png')
            plt.close()
            
            # 5. Gráfico de exposición por posición
            if len(self.portfolio.columns) > 1:
                # Calcular exposición absoluta total por día
                self.portfolio['total_exposure'] = self.portfolio.drop(columns=['cash']).abs().sum(axis=1)
                
                # Seleccionar algunas fechas representativas
                dates = self.portfolio.index[::len(self.portfolio)//10]  # 10 fechas
                
                exposures = []
                for date in dates:
                    # Normalizar pesos por exposición total
                    weights = self.portfolio.loc[date].drop(['cash', 'total_exposure'])
                    abs_weights = weights.abs() / self.portfolio.loc[date, 'total_exposure'] if self.portfolio.loc[date, 'total_exposure'] > 0 else weights.abs()
                    
                    # Ordenar por magnitud
                    sorted_weights = abs_weights.sort_values(ascending=False)
                    
                    # Tomar top 10
                    top_weights = sorted_weights.head(10)
                    
                    # Guardar para gráfico
                    for ticker, weight in top_weights.items():
                        sign = np.sign(weights[ticker])
                        exposures.append({
                            'Date': date,
                            'Ticker': ticker,
                            'Weight': weight * 100,  # En porcentaje
                            'Direction': 'Long' if sign > 0 else 'Short'
                        })
                
                if exposures:
                    exposure_df = pd.DataFrame(exposures)
                    
                    plt.figure(figsize=(14, 8))
                    for i, date in enumerate(exposure_df['Date'].unique()):
                        date_df = exposure_df[exposure_df['Date'] == date]
                        
                        plt.subplot(2, 5, i+1)
                        colors = ['green' if d == 'Long' else 'red' for d in date_df['Direction']]
                        plt.barh(date_df['Ticker'], date_df['Weight'], color=colors)
                        plt.title(date.strftime('%Y-%m-%d'))
                        plt.xlabel('Peso (%)')
                        plt.grid(True, axis='x')
                    
                    plt.tight_layout()
                    plt.savefig('./artifacts/results/figures/position_exposure.png')
                    plt.close()
                
                # Eliminar columna auxiliar
                self.portfolio.drop(columns=['total_exposure'], inplace=True)
            
            # 6. Tabla de métricas de rendimiento
            metrics_table = pd.DataFrame(list(self.performance_metrics.items()), columns=['Metric', 'Value'])
            metrics_table['Value'] = metrics_table['Value'].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
            
            plt.figure(figsize=(10, 6))
            plt.axis('off')
            plt.table(cellText=metrics_table.values, colLabels=metrics_table.columns, loc='center', cellLoc='center')
            plt.title('Métricas de Rendimiento')
            plt.tight_layout()
            plt.savefig('./artifacts/results/figures/performance_metrics.png')
            plt.close()
            
            # 7. Guardar datos de rendimiento
            self.performance.to_csv('./artifacts/results/data/performance_data.csv')
            
            # 8. Guardar portafolio
            self.portfolio.to_csv('./artifacts/results/data/portfolio_weights.csv')
            
            print("Reportes generados y guardados")
        except Exception as e:
            logging.error(f"Error en generate_reports: {str(e)}")
            raise
    
    def run_strategy(self, strategy_type='long_only'):
        """
        Ejecuta la estrategia completa
        
        Args:
            strategy_type: 'long_only' o 'market_neutral'
        """
        try:
            print("Iniciando estrategia de ranking adaptativo multifactorial...")
            
            # 1. Obtener datos de mercado
            print("\n1. Obteniendo datos de mercado...")
            self.fetch_market_data()
            
            # 2. Generar características
            print("\n2. Generando características adaptativas...")
            self.generate_features()
            
            # 3. Normalizar características
            print("\n3. Normalizando características...")
            self.normalize_features()
            
            # 4. Reducir dimensionalidad
            print("\n4. Reduciendo dimensionalidad...")
            self.reduce_dimensions()
            
            # 5. Calcular rankings
            print("\n5. Calculando rankings...")
            self.calculate_rankings()
            
            # 6. Generar portafolio
            print(f"\n6. Generando portafolio ({strategy_type})...")
            self.generate_portfolio(strategy_type)
            
            # 7. Realizar backtest
            print("\n7. Realizando backtest...")
            self.backtest_strategy()
            
            # 8. Generar reportes
            print("\n8. Generando reportes...")
            self.generate_reports()
            
            print("\nEstrategia completada con éxito!")
            
            # Mostrar resumen de rendimiento
            print("\nResumen de rendimiento:")
            for metric, value in self.performance_metrics.items():
                print(f"{metric}: {value:.4f}")
        except Exception as e:
            logging.error(f"Error ejecutando estrategia: {str(e)}")
            print(f"Error ejecutando estrategia: {str(e)}")
            raise

# Ejecutar estrategia
if __name__ == "__main__":
    try:
        # Definir período de backtest
        start_date = '2018-01-01'
        end_date = '2023-12-31'
        
        # Inicializar sistema
        system = AdaptiveMultifactorRankingSystem(start_date=start_date, end_date=end_date)
        
        # Ejecutar estrategia
        system.run_strategy(strategy_type='long_only')
        
        print("Estrategia ejecutada correctamente")
    except Exception as e:
        logging.error(f"Error en ejecución principal: {str(e)}")
        print(f"Error en ejecución principal: {str(e)}")
