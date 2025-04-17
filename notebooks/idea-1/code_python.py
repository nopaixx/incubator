
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from statsmodels.tsa.stattools import coint, adfuller, kpss
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.api as sm
from scipy import stats
from sklearn.cluster import KMeans
from hmmlearn import hmm
import warnings
from datetime import datetime, timedelta
import sqlite3
import itertools
from tqdm import tqdm
from scipy.optimize import minimize

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

# Desactivar advertencias para lectura más limpia
warnings.filterwarnings('ignore')
np.random.seed(42)


class DataManager:
    """Gestiona la obtención y almacenamiento de datos de mercado"""
    
    def __init__(self, use_cache=True):
        self.use_cache = use_cache
        self.db_path = './artifacts/results/data/market_data.db'
        self._setup_db()
        
    def _setup_db(self):
        """Configura la base de datos SQLite para caché de datos"""
        if not self.use_cache:
            return
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Crear tabla para precios
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS prices (
                ticker TEXT,
                date TEXT,
                price REAL,
                volume REAL,
                PRIMARY KEY (ticker, date)
            )
            ''')
            
            # Crear tabla para datos de mercado
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                date TEXT,
                indicator TEXT,
                value REAL,
                PRIMARY KEY (date, indicator)
            )
            ''')
            
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"Error al configurar la base de datos: {str(e)}")
    
    def get_sp500_tickers(self, limit=50):
        """Obtener lista de tickers representativos del S&P 500"""
        try:
            # En una implementación real, obtendríamos el listado histórico completo
            # Para este ejemplo, usamos un subconjunto representativo
            top_sp500 = [
                'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'BRK-B', 'UNH', 'JNJ',
                'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'BAC', 'ABBV', 'PFE', 'AVGO', 'COST',
                'DIS', 'KO', 'PEP', 'CSCO', 'TMO', 'MRK', 'CMCSA', 'WMT', 'CRM', 'VZ', 'ACN', 'ADBE',
                'LLY', 'ABT', 'DHR', 'MCD', 'TXN', 'BMY', 'INTC', 'QCOM', 'AMD', 'AMGN', 'LIN',
                'PM', 'UPS', 'SBUX', 'INTU', 'NEE', 'IBM', 'RTX', 'CAT', 'GS', 'HON', 'AMAT', 'LOW',
                'DE', 'MS', 'BLK', 'SPGI', 'BA', 'MMM', 'ISRG', 'AXP', 'BKNG', 'GILD', 'TJX', 'ADI'
            ]
            return top_sp500[:limit]
        except Exception as e:
            logging.error(f"Error al obtener tickers del S&P 500: {str(e)}")
            return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META'][:limit]
    
    def get_stock_data(self, tickers, start_date, end_date):
        """Obtener datos de precios para una lista de tickers"""
        if not self.use_cache:
            return self._download_stock_data(tickers, start_date, end_date)
        
        try:
            # Verificar caché primero
            conn = sqlite3.connect(self.db_path)
            missing_tickers = []
            data_dict = {}
            
            for ticker in tickers:
                query = f"""
                SELECT date, price, volume FROM prices 
                WHERE ticker = '{ticker}' 
                AND date BETWEEN '{start_date}' AND '{end_date}'
                ORDER BY date
                """
                df = pd.read_sql_query(query, conn)
                
                if len(df) == 0 or len(df) < (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days * 0.7:
                    missing_tickers.append(ticker)
                else:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    df.columns = [f'{ticker}_close', f'{ticker}_volume']
                    data_dict[ticker] = df
                    
            # Descargar datos faltantes
            if missing_tickers:
                new_data = self._download_stock_data(missing_tickers, start_date, end_date)
                
                # Guardar en caché
                for ticker in missing_tickers:
                    if ticker in new_data:
                        ticker_data = new_data[ticker]
                        
                        for date, row in ticker_data.iterrows():
                            try:
                                conn.execute(
                                    "INSERT OR REPLACE INTO prices VALUES (?, ?, ?, ?)",
                                    (ticker, date.strftime('%Y-%m-%d'), row[f'{ticker}_close'], row[f'{ticker}_volume'])
                                )
                        data_dict[ticker] = ticker_data
                
                conn.commit()
                
            conn.close()
            return data_dict
        except Exception as e:
            logging.error(f"Error al obtener datos de caché: {str(e)}")
            return self._download_stock_data(tickers, start_date, end_date)
        
    def _download_stock_data(self, tickers, start_date, end_date):
        """Descargar datos de precios utilizando yfinance"""
        data_dict = {}
        
        for ticker in tickers:
            try:
                # Descargar datos
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if data.empty:
                    continue
                    
                # Preparar datos
                df = pd.DataFrame({
                    f'{ticker}_close': data['Close'],
                    f'{ticker}_volume': data['Volume']
                })
                
                # Manejar datos faltantes
                if df.isna().sum().sum() > 0:
                    # Interpolación para gaps pequeños
                    df = df.interpolate(method='linear', limit=3)
                    # Eliminar filas que aún tienen NaN
                    df = df.dropna()
                
                if len(df) > 0:
                    data_dict[ticker] = df
                    
            except Exception as e:
                logging.error(f"Error al descargar datos para {ticker}: {str(e)}")
                
        return data_dict
        
    def get_market_factors(self, start_date, end_date):
        """Obtener factores de mercado: VIX, Term Spread, Credit Spread, Liquidez"""
        if not self.use_cache:
            return self._download_market_factors(start_date, end_date)
            
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Verificar si tenemos datos en caché
            query = f"""
            SELECT date, indicator, value FROM market_data
            WHERE date BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY date, indicator
            """
            cached_data = pd.read_sql_query(query, conn)
            
            if len(cached_data) == 0 or len(cached_data['date'].unique()) < (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days * 0.7:
                # No hay suficientes datos en caché, descargar
                market_data = self._download_market_factors(start_date, end_date)
                
                # Guardar en caché
                for date, row in market_data.iterrows():
                    for indicator in row.index:
                        try:
                            conn.execute(
                                "INSERT OR REPLACE INTO market_data VALUES (?, ?, ?)",
                                (date.strftime('%Y-%m-%d'), indicator, float(row[indicator]))
                            )
                        except Exception as e:
                            logging.error(f"Error al guardar factor de mercado {indicator}: {str(e)}")
                
                conn.commit()
                conn.close()
                return market_data
            else:
                # Formatear datos de caché
                cached_data['date'] = pd.to_datetime(cached_data['date'])
                market_data = cached_data.pivot(index='date', columns='indicator', values='value')
                conn.close()
                return market_data
        except Exception as e:
            logging.error(f"Error al obtener factores de mercado de caché: {str(e)}")
            return self._download_market_factors(start_date, end_date)
    
    def _download_market_factors(self, start_date, end_date):
        """Descargar factores de mercado utilizando yfinance"""
        try:
            # VIX
            vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)['Close']
            vix = np.log(vix)
            
            # Tasas del Tesoro para term spread
            try:
                t10y = yf.download('^TNX', start=start_date, end=end_date, progress=False)['Close'] / 10
                t2y = yf.download('^TYX', start=start_date, end=end_date, progress=False)['Close'] / 10
                term_spread = t10y - t2y
            except:
                # Fallback si no se pueden obtener datos de tasas
                spy_data = yf.download('SPY', start=start_date, end=end_date, progress=False)
                term_spread = pd.Series(0.01, index=spy_data.index)  # Valor predeterminado
            
            # Proxy para credit spread 
            spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
            credit_spread = spy['High'].pct_change().rolling(20).std() * np.sqrt(252)
            
            # Índice de liquidez (volumen normalizado)
            liquidity_index = spy['Volume'].rolling(20).mean() / spy['Volume'].rolling(60).mean()
            
            # Combinar todos los factores
            common_index = vix.index.intersection(term_spread.index).intersection(credit_spread.index).intersection(liquidity_index.index)
            
            market_data = pd.DataFrame({
                'vix': vix.loc[common_index],
                'term_spread': term_spread.loc[common_index],
                'credit_spread': credit_spread.loc[common_index],
                'liquidity_index': liquidity_index.loc[common_index]
            })
            
            # Manejar valores faltantes
            market_data = market_data.fillna(method='ffill').dropna()
            
            return market_data
            
        except Exception as e:
            logging.error(f"Error al descargar factores de mercado: {str(e)}")
            # Retornar un DataFrame vacío como fallback
            dates = pd.date_range(start=start_date, end=end_date, freq='B')
            return pd.DataFrame(index=dates, columns=['vix', 'term_spread', 'credit_spread', 'liquidity_index']).fillna(0)


class PairSelector:
    """Selecciona pares cointegrados utilizando filtros secuenciales"""
    
    def __init__(self, data_manager, lookback_period=252):
        self.data_manager = data_manager
        self.lookback_period = lookback_period
        
    def select_pairs(self, end_date, top_n=30):
        """Seleccionar los mejores pares cointegrados"""
        start_date = (pd.to_datetime(end_date) - pd.Timedelta(days=self.lookback_period)).strftime('%Y-%m-%d')
        
        # Obtener tickers del S&P 500 (limitados para este ejemplo)
        tickers = self.data_manager.get_sp500_tickers(limit=30)
        
        # Obtener datos de precios
        stock_data = self.data_manager.get_stock_data(tickers, start_date, end_date)
        
        # Combinar datos y llenar valores faltantes
        prices_df = self._combine_price_data(stock_data)
        
        if prices_df.empty or len(prices_df.columns) < 5:
            logging.error("No hay suficientes datos de precios para seleccionar pares")
            return []
        
        # Agrupar por correlación
        corr_matrix = prices_df.pct_change().dropna().corr()
        ticker_groups = self._cluster_stocks(corr_matrix)
        
        # Evaluar cointegración dentro de cada grupo
        coint_pairs = []
        
        for group in ticker_groups:
            if len(group) < 2:
                continue
                
            group_pairs = self._test_cointegration(group, prices_df)
            coint_pairs.extend(group_pairs)
        
        # Ordenar pares por p-valor de cointegración
        coint_pairs.sort(key=lambda x: x[2])
        
        # Seleccionar los mejores pares que cumplen criterios
        selected_pairs = []
        for pair in coint_pairs:
            if len(selected_pairs) >= top_n:
                break
                
            # Verificar half-life
            if 1 <= pair[5] <= 30:
                selected_pairs.append(pair)
        
        return selected_pairs
    
    def _combine_price_data(self, stock_data):
        """Combinar datos de precios de múltiples acciones"""
        if not stock_data:
            return pd.DataFrame()
            
        # Extraer columnas de precios de cierre
        dfs = []
        for ticker, data in stock_data.items():
            df = pd.DataFrame({ticker: data[f'{ticker}_close']})
            dfs.append(df)
        
        # Combinar datos
        if not dfs:
            return pd.DataFrame()
            
        prices_df = pd.concat(dfs, axis=1)
        
        # Eliminar columnas con más de 20% de valores faltantes
        threshold = len(prices_df) * 0.2
        prices_df = prices_df.dropna(axis=1, thresh=threshold)
        
        # Llenar valores faltantes restantes
        prices_df = prices_df.fillna(method='ffill')
        
        return prices_df
    
    def _cluster_stocks(self, corr_matrix, n_clusters=None):
        """Agrupar acciones basado en correlaciones"""
        # Convertir matriz de correlación a matriz de distancia
        distance_matrix = 1 - np.abs(corr_matrix.values)
        
        # Ajustar número de clusters según datos disponibles
        n_stocks = len(corr_matrix)
        if n_clusters is None:
            n_clusters = max(2, min(10, n_stocks // 5))
        
        # Aplicar K-means
        if n_stocks >= n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(distance_matrix)
            
            # Crear grupos
            groups = []
            for i in range(n_clusters):
                group_indices = np.where(labels == i)[0]
                group = [corr_matrix.index[idx] for idx in group_indices]
                groups.append(group)
                
            return groups
        else:
            # Si no hay suficientes stocks, retornar un solo grupo
            return [corr_matrix.index.tolist()]
    
    def _test_cointegration(self, tickers, prices_df):
        """Probar cointegración entre pares de acciones"""
        coint_pairs = []
        
        for i, j in itertools.combinations(tickers, 2):
            if i not in prices_df.columns or j not in prices_df.columns:
                continue
                
            stock1 = prices_df[i].values
            stock2 = prices_df[j].values
            
            # Verificar datos válidos
            if np.isnan(stock1).any() or np.isnan(stock2).any():
                continue
                
            # Test de cointegración de Engle-Granger
            result = coint(stock1, stock2)
            pvalue = result[1]
            
            if pvalue < 0.01:  # Umbral de 1%
                # Estimar parámetros de cointegración
                X = sm.add_constant(stock1)
                model = sm.OLS(stock2, X).fit()
                beta = model.params[1]
                alpha = model.params[0]
                
                # Calcular residuos
                residuals = stock2 - (beta * stock1 + alpha)
                
                # Test de estacionariedad de residuos (KPSS)
                try:
                    kpss_stat, kpss_pvalue = kpss(residuals, regression='c')
                    
                    # Aceptar solo si residuos son estacionarios (H0 de KPSS es estacionariedad)
                    if kpss_pvalue > 0.05:
                        # Estimar half-life
                        ar_model = AutoReg(residuals, lags=1).fit()
                        phi = ar_model.params[1]
                        half_life = np.log(0.5) / np.log(abs(phi)) if abs(phi) < 1 else np.inf
                        
                        # Filtrar por half-life
                        if 1 <= half_life <= 30:
                            # Estimar intervalo de confianza bayesiano simplificado
                            n = len(residuals)
                            std_error = model.bse[1]
                            t_stat = stats.t.ppf(0.975, n-2)  # 95% de confianza
                            beta_lower = beta - t_stat * std_error
                            beta_upper = beta + t_stat * std_error
                            
                            # Verificar que beta sea significativo (IC no incluye cero)
                            if beta_lower * beta_upper > 0:
                                # Identificar ciclos de trading
                                zero_crossings = np.where(np.diff(np.signbit(residuals)))[0]
                                if len(zero_crossings) >= 10:  # Al menos 10 ciclos
                                    coint_pairs.append((i, j, pvalue, beta, alpha, half_life))
                except Exception as e:
                    logging.error(f"Error en prueba KPSS para {i}-{j}: {str(e)}")
        
        return coint_pairs
    
    def estimate_bayesian_parameters(self, pair, prices_df):
        """Estimar parámetros bayesianos para un par (versión simplificada)"""
        stock1, stock2 = pair[0], pair[1]
        beta, alpha = pair[3], pair[4]
        
        # Calcular residuos
        residuals = prices_df[stock2] - (beta * prices_df[stock1] + alpha)
        
        # Estimar modelo AR(1) para residuos
        ar_model = AutoReg(residuals, lags=1).fit()
        phi = ar_model.params[1]
        sigma = np.std(ar_model.resid)
        
        # Calcular half-life
        half_life = np.log(0.5) / np.log(abs(phi)) if abs(phi) < 1 else 30
        
        # Simulación para estimar error estándar
        n_simulations = 1000
        beta_samples = np.random.normal(beta, np.std(residuals) / np.std(prices_df[stock1]) / np.sqrt(len(residuals)), n_simulations)
        alpha_samples = np.random.normal(alpha, np.std(residuals) / np.sqrt(len(residuals)), n_simulations)
        
        # Calcular intervalos de credibilidad
        beta_hdi = [np.percentile(beta_samples, 2.5), np.percentile(beta_samples, 97.5)]
        alpha_hdi = [np.percentile(alpha_samples, 2.5), np.percentile(alpha_samples, 97.5)]
        
        return {
            'beta': beta,
            'alpha': alpha,
            'phi': phi,
            'sigma': sigma,
            'half_life': half_life,
            'beta_hdi': beta_hdi,
            'alpha_hdi': alpha_hdi
        }


class RegimeDetector:
    """Detecta regímenes de mercado utilizando HMM"""
    
    def __init__(self, data_manager, n_regimes=3, lookback_period=504):
        self.data_manager = data_manager
        self.n_regimes = n_regimes
        self.lookback_period = lookback_period
        self.model = None
        self.regime_map = None
        
    def fit(self, end_date):
        """Ajustar modelo HMM a datos de mercado"""
        start_date = (pd.to_datetime(end_date) - pd.Timedelta(days=self.lookback_period)).strftime('%Y-%m-%d')
        
        # Obtener factores de mercado
        market_data = self.data_manager.get_market_factors(start_date, end_date)
        
        if market_data.empty:
            logging.error("No hay datos de mercado para ajustar el modelo de regímenes")
            return False
            
        # Preprocesar datos
        X = self._preprocess_features(market_data)
        
        if X.shape[0] < 50:  # Verificar que hay suficientes datos
            logging.error(f"Insuficientes datos para ajustar HMM: {X.shape[0]} observaciones")
            return False
            
        try:
            # Inicializar modelo con K-means
            kmeans = KMeans(n_clusters=self.n_regimes, random_state=42)
            kmeans.fit(X)
            
            # Ajustar HMM
            self.model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="diag",
                n_iter=100,
                random_state=42
            )
            
            # Inicializar con K-means
            self.model.startprob_ = np.ones(self.n_regimes) / self.n_regimes
            self.model.transmat_ = np.ones((self.n_regimes, self.n_regimes)) * 0.1
            np.fill_diagonal(self.model.transmat_, 0.9)  # Alta probabilidad de permanecer en mismo régimen
            
            # Ajustar modelo
            self.model.fit(X)
            
            # Decodificar estados
            decoded_states = self.model.predict(X)
            
            # Identificar regímenes por volatilidad
            self._label_regimes(X, decoded_states)
            
            return True
            
        except Exception as e:
            logging.error(f"Error al ajustar modelo HMM: {str(e)}")
            return False
            
    def _preprocess_features(self, market_data):
        """Preprocesar características para HMM"""
        # Seleccionar y normalizar características
        features = [
            'vix',
            'term_spread',
            'credit_spread',
            'liquidity_index'
        ]
        
        # Asegurar que existen todas las columnas
        for feat in features:
            if feat not in market_data.columns:
                market_data[feat] = 0.0
                
        # Seleccionar características
        X = market_data[features].copy()
        
        # Normalizar
        for col in X.columns:
            X[col] = (X[col] - X[col].mean()) / (X[col].std() + 1e-6)
            
        return X.values
        
    def _label_regimes(self, X, decoded_states):
        """Etiquetar regímenes según su volatilidad"""
        # Calcular volatilidad por régimen
        regime_volatility = {}
        
        for i in range(self.n_regimes):
            regime_data = X[decoded_states == i]
            if len(regime_data) > 0:
                volatility = np.mean(np.std(regime_data, axis=0))
                regime_volatility[i] = volatility
        
        # Ordenar regímenes por volatilidad
        sorted_regimes = sorted(regime_volatility.items(), key=lambda x: x[1])
        
        # Mapear estados originales a estados ordenados (1=baja vol, 2=media vol, 3=alta vol)
        self.regime_map = {}
        for i, (original_state, _) in enumerate(sorted_regimes):
            self.regime_map[original_state] = i + 1
        
        print(f"Regímenes identificados: {self.regime_map}")
            
    def predict_current_regime(self, date):
        """Predecir régimen para una fecha específica"""
        if self.model is None:
            logging.error("Modelo HMM no ajustado")
            return 2  # Retornar régimen normal como fallback
            
        # Obtener datos recientes
        start_date = (pd.to_datetime(date) - pd.Timedelta(days=10)).strftime('%Y-%m-%d')
        market_data = self.data_manager.get_market_factors(start_date, date)
        
        if market_data.empty:
            logging.error(f"No hay datos de mercado para predecir régimen en {date}")
            return 2
            
        # Preprocesar datos
        X = self._preprocess_features(market_data)
        
        if len(X) == 0:
            return 2
            
        # Predecir estado
        latest_x = X[-1].reshape(1, -1)
        state = self.model.predict(latest_x)[0]
        
        # Mapear a régimen etiquetado
        if self.regime_map is not None:
            return self.regime_map.get(state, 2)
        else:
            return state + 1
        
    def get_regime_thresholds(self, regime):
        """Obtener umbrales de trading según régimen"""
        # Umbrales por régimen
        thresholds = {
            1: {'entry': 1.25, 'exit': 0.5},   # Baja volatilidad
            2: {'entry': 1.75, 'exit': 0.75},  # Volatilidad normal
            3: {'entry': 2.25, 'exit': 1.0}    # Alta volatilidad
        }
        
        return thresholds.get(regime, thresholds[2])
        
    def get_regime_exposure(self, regime):
        """Obtener exposición máxima según régimen"""
        exposures = {
            1: 1.0,    # 100% en baja volatilidad
            2: 0.7,    # 70% en volatilidad normal
            3: 0.4     # 40% en alta volatilidad
        }
        
        return exposures.get(regime, exposures[2])


class StatisticalArbitrageStrategy:
    """Implementa la estrategia de arbitraje estadístico con adaptación a regímenes"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.pair_selector = PairSelector(data_manager)
        self.regime_detector = RegimeDetector(data_manager)
        self.active_pairs = []
        self.positions = {}
        self.quarantine_pairs = {}
        self.last_update = None
        
    def initialize(self, end_date):
        """Inicializar la estrategia"""
        print("Inicializando estrategia...")
        
        # Calibrar detector de regímenes
        print("Calibrando detector de regímenes...")
        regime_success = self.regime_detector.fit(end_date)
        if not regime_success:
            logging.error("No se pudo inicializar el detector de regímenes")
            
        # Seleccionar pares cointegrados
        print("Seleccionando pares cointegrados...")
        self.active_pairs = self.pair_selector.select_pairs(end_date)
        
        if not self.active_pairs:
            logging.error("No se encontraron pares cointegrados")
            
        self.last_update = pd.to_datetime(end_date)
        
        print(f"Inicialización completada. {len(self.active_pairs)} pares identificados.")
        return len(self.active_pairs) > 0
        
    def update_model(self, end_date):
        """Actualizar modelos semanalmente"""
        end_date_dt = pd.to_datetime(end_date)
        
        # Verificar si es hora de actualizar (semanal)
        if self.last_update is not None:
            days_since_update = (end_date_dt - self.last_update).days
            if days_since_update < 5:  # Menos de 5 días desde actualización
                return False
        
        print("Actualizando modelos...")
        
        # Recalibrar detector de regímenes
        self.regime_detector.fit(end_date)
        
        # Actualizar pares
        self.update_pairs(end_date)
        
        self.last_update = end_date_dt
        return True
        
    def update_pairs(self, date):
        """Actualizar universo de pares cointegrados"""
        # Revisar pares en cuarentena
        current_date = pd.to_datetime(date)
        pairs_to_remove = []
        
        for pair, quarantine_end in list(self.quarantine_pairs.items()):
            if current_date >= quarantine_end:
                pairs_to_remove.append(pair)
                
        # Eliminar pares que salen de cuarentena
        for pair in pairs_to_remove:
            del self.quarantine_pairs[pair]
        
        # Seleccionar nuevos pares
        new_pairs = self.pair_selector.select_pairs(date)
        
        # Filtrar pares en cuarentena
        filtered_pairs = [p for p in new_pairs if (p[0], p[1]) not in self.quarantine_pairs]
        
        self.active_pairs = filtered_pairs
        print(f"Universo de pares actualizado: {len(self.active_pairs)} pares activos")
        
    def calculate_zscore(self, stock1_prices, stock2_prices, beta, alpha):
        """Calcular z-score para un par"""
        # Calcular spread
        spread = stock2_prices - (beta * stock1_prices + alpha)
        
        # Calcular estadísticas del spread
        mean_spread = np.mean(spread)
        std_spread = np.std(spread)
        
        # Calcular z-score
        if std_spread > 0:
            zscore = (spread[-1] - mean_spread) / std_spread
        else:
            zscore = 0
            
        return zscore, spread
        
    def detect_structural_break(self, spread, cusum_threshold=5.0):
        """Detectar cambios estructurales con CUSUM"""
        # Calcular cambios
        s_t = 0
        cusum_values = [0]
        changes = np.diff(spread)
        std_changes = np.std(changes) + 1e-6  # Evitar división por cero
        
        # CUSUM recursivo
        k = 0.5  # Sensibilidad
        
        for t in range(1, len(changes)):
            s_t = max(0, s_t + (abs(changes[t]) - k * std_changes))
            cusum_values.append(s_t)
            
        # Verificar si supera umbral
        if s_t > cusum_threshold:
            return True
        else:
            return False
            
    def generate_signals(self, current_date):
        """Generar señales de trading para la fecha actual"""
        current_date_dt = pd.to_datetime(current_date)
        
        # Obtener régimen actual
        current_regime = self.regime_detector.predict_current_regime(current_date)
        
        # Obtener umbrales según régimen
        thresholds = self.regime_detector.get_regime_thresholds(current_regime)
        
        # Fecha de inicio para datos
        start_date = (current_date_dt - pd.Timedelta(days=60)).strftime('%Y-%m-%d')
        
        # Procesar cada par activo
        signals = []
        
        for pair in self.active_pairs:
            stock1, stock2, _, beta, alpha, half_life = pair
            
            # Obtener datos recientes
            try:
                stock_data = self.data_manager.get_stock_data([stock1, stock2], start_date, current_date)
                
                if not stock1 in stock_data or not stock2 in stock_data:
                    continue
                    
                stock1_data = stock_data[stock1]
                stock2_data = stock_data[stock2]
                
                # Verificar volumen
                vol1 = stock1_data[f'{stock1}_volume']
                vol2 = stock2_data[f'{stock2}_volume']
                
                avg_vol1 = vol1.rolling(20).mean().iloc[-1]
                avg_vol2 = vol2.rolling(20).mean().iloc[-1]
                
                if vol1.iloc[-1] < avg_vol1 * 0.7 or vol2.iloc[-1] < avg_vol2 * 0.7:
                    continue  # Volumen insuficiente
                    
                # Calcular z-score
                stock1_prices = stock1_data[f'{stock1}_close'].values
                stock2_prices = stock2_data[f'{stock2}_close'].values
                
                zscore, spread = self.calculate_zscore(stock1_prices, stock2_prices, beta, alpha)
                
                # Verificar cambio estructural
                if self.detect_structural_break(spread):
                    if (stock1, stock2) not in self.quarantine_pairs:
                        # Poner par en cuarentena
                        quarantine_end = current_date_dt + pd.Timedelta(days=21)
                        self.quarantine_pairs[(stock1, stock2)] = quarantine_end
                        
                        # Si hay posición abierta, añadir señal de cierre
                        pair_key = f"{stock1}_{stock2}"
                        if pair_key in self.positions:
                            signals.append({
                                'pair': (stock1, stock2),
                                'type': 'exit',
                                'reason': 'structural_break',
                                'zscore': zscore
                            })
                    continue
                
                # Verificar señales de entrada/salida
                pair_key = f"{stock1}_{stock2}"
                
                if pair_key not in self.positions:
                    # Verificar criterios de re-entrada
                    if pair_key in self.position_history:
                        last_exit = self.position_history[pair_key]['exit_date']
                        if (current_date_dt - last_exit).days < 3:
                            continue  # Esperar al menos 3 días para re-entrar
                    
                    # Calcular ratio retorno esperado/riesgo
                    expected_return = abs(zscore) * beta * stock1_prices[-1] / stock2_prices[-1]
                    risk = np.std(spread) * 2  # 2-sigma como estimación de riesgo
                    reward_risk_ratio = expected_return / risk if risk > 0 else 0
                    
                    # Señal de entrada
                    if zscore < -thresholds['entry'] and reward_risk_ratio > 1.2:
                        signals.append({
                            'pair': (stock1, stock2),
                            'type': 'entry',
                            'direction': 'long',
                            'zscore': zscore,
                            'beta': beta,
                            'alpha': alpha,
                            'half_life': half_life
                        })
                    elif zscore > thresholds['entry'] and reward_risk_ratio > 1.2:
                        signals.append({
                            'pair': (stock1, stock2),
                            'type': 'entry',
                            'direction': 'short',
                            'zscore': zscore,
                            'beta': beta,
                            'alpha': alpha,
                            'half_life': half_life
                        })
                else:
                    # Señal de salida
                    position = self.positions[pair_key]
                    entry_date = position['entry_date']
                    days_in_position = (current_date_dt - entry_date).days
                    
                    # Criterios de salida
                    exit_zscore = position['direction'] == 'long' and zscore > -thresholds['exit']
                    exit_zscore_short = position['direction'] == 'short' and zscore < thresholds['exit']
                    exit_time = days_in_position > min(2 * half_life, 30)
                    
                    # Stop-loss adaptativo
                    stock1_entry = position.get('stock1_price', stock1_prices[-1])
                    stock2_entry = position.get('stock2_price', stock2_prices[-1])
                    
                    if position['direction'] == 'long':
                        # Long Y, Short X
                        pair_return = (stock2_prices[-1]/stock2_entry - beta * stock1_prices[-1]/stock1_entry)
                    else:
                        # Short Y, Long X
                        pair_return = (beta * stock1_prices[-1]/stock1_entry - stock2_prices[-1]/stock2_entry)
                    
                    # Stop-loss: 2-sigma del par
                    stop_loss_hit = pair_return < -2 * np.std(spread)
                    
                    if exit_zscore or exit_zscore_short or exit_time or stop_loss_hit:
                        reason = 'target_reached' if (exit_zscore or exit_zscore_short) else \
                                'time_limit' if exit_time else 'stop_loss'
                        signals.append({
                            'pair': (stock1, stock2),
                            'type': 'exit',
                            'reason': reason,
                            'zscore': zscore
                        })
                    
            except Exception as e:
                logging.error(f"Error al generar señales para {stock1}-{stock2}: {str(e)}")
        
        return signals, current_regime
        
    def execute_signals(self, signals, date, regime):
        """Ejecutar señales de trading"""
        # Inicializar historial de posiciones si no existe
        if not hasattr(self, 'position_history'):
            self.position_history = {}
            
        # Historial de trading actual
        current_trades = []
        
        # Registrar señales
        for signal in signals:
            pair = signal['pair']
            pair_key = f"{pair[0]}_{pair[1]}"
            
            if signal['type'] == 'entry' and pair_key not in self.positions:
                # Obtener precios actuales
                start_date = (pd.to_datetime(date) - pd.Timedelta(days=5)).strftime('%Y-%m-%d')
                stock_data = self.data_manager.get_stock_data([pair[0], pair[1]], start_date, date)
                
                if pair[0] not in stock_data or pair[1] not in stock_data:
                    continue
                    
                stock1_price = stock_data[pair[0]][f'{pair[0]}_close'].iloc[-1]
                stock2_price = stock_data[pair[1]][f'{pair[1]}_close'].iloc[-1]
                
                # Abrir posición
                self.positions[pair_key] = {
                    'direction': signal['direction'],
                    'entry_date': pd.to_datetime(date),
                    'entry_zscore': signal['zscore'],
                    'beta': signal['beta'],
                    'alpha': signal['alpha'],
                    'half_life': signal['half_life'],
                    'stock1_price': stock1_price,
                    'stock2_price': stock2_price
                }
                
                # Registrar trade
                current_trades.append({
                    'pair': pair_key,
                    'action': 'OPEN',
                    'direction': signal['direction'],
                    'zscore': signal['zscore'],
                    'date': date
                })
                
            elif signal['type'] == 'exit' and pair_key in self.positions:
                # Obtener posición
                position = self.positions[pair_key]
                
                # Registrar en historial
                self.position_history[pair_key] = {
                    'entry_date': position['entry_date'],
                    'exit_date': pd.to_datetime(date),
                    'direction': position['direction'],
                    'half_life': position['half_life']
                }
                
                # Registrar trade
                current_trades.append({
                    'pair': pair_key,
                    'action': 'CLOSE',
                    'direction': position['direction'],
                    'zscore': signal['zscore'],
                    'reason': signal.get('reason', 'unknown'),
                    'date': date
                })
                
                # Cerrar posición
                del self.positions[pair_key]
        
        # Calcular exposición total
        max_exposure = self.regime_detector.get_regime_exposure(regime)
        n_positions = len(self.positions)
        
        # Calcular ponderación por par 
        weights = {}
        if n_positions > 0:
            # Base: 1/sqrt(N)
            base_weight = 1.0 / np.sqrt(n_positions)
            
            # Ajustar por calidad (inverso del ancho de half-life)
            quality_scores = {}
            total_quality = 0
            
            for pair_key, position in self.positions.items():
                half_life = position['half_life']
                quality = 1.0 / max(1, min(30, half_life))  # Invertir y limitar
                quality_scores[pair_key] = quality
                total_quality += quality
            
            # Normalizar por calidad
            for pair_key in self.positions:
                normalized_quality = quality_scores[pair_key] / total_quality if total_quality > 0 else 1.0 / n_positions
                
                # Peso final ajustado por régimen
                weights[pair_key] = normalized_quality * max_exposure
                
                # Aplicar cap máximo
                weights[pair_key] = min(weights[pair_key], 0.05)  # 5% máximo por par
        
        return weights, current_trades
        
    def backtest(self, start_date, end_date):
        """Realizar backtest de la estrategia"""
        print(f"Iniciando backtest desde {start_date} hasta {end_date}...")
        
        # Inicializar tracking de rendimiento
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        performance = pd.DataFrame(index=dates, columns=['return', 'cumulative_return', 'regime'])
        performance['return'] = 0.0
        performance['cumulative_return'] = 1.0
        performance['regime'] = 2  # Régimen normal por defecto
        
        # Historial de transacciones
        transactions = []
        
        # Historial de posiciones
        position_history = []
        
        # Inicializar estrategia (usar datos anteriores al periodo de backtest)
        init_date = (pd.to_datetime(start_date) - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        self.initialize(init_date)
        
        # Loop principal de backtesting
        for i, date in enumerate(tqdm(dates)):
            current_date = date.strftime('%Y-%m-%d')
            
            # Actualizar modelo semanalmente (viernes)
            if date.dayofweek == 4 or i == 0:
                self.update_model(current_date)
                
            # Generar señales
            signals, regime = self.generate_signals(current_date)
            
            # Guardar régimen actual
            performance.loc[date, 'regime'] = regime
            
            # Ejecutar señales
            weights, trades = self.execute_signals(signals, current_date, regime)
            
            # Registrar transacciones
            for trade in trades:
                transactions.append(trade)
            
            # Calcular rendimiento (simulado)
            if self.positions:
                # Guardar historial de posiciones
                position_snapshot = {
                    'date': current_date,
                    'n_positions': len(self.positions),
                    'regime': regime
                }
                position_history.append(position_snapshot)
                
                # Simular rendimiento de posiciones abiertas
                daily_return = 0.0
                
                for pair_key, position in self.positions.items():
                    stock1, stock2 = pair_key.split('_')
                    
                    # Obtener precios para la fecha actual y anterior
                    try:
                        prev_date = (date - pd.Timedelta(days=5)).strftime('%Y-%m-%d')
                        stock_data = self.data_manager.get_stock_data([stock1, stock2], prev_date, current_date)
                        
                        if stock1 not in stock_data or stock2 not in stock_data:
                            continue
                            
                        # Calcular rendimientos
                        stock1_df = stock_data[stock1][f'{stock1}_close']
                        stock2_df = stock_data[stock2][f'{stock2}_close']
                        
                        if len(stock1_df) < 2 or len(stock2_df) < 2:
                            continue
                            
                        stock1_ret = stock1_df.pct_change().iloc[-1]
                        stock2_ret = stock2_df.pct_change().iloc[-1]
                        
                        # Evitar NaN
                        if np.isnan(stock1_ret) or np.isnan(stock2_ret):
                            continue
                            
                        # Calcular spread return
                        beta = position['beta']
                        
                        if position['direction'] == 'long':
                            # Long Y, Short X
                            pair_return = stock2_ret - beta * stock1_ret
                        else:
                            # Short Y, Long X
                            pair_return = beta * stock1_ret - stock2_ret
                            
                        # Aplicar peso
                        pair_weight = weights.get(pair_key, 0)
                        daily_return += pair_return * pair_weight
                        
                    except Exception as e:
                        logging.error(f"Error al calcular rendimiento para {pair_key}: {str(e)}")
                
                # Guardar rendimiento
                performance.loc[date, 'return'] = daily_return
            
            # Actualizar rendimiento acumulado
            if i > 0:
                prev_cum_return = performance.iloc[i-1]['cumulative_return']
                performance.loc[date, 'cumulative_return'] = prev_cum_return * (1 + performance.loc[date, 'return'])
        
        # Calcular métricas de rendimiento
        metrics = self.calculate_performance_metrics(performance)
        
        # Guardar resultados
        self.save_results(performance, transactions, position_history, metrics)
        
        return metrics, performance
        
    def calculate_performance_metrics(self, performance):
        """Calcular métricas de rendimiento"""
        daily_returns = performance['return'].values
        
        # Métricas básicas
        total_return = performance['cumulative_return'].iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
        daily_vol = np.std(daily_returns)
        annual_vol = daily_vol * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Drawdown
        cum_returns = performance['cumulative_return'].values
        expanding_max = np.maximum.accumulate(cum_returns)
        drawdowns = 1 - cum_returns / expanding_max
        max_drawdown = np.max(drawdowns)
        
        # Calmar ratio
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        # Métricas por régimen
        regime_metrics = {}
        for regime in [1, 2, 3]:  # Baja, media, alta volatilidad
            regime_data = performance[performance['regime'] == regime]
            
            if len(regime_data) > 0:
                regime_return = regime_data['return'].mean() * 252  # Anualizado
                regime_vol = regime_data['return'].std() * np.sqrt(252)
                regime_sharpe = regime_return / regime_vol if regime_vol > 0 else 0
                
                regime_metrics[f'regime_{regime}'] = {
                    'days': len(regime_data),
                    'return': regime_return,
                    'volatility': regime_vol,
                    'sharpe': regime_sharpe
                }
        
        # Combinar métricas
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'regime_metrics': regime_metrics
        }
        
        return metrics
        
    def save_results(self, performance, transactions, position_history, metrics):
        """Guardar resultados del backtest"""
        # Guardar performance
        performance.to_csv('./artifacts/results/data/performance.csv')
        
        # Guardar transacciones
        transactions_df = pd.DataFrame(transactions)
        if not transactions_df.empty:
            transactions_df.to_csv('./artifacts/results/data/transactions.csv', index=False)
        
        # Guardar historial de posiciones
        positions_df = pd.DataFrame(position_history)
        if not positions_df.empty:
            positions_df.to_csv('./artifacts/results/data/positions.csv', index=False)
        
        # Guardar métricas
        metrics_df = pd.DataFrame({
            'Metric': [
                'Total Return',
                'Annual Return',
                'Annual Volatility',
                'Sharpe Ratio',
                'Max Drawdown',
                'Calmar Ratio'
            ],
            'Value': [
                metrics['total_return'],
                metrics['annual_return'],
                metrics['annual_volatility'],
                metrics['sharpe_ratio'],
                metrics['max_drawdown'],
                metrics['calmar_ratio']
            ]
        })
        metrics_df.to_csv('./artifacts/results/data/metrics.csv', index=False)
        
        # Guardar métricas por régimen
        regime_rows = []
        for regime, regime_data in metrics['regime_metrics'].items():
            regime_rows.append({
                'Regime': regime,
                'Days': regime_data['days'],
                'Annual Return': regime_data['return'],
                'Annual Volatility': regime_data['volatility'],
                'Sharpe Ratio': regime_data['sharpe']
            })
        
        if regime_rows:
            regime_df = pd.DataFrame(regime_rows)
            regime_df.to_csv('./artifacts/results/data/regime_metrics.csv', index=False)
        
        # Crear gráficos
        self.create_performance_plots(performance, transactions_df, positions_df)
        
    def create_performance_plots(self, performance, transactions, position_history):
        """Crear gráficos de rendimiento"""
        # Estilo de gráficos
        plt.style.use('seaborn-darkgrid')
        
        # 1. Gráfico de rendimiento acumulado
        plt.figure(figsize=(12, 6))
        
        # Separar por régimen
        for regime in [1, 2, 3]:
            regime_data = performance[performance['regime'] == regime]
            if not regime_data.empty:
                plt.plot(regime_data.index, regime_data['cumulative_return'], 
                         label=f'Regime {regime}', alpha=0.8)
        
        # Línea principal
        plt.plot(performance.index, performance['cumulative_return'], 'k--', 
                 label='Cumulative Return', linewidth=1.0)
        
        # Añadir transacciones si hay datos
        if not isinstance(transactions, pd.DataFrame) or transactions.empty:
            pass  # Sin transacciones para mostrar
        else:
            try:
                for _, t in transactions.iterrows():
                    if 'date' in t and t['date'] in performance.index:
                        date = pd.to_datetime(t['date'])
                        if date in performance.index:
                            cum_ret = performance.loc[date, 'cumulative_return']
                            if t['action'] == 'OPEN':
                                marker = '^' if t['direction'] == 'long' else 'v'
                                color = 'g' if t['direction'] == 'long' else 'r'
                                plt.scatter(date, cum_ret, marker=marker, color=color, s=50, alpha=0.7)
                            elif t['action'] == 'CLOSE':
                                plt.scatter(date, cum_ret, marker='o', color='blue', s=30, alpha=0.5)
            except Exception as e:
                logging.error(f"Error al añadir transacciones al gráfico: {str(e)}")
        
        plt.title('Cumulative Return by Market Regime')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.tight_layout()
        plt.savefig('./artifacts/results/figures/cumulative_return.png')
        plt.close()
        
        # 2. Drawdown
        plt.figure(figsize=(12, 4))
        cum_returns = performance['cumulative_return'].values
        expanding_max = np.maximum.accumulate(cum_returns)
        drawdowns = 1 - cum_returns / expanding_max
        
        plt.fill_between(performance.index, 0, drawdowns, color='red', alpha=0.3)
        plt.title('Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.tight_layout()
        plt.savefig('./artifacts/results/figures/drawdown.png')
        plt.close()
        
        # 3. Número de posiciones vs régimen
        if isinstance(position_history, pd.DataFrame) and not position_history.empty:
            try:
                positions_df = position_history.copy()
                if 'date' in positions_df.columns:
                    positions_df['date'] = pd.to_datetime(positions_df['date'])
                    positions_df.set_index('date', inplace=True)
                    
                    # Rellenar fechas faltantes
                    if not positions_df.empty:
                        full_idx = pd.date_range(positions_df.index.min(), positions_df.index.max(), freq='B')
                        positions_df = positions_df.reindex(full_idx, method='ffill')
                        
                        fig, ax1 = plt.subplots(figsize=(12, 5))
                        
                        # Número de posiciones
                        ax1.plot(positions_df.index, positions_df['n_positions'], 'b-', label='Number of Positions')
                        ax1.set_xlabel('Date')
                        ax1.set_ylabel('Number of Positions', color='b')
                        ax1.tick_params(axis='y', labelcolor='b')
                        
                        # Régimen
                        ax2 = ax1.twinx()
                        ax2.plot(positions_df.index, positions_df['regime'], 'r--', label='Market Regime')
                        ax2.set_ylabel('Market Regime', color='r')
                        ax2.tick_params(axis='y', labelcolor='r')
                        
                        # Líneas de referencia para regímenes
                        for regime in [1, 2, 3]:
                            plt.axhline(y=regime, color='gray', linestyle='--', alpha=0.3)
                        
                        fig.tight_layout()
                        plt.title('Number of Positions vs Market Regime')
                        plt.savefig('./artifacts/results/figures/positions_vs_regime.png')
                        plt.close()
            except Exception as e:
                logging.error(f"Error al crear gráfico de posiciones vs régimen: {str(e)}")
            
        # 4. Rendimientos por régimen
        plt.figure(figsize=(12, 5))
        
        colors = ['green', 'blue', 'red']
        
        for i, regime in enumerate([1, 2, 3]):
            regime_data = performance[performance['regime'] == regime]
            if not regime_data.empty:
                plt.hist(regime_data['return'], bins=30, alpha=0.5, 
                         label=f'Regime {regime}', color=colors[i])
        
        plt.axvline(x=0, color='black', linestyle='--')
        plt.title('Return Distribution by Regime')
        plt.xlabel('Daily Return')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig('./artifacts/results/figures/return_distribution.png')
        plt.close()
        
    def walkforward_test(self, start_date, end_date, train_days=252, test_days=63):
        """Realizar test walk-forward con ventanas deslizantes"""
        print(f"Iniciando walk-forward test desde {start_date} hasta {end_date}...")
        
        # Convertir fechas a datetime
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Crear ventanas de train-test
        current_train_start = start_dt
        results = []
        
        while current_train_start + pd.Timedelta(days=train_days) < end_dt:
            # Definir ventanas
            train_end = current_train_start + pd.Timedelta(days=train_days)
            test_end = min(train_end + pd.Timedelta(days=test_days), end_dt)
            
            # Formatear fechas
            train_start_str = current_train_start.strftime('%Y-%m-%d')
            train_end_str = train_end.strftime('%Y-%m-%d')
            test_end_str = test_end.strftime('%Y-%m-%d')
            
            print(f"\nVentana: Train {train_start_str} a {train_end_str}, Test hasta {test_end_str}")
            
            # Reiniciar estrategia para esta ventana
            self.active_pairs = []
            self.positions = {}
            self.quarantine_pairs = {}
            
            # Inicializar con datos de entrenamiento
            self.initialize(train_end_str)
            
            # Backtest en periodo de test
            metrics, performance = self.backtest(train_end_str, test_end_str)
            
            # Guardar resultados de esta ventana
            window_result = {
                'train_start': train_start_str,
                'train_end': train_end_str,
                'test_end': test_end_str,
                'sharpe': metrics['sharpe_ratio'],
                'return': metrics['total_return'],
                'max_drawdown': metrics['max_drawdown']
            }
            results.append(window_result)
            
            # Avanzar ventana (50% de solapamiento)
            current_train_start = current_train_start + pd.Timedelta(days=train_days//2)
        
        # Consolidar resultados
        results_df = pd.DataFrame(results)
        results_df.to_csv('./artifacts/results/data/walkforward_results.csv', index=False)
        
        # Visualizar resultados
        if not results_df.empty:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(3, 1, 1)
            plt.plot(results_df.index, results_df['sharpe'], 'o-')
            plt.title('Sharpe Ratio por Ventana')
            plt.grid(True)
            
            plt.subplot(3, 1, 2)
            plt.plot(results_df.index, results_df['return'], 'o-')
            plt.title('Retorno por Ventana')
            plt.grid(True)
            
            plt.subplot(3, 1, 3)
            plt.plot(results_df.index, results_df['max_drawdown'], 'o-')
            plt.title('Máximo Drawdown por Ventana')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('./artifacts/results/figures/walkforward_performance.png')
            plt.close()
        
        return results_df


def main():
    try:
        print("Iniciando sistema de arbitraje estadístico bayesiano adaptativo...")
        
        # Crear gestor de datos
        data_manager = DataManager(use_cache=True)
        
        # Crear estrategia
        strategy = StatisticalArbitrageStrategy(data_manager)
        
        # Configurar fechas para backtest (últimos 3 años)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
        
        # Ejecutar backtest
        metrics, performance = strategy.backtest(start_date, end_date)
        
        # Mostrar métricas
        print("\nResultados del backtest:")
        print(f"Rendimiento total: {metrics['total_return']:.2%}")
        print(f"Rendimiento anualizado: {metrics['annual_return']:.2%}")
        print(f"Volatilidad anualizada: {metrics['annual_volatility']:.2%}")
        print(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Máximo drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Calmar ratio: {metrics['calmar_ratio']:.2f}")
        
        # Rendimiento por régimen
        print("\nRendimiento por régimen:")
        for regime, data in metrics['regime_metrics'].items():
            print(f"  Régimen {regime}: {data['days']} días, Sharpe: {data['sharpe']:.2f}")
        
        # Realizar test walkforward con ventana más corta
        walkforward_start = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
        walkforward_results = strategy.walkforward_test(walkforward_start, end_date, train_days=180, test_days=45)
        
        print("\nResultados de walk-forward test guardados.")
        print("\nTodos los resultados han sido guardados en ./artifacts/results/")
        
    except Exception as e:
        logging.error(f"Error en la ejecución principal: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        print(f"Error en la ejecución. Ver ./artifacts/errors.txt para detalles.")

if __name__ == "__main__":
    main()
