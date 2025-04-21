import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from statsmodels.tsa.stattools import coint, adfuller
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats
from datetime import datetime, timedelta
import warnings
from tqdm import tqdm
import itertools
import pickle
from scipy.optimize import minimize
import matplotlib.dates as mdates
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

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

class StatisticalArbitrageStrategy:
    def __init__(self, start_date='2010-01-01', end_date=None, universe_size=100, 
                 max_active_pairs=20, trading_params=None):
        """
        Inicializa la estrategia de arbitraje estadístico multi-régimen.
        
        Args:
            start_date (str): Fecha de inicio para el análisis
            end_date (str): Fecha de fin para el análisis (None = hoy)
            universe_size (int): Número de acciones a considerar del S&P 500
            max_active_pairs (int): Número máximo de pares activos simultáneamente
            trading_params (dict): Parámetros de trading
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date) if end_date else pd.to_datetime(datetime.now().date())
        self.universe_size = universe_size
        self.max_active_pairs = max_active_pairs
        
        # Parámetros de trading por defecto
        self.trading_params = trading_params or {
            'z_entry': 2.0,       # Umbral de entrada
            'z_exit': 0.0,        # Umbral de salida
            'max_holding_period': 20,  # Período máximo de tenencia (días)
            'transaction_cost': 0.0005,  # Costo de transacción (5 bps)
            'lookback_short': 5,   # Ventana corta para z-score
            'lookback_medium': 21, # Ventana media para z-score
            'lookback_long': 63,   # Ventana larga para z-score
            'regime_lookback': 126, # Ventana para detección de régimen
            'pair_lookback': 252,  # Ventana para selección de pares
            'rebalance_frequency': 21, # Frecuencia de rebalanceo (días)
            'coint_threshold': 0.05, # Umbral para test de cointegración
            'min_half_life': 5,    # Half-life mínimo para considerar un par
            'max_half_life': 126,  # Half-life máximo para considerar un par
        }
        
        # Variables de estado
        self.universe = None
        self.price_data = None
        self.active_pairs = {}
        self.pair_history = {}
        self.regime_model = None
        self.current_regime = None
        self.regime_history = {}
        self.performance_metrics = {}
        
    def load_data(self):
        """
        Carga los datos de precios del universo de acciones del S&P 500.
        """
        try:
            # Obtener lista de símbolos del S&P 500 desde Wikipedia
            sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            sp500_table = pd.read_html(sp500_url)
            sp500_symbols = sp500_table[0]['Symbol'].tolist()
            
            # Filtrar símbolos con caracteres especiales
            sp500_symbols = [s.replace('.', '-') for s in sp500_symbols if '/' not in s]
            
            # Seleccionar un subconjunto aleatorio si es necesario
            if self.universe_size < len(sp500_symbols):
                np.random.seed(42)  # Para reproducibilidad
                self.universe = np.random.choice(sp500_symbols, self.universe_size, replace=False)
            else:
                self.universe = sp500_symbols
            
            # Descargar datos de precios
            self.price_data = yf.download(
                self.universe, 
                start=self.start_date - pd.Timedelta(days=365),  # Datos adicionales para cálculos iniciales
                end=self.end_date,
                progress=False
            )['Close']
            
            # Manejar valores faltantes
            self.price_data = self.price_data.ffill().bfill()
            
            # Filtrar acciones con demasiados valores faltantes
            missing_pct = self.price_data.isna().mean()
            valid_stocks = missing_pct[missing_pct < 0.1].index.tolist()
            self.price_data = self.price_data[valid_stocks]
            self.universe = valid_stocks
            
            # Guardar lista de símbolos
            pd.Series(self.universe).to_csv('./artifacts/results/data/universe.csv', index=False)
            
            print(f"Datos cargados para {len(self.universe)} acciones desde {self.start_date} hasta {self.end_date}")
            return True
            
        except Exception as e:
            logging.error(f"Error al cargar datos: {str(e)}", exc_info=True)
            return False
    
    def calculate_market_features(self, current_date):
        """
        Calcula características del mercado para la detección de regímenes.
        
        Args:
            current_date (datetime): Fecha actual para el análisis
        
        Returns:
            pd.DataFrame: Características del mercado
        """
        try:
            # Filtrar datos hasta la fecha actual
            cutoff_date = pd.to_datetime(current_date)
            lookback = self.trading_params['regime_lookback']
            
            # Asegurar que tenemos suficientes datos
            start_date = cutoff_date - pd.Timedelta(days=lookback*2)
            
            # Descargar datos del SPY como proxy del mercado
            spy_data = yf.download('SPY', start=start_date, end=cutoff_date, progress=False)
            
            if spy_data.empty:
                raise ValueError(f"No se pudieron obtener datos de SPY para {start_date} a {cutoff_date}")
            
            # Calcular retornos
            spy_data['returns'] = spy_data['Close'].pct_change()
            
            # Calcular características
            features = pd.DataFrame(index=[cutoff_date])
            
            # Volatilidad (diferentes ventanas)
            features['volatility_10d'] = spy_data['returns'].rolling(10).std().iloc[-1] * np.sqrt(252)
            features['volatility_30d'] = spy_data['returns'].rolling(30).std().iloc[-1] * np.sqrt(252)
            features['volatility_60d'] = spy_data['returns'].rolling(60).std().iloc[-1] * np.sqrt(252)
            
            # Momentum (diferentes ventanas)
            features['momentum_10d'] = spy_data['Close'].pct_change(10).iloc[-1]
            features['momentum_30d'] = spy_data['Close'].pct_change(30).iloc[-1]
            features['momentum_60d'] = spy_data['Close'].pct_change(60).iloc[-1]
            
            # Volatilidad relativa (ratio de volatilidades)
            features['rel_vol_10_30'] = features['volatility_10d'] / features['volatility_30d']
            features['rel_vol_10_60'] = features['volatility_10d'] / features['volatility_60d']
            
            # Drawdown
            rolling_max = spy_data['Close'].rolling(60, min_periods=1).max()
            drawdown = (spy_data['Close'] / rolling_max - 1.0)
            features['max_drawdown_60d'] = drawdown.rolling(60).min().iloc[-1]
            
            # Asimetría y curtosis
            features['skewness_30d'] = spy_data['returns'].rolling(30).skew().iloc[-1]
            features['kurtosis_30d'] = spy_data['returns'].rolling(30).kurt().iloc[-1]
            
            # Manejar valores faltantes
            features = features.fillna(method='ffill').fillna(0)
            
            return features
            
        except Exception as e:
            logging.error(f"Error al calcular características del mercado: {str(e)}", exc_info=True)
            return pd.DataFrame()
    
    def detect_market_regime(self, current_date, i=0):
        """
        Detecta el régimen de mercado actual utilizando clustering.
        
        Args:
            current_date (datetime): Fecha actual para el análisis
            i (int): Índice de iteración para actualización periódica
        
        Returns:
            int: Identificador del régimen actual
        """
        try:
            # Calcular características del mercado
            features = self.calculate_market_features(current_date)
            
            if features.empty:
                return 1  # Régimen neutral por defecto
            
            # Escalar características
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Actualizar modelo periódicamente o inicializarlo
            if self.regime_model is None or i % 63 == 0:  # Actualizar cada ~3 meses
                # Entrenar modelo de clustering
                self.regime_model = KMeans(n_clusters=3, random_state=42)
                self.regime_model.fit(scaled_features)
            
            # Predecir régimen
            regime = self.regime_model.predict(scaled_features)[0]
            
            # Guardar historial de regímenes
            self.regime_history[current_date] = int(regime)
            
            return int(regime)
            
        except Exception as e:
            logging.error(f"Error al detectar régimen de mercado: {str(e)}", exc_info=True)
            return 1  # Régimen neutral por defecto
    
    def find_cointegrated_pairs(self, current_date):
        """
        Encuentra pares cointegrados utilizando datos históricos hasta la fecha actual.
        
        Args:
            current_date (datetime): Fecha actual para el análisis
        
        Returns:
            list: Lista de pares cointegrados con sus estadísticas
        """
        try:
            # Implementar separación temporal estricta
            selection_end = current_date - pd.Timedelta(days=63)
            validation_start = selection_end
            validation_end = current_date
            
            # Filtrar datos para selección y validación
            selection_data = self.price_data.loc[:selection_end].copy()
            validation_data = self.price_data.loc[validation_start:validation_end].copy()
            
            # Asegurar suficientes datos
            min_samples = self.trading_params['pair_lookback']
            if len(selection_data) < min_samples:
                return []
            
            # Usar solo los últimos datos para selección
            selection_data = selection_data.iloc[-min_samples:]
            
            # Calcular retornos logarítmicos para selección
            log_prices = np.log(selection_data)
            
            # Lista para almacenar pares cointegrados
            cointegrated_pairs = []
            
            # Obtener lista de símbolos con datos suficientes
            valid_symbols = selection_data.columns[selection_data.isna().sum() < min_samples * 0.1].tolist()
            
            # Limitar número de combinaciones para eficiencia
            if len(valid_symbols) > 50:
                np.random.seed(42 + pd.Timestamp(current_date).dayofyear)
                valid_symbols = np.random.choice(valid_symbols, 50, replace=False)
            
            # Probar todas las combinaciones de pares
            for ticker1, ticker2 in itertools.combinations(valid_symbols, 2):
                # Verificar datos suficientes
                pair_data = log_prices[[ticker1, ticker2]].dropna()
                if len(pair_data) < min_samples * 0.9:
                    continue
                
                # Test de cointegración
                score, pvalue, _ = coint(pair_data[ticker1], pair_data[ticker2])
                
                if pvalue < self.trading_params['coint_threshold']:
                    # Calcular beta (ratio de cointegración)
                    model = OLS(pair_data[ticker1], sm.add_constant(pair_data[ticker2]))
                    results = model.fit()
                    beta = results.params[1]
                    
                    # Calcular spread
                    spread = pair_data[ticker1] - beta * pair_data[ticker2]
                    
                    # Calcular half-life de reversión a la media
                    half_life = self._calculate_half_life(spread)
                    
                    # Filtrar por half-life
                    min_hl = self.trading_params['min_half_life']
                    max_hl = self.trading_params['max_half_life']
                    
                    if min_hl <= half_life <= max_hl:
                        # Validar en datos recientes
                        valid = self._validate_pair(ticker1, ticker2, beta, validation_data)
                        
                        if valid:
                            # Calcular métricas adicionales
                            spread_mean = spread.mean()
                            spread_std = spread.std()
                            sharpe = self._calculate_pair_sharpe(ticker1, ticker2, beta, validation_data)
                            
                            cointegrated_pairs.append({
                                'ticker1': ticker1,
                                'ticker2': ticker2,
                                'beta': beta,
                                'half_life': half_life,
                                'pvalue': pvalue,
                                'spread_mean': spread_mean,
                                'spread_std': spread_std,
                                'sharpe': sharpe
                            })
            
            # Ordenar pares por sharpe ratio y half-life
            if cointegrated_pairs:
                cointegrated_pairs.sort(key=lambda x: (-x['sharpe'], x['half_life']))
            
            return cointegrated_pairs
            
        except Exception as e:
            logging.error(f"Error al encontrar pares cointegrados: {str(e)}", exc_info=True)
            return []
    
    def _calculate_half_life(self, spread):
        """
        Calcula el half-life de reversión a la media para un spread.
        
        Args:
            spread (pd.Series): Serie temporal del spread
        
        Returns:
            float: Half-life en días
        """
        try:
            # Calcular cambios en el spread
            spread_lag = spread.shift(1)
            delta_spread = spread - spread_lag
            
            # Eliminar NaN
            spread_lag = spread_lag.dropna()
            delta_spread = delta_spread.dropna()
            
            # Regresión para estimar velocidad de reversión
            model = OLS(delta_spread, sm.add_constant(spread_lag))
            results = model.fit()
            
            # Extraer coeficiente gamma
            gamma = results.params[1]
            
            # Mejorar manejo de casos límite
            if gamma >= 0 or gamma < -1:  # No hay reversión a la media o reversión extrema
                return 126  # Valor máximo predeterminado
            
            # Calcular half-life
            half_life = -np.log(2) / gamma
            
            return max(1, min(126, half_life))  # Limitar entre 1 y 126 días
            
        except Exception as e:
            logging.error(f"Error al calcular half-life: {str(e)}", exc_info=True)
            return 126  # Valor por defecto
    
    def _validate_pair(self, ticker1, ticker2, beta, validation_data):
        """
        Valida un par en datos recientes.
        
        Args:
            ticker1 (str): Primer ticker
            ticker2 (str): Segundo ticker
            beta (float): Coeficiente de cointegración
            validation_data (pd.DataFrame): Datos para validación
        
        Returns:
            bool: True si el par es válido, False en caso contrario
        """
        try:
            # Verificar datos suficientes
            pair_data = validation_data[[ticker1, ticker2]].dropna()
            if len(pair_data) < 10:  # Mínimo 10 días para validación
                return False
            
            # Calcular spread en datos de validación
            log_prices = np.log(pair_data)
            spread = log_prices[ticker1] - beta * log_prices[ticker2]
            
            # Test de estacionariedad en datos de validación
            adf_result = adfuller(spread)
            
            # Verificar estacionariedad
            if adf_result[1] > 0.1:  # p-valor mayor a 0.1
                return False
            
            # Verificar volatilidad del spread
            spread_std = spread.std()
            if spread_std == 0 or np.isnan(spread_std):
                return False
            
            # Verificar cruces por la media
            mean_crossings = ((spread.shift(1) - spread.mean()) * 
                             (spread - spread.mean()) < 0).sum()
            if mean_crossings < 3:  # Menos de 3 cruces
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error al validar par: {str(e)}", exc_info=True)
            return False
    
    def _calculate_pair_sharpe(self, ticker1, ticker2, beta, data):
        """
        Calcula el Sharpe ratio para un par.
        
        Args:
            ticker1 (str): Primer ticker
            ticker2 (str): Segundo ticker
            beta (float): Coeficiente de cointegración
            data (pd.DataFrame): Datos de precios
        
        Returns:
            float: Sharpe ratio anualizado
        """
        try:
            # Verificar datos suficientes
            pair_data = data[[ticker1, ticker2]].dropna()
            if len(pair_data) < 10:
                return 0
            
            # Calcular retornos diarios
            returns1 = pair_data[ticker1].pct_change()
            returns2 = pair_data[ticker2].pct_change()
            
            # Calcular spread
            log_prices = np.log(pair_data)
            spread = log_prices[ticker1] - beta * log_prices[ticker2]
            
            # Calcular z-score
            z_score = (spread - spread.rolling(window=10).mean()) / spread.rolling(window=10).std()
            z_score = z_score.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Generar señales
            position = np.zeros(len(z_score))
            position[z_score < -2] = 1    # Comprar spread
            position[z_score > 2] = -1    # Vender spread
            
            # Calcular retornos de la estrategia
            pair_return = pd.Series(position[:-1]) * (returns1.values[1:] - beta * returns2.values[1:])
            
            # Calcular Sharpe ratio
            if len(pair_return) < 2 or pair_return.std() == 0:
                return 0
                
            sharpe = (pair_return.mean() / pair_return.std()) * np.sqrt(252)
            
            return sharpe
            
        except Exception as e:
            logging.error(f"Error al calcular Sharpe ratio: {str(e)}", exc_info=True)
            return 0
    
    def generate_signals(self, ticker1, ticker2, beta, price_data, current_date, lookback=None):
        """
        Genera señales de trading para un par.
        
        Args:
            ticker1 (str): Primer ticker
            ticker2 (str): Segundo ticker
            beta (float): Coeficiente de cointegración
            price_data (dict): Datos de precios
            current_date (datetime): Fecha actual
            lookback (int): Período de lookback
        
        Returns:
            pd.DataFrame: DataFrame con señales
        """
        try:
            # Configurar lookback
            if lookback is None:
                lookback = self.trading_params['pair_lookback']
            
            # Filtrar datos hasta la fecha actual
            cutoff_date = pd.to_datetime(current_date)
            
            # Extraer precios
            prices1 = price_data[ticker1].copy()
            prices2 = price_data[ticker2].copy()
            
            # Calcular retornos
            returns1 = prices1.pct_change()
            returns2 = prices2.pct_change()
            
            # Calcular spread logarítmico
            log_prices1 = np.log(prices1)
            log_prices2 = np.log(prices2)
            spread = log_prices1 - beta * log_prices2
            
            # Calcular z-scores para diferentes horizontes
            lookback_short = self.trading_params['lookback_short']
            lookback_medium = self.trading_params['lookback_medium']
            lookback_long = self.trading_params['lookback_long']
            
            # Z-score de corto plazo
            z_short = (spread - spread.rolling(window=lookback_short).mean()) / spread.rolling(window=lookback_short).std()
            z_short = z_short.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Z-score de medio plazo
            z_medium = (spread - spread.rolling(window=lookback_medium).mean()) / spread.rolling(window=lookback_medium).std()
            z_medium = z_medium.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Z-score de largo plazo
            z_long = (spread - spread.rolling(window=lookback_long).mean()) / spread.rolling(window=lookback_long).std()
            z_long = z_long.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Combinar z-scores (ponderación adaptativa)
            z_score = 0.5 * z_short + 0.3 * z_medium + 0.2 * z_long
            
            # Crear DataFrame de señales
            signals = pd.DataFrame(index=spread.index)
            signals['spread'] = spread
            signals['z_score'] = z_score
            signals['z_short'] = z_short
            signals['z_medium'] = z_medium
            signals['z_long'] = z_long
            signals['return1'] = returns1
            signals['return2'] = returns2
            
            # Implementar señal condicional no lineal
            signals['signal_intensity'] = signals['z_score'].copy()
            
            # Ajuste condicional basado en concordancia de señales
            for i in range(1, len(signals)):
                if abs(signals['z_short'].iloc[i]) > 2 and np.sign(signals['z_short'].iloc[i]) == np.sign(signals['z_medium'].iloc[i]):
                    signals.loc[signals.index[i], 'signal_intensity'] = 1.5 * signals['z_score'].iloc[i]
                elif np.sign(signals['z_short'].iloc[i]) != np.sign(signals['z_medium'].iloc[i]):
                    signals.loc[signals.index[i], 'signal_intensity'] = 0.5 * signals['z_score'].iloc[i]
            
            # Calcular SNR para filtrar señales de baja calidad
            signals['signal_mean'] = signals['z_score'].rolling(21).apply(lambda x: np.abs(x).mean())
            signals['signal_std'] = signals['z_score'].rolling(21).std()
            signals['snr'] = signals['signal_mean'] / signals['signal_std'].replace(0, np.nan)
            signals['snr'] = signals['snr'].fillna(0)
            
            # Generar posiciones
            signals['position'] = 0.0
            
            # Umbral de entrada y salida
            z_entry = self.trading_params['z_entry']
            z_exit = self.trading_params['z_exit']
            
            # Inicializar posición
            position = 0
            entry_date = None
            
            # Generar señales
            for i in range(1, len(signals)):
                date = signals.index[i]
                
                # Filtrar por SNR
                snr_threshold = 0.5
                snr_valid = signals['snr'].iloc[i] > snr_threshold
                
                # Lógica de entrada y salida
                if position == 0:  # Sin posición
                    if signals['signal_intensity'].iloc[i] < -z_entry and snr_valid:
                        position = 1  # Comprar spread (long ticker1, short ticker2)
                        entry_date = date
                    elif signals['signal_intensity'].iloc[i] > z_entry and snr_valid:
                        position = -1  # Vender spread (short ticker1, long ticker2)
                        entry_date = date
                
                elif position == 1:  # Posición larga en spread
                    # Salir si el z-score cruza el umbral de salida o se alcanza el período máximo
                    days_held = (date - entry_date).days if entry_date else 0
                    if (signals['signal_intensity'].iloc[i] >= z_exit or 
                        days_held > self.trading_params['max_holding_period']):
                        position = 0
                        entry_date = None
                
                elif position == -1:  # Posición corta en spread
                    # Salir si el z-score cruza el umbral de salida o se alcanza el período máximo
                    days_held = (date - entry_date).days if entry_date else 0
                    if (signals['signal_intensity'].iloc[i] <= -z_exit or 
                        days_held > self.trading_params['max_holding_period']):
                        position = 0
                        entry_date = None
                
                signals.loc[date, 'position'] = position
            
            # Calcular retornos de la estrategia
            signals['pair_return'] = signals['position'].shift(1) * (
                signals['return1'] - beta * signals['return2']
            )
            
            # Calcular retornos acumulados
            signals['cumulative_return'] = (1 + signals['pair_return']).cumprod() - 1
            
            return signals
            
        except Exception as e:
            logging.error(f"Error al generar señales: {str(e)}", exc_info=True)
            return pd.DataFrame()
    
    def update_trading_parameters(self, current_regime, previous_regime, performance_history):
        """
        Actualiza los parámetros de trading según el régimen actual.
        
        Args:
            current_regime (int): Régimen actual
            previous_regime (int): Régimen anterior
            performance_history (dict): Historial de rendimiento
        
        Returns:
            dict: Parámetros actualizados
        """
        try:
            # Parámetros específicos por régimen
            regime_params = {
                0: {'z_entry': 2.5, 'z_exit': 0.5, 'max_holding_period': 15},  # Régimen volátil: más conservador
                1: {'z_entry': 2.0, 'z_exit': 0.0, 'max_holding_period': 20},  # Régimen neutral
                2: {'z_entry': 1.5, 'z_exit': 0.0, 'max_holding_period': 25}   # Régimen estable: más agresivo
            }
            
            # Calcular probabilidad de cambio de régimen
            p_change = abs(current_regime - previous_regime) / 2  # Simplificación
            
            # Actualizar parámetros gradualmente
            updated_params = self.trading_params.copy()
            
            # Aplicar parámetros del régimen actual
            for param, value in regime_params[current_regime].items():
                # Interpolación lineal entre parámetros actuales y nuevos
                current_value = updated_params[param]
                target_value = value
                updated_params[param] = current_value * (1 - p_change) + target_value * p_change
            
            # Ajustar según rendimiento reciente si hay suficientes datos
            if len(performance_history) > 10:
                recent_returns = list(performance_history.values())[-10:]
                recent_sharpe = np.mean(recent_returns) / (np.std(recent_returns) + 1e-6) * np.sqrt(252)
                
                # Ajustar agresividad según Sharpe ratio reciente
                if recent_sharpe < 0.5:
                    # Más conservador
                    updated_params['z_entry'] = min(3.0, updated_params['z_entry'] * 1.1)
                    updated_params['max_holding_period'] = max(10, updated_params['max_holding_period'] * 0.9)
                elif recent_sharpe > 1.5:
                    # Más agresivo
                    updated_params['z_entry'] = max(1.5, updated_params['z_entry'] * 0.9)
                    updated_params['max_holding_period'] = min(30, updated_params['max_holding_period'] * 1.1)
            
            return updated_params
            
        except Exception as e:
            logging.error(f"Error al actualizar parámetros: {str(e)}", exc_info=True)
            return self.trading_params
    
    def manage_portfolio(self, current_date, active_pairs, available_pairs):
        """
        Gestiona la cartera de pares activos.
        
        Args:
            current_date (datetime): Fecha actual
            active_pairs (dict): Pares actualmente en cartera
            available_pairs (list): Nuevos pares disponibles
        
        Returns:
            dict: Pares actualizados en cartera
        """
        try:
            updated_portfolio = {}
            
            # Evaluar pares activos
            for pair_id, pair_info in active_pairs.items():
                ticker1, ticker2 = pair_info['ticker1'], pair_info['ticker2']
                beta = pair_info['beta']
                entry_date = pair_info['entry_date']
                
                # Verificar si debemos mantener el par
                days_held = (current_date - entry_date).days
                
                # Obtener señales actualizadas
                signals = self.generate_signals(
                    ticker1, ticker2, beta, 
                    {ticker1: self.price_data[ticker1], ticker2: self.price_data[ticker2]},
                    current_date
                )
                
                if signals.empty:
                    continue
                
                current_position = signals['position'].iloc[-1]
                current_z_score = signals['z_score'].iloc[-1]
                
                # Decidir si mantener o cerrar la posición
                if current_position != 0:
                    # Mantener par activo
                    updated_portfolio[pair_id] = pair_info
                    updated_portfolio[pair_id]['current_position'] = current_position
                    updated_portfolio[pair_id]['current_z_score'] = current_z_score
                    updated_portfolio[pair_id]['days_held'] = days_held
                    
                    # Actualizar rendimiento
                    if 'cumulative_return' in signals.columns and not signals['cumulative_return'].empty:
                        updated_portfolio[pair_id]['current_return'] = signals['cumulative_return'].iloc[-1]
                
            # Añadir nuevos pares si hay espacio
            remaining_slots = self.max_active_pairs - len(updated_portfolio)
            
            if remaining_slots > 0 and available_pairs:
                # Ordenar pares disponibles por potencial
                sorted_pairs = sorted(available_pairs, key=lambda x: (-x['sharpe'], x['half_life']))
                
                # Añadir nuevos pares
                for i, pair in enumerate(sorted_pairs[:remaining_slots]):
                    pair_id = f"{pair['ticker1']}_{pair['ticker2']}_{current_date.strftime('%Y%m%d')}"
                    
                    # Generar señales iniciales
                    signals = self.generate_signals(
                        pair['ticker1'], pair['ticker2'], pair['beta'],
                        {pair['ticker1']: self.price_data[pair['ticker1']], 
                         pair['ticker2']: self.price_data[pair['ticker2']]},
                        current_date
                    )
                    
                    if signals.empty:
                        continue
                    
                    current_position = signals['position'].iloc[-1]
                    current_z_score = signals['z_score'].iloc[-1]
                    
                    # Solo añadir si hay señal activa
                    if current_position != 0:
                        updated_portfolio[pair_id] = {
                            'ticker1': pair['ticker1'],
                            'ticker2': pair['ticker2'],
                            'beta': pair['beta'],
                            'entry_date': current_date,
                            'current_position': current_position,
                            'current_z_score': current_z_score,
                            'days_held': 0,
                            'current_return': 0.0
                        }
            
            return updated_portfolio
            
        except Exception as e:
            logging.error(f"Error al gestionar cartera: {str(e)}", exc_info=True)
            return active_pairs
    
    def calculate_portfolio_returns(self, active_pairs, current_date, previous_date):
        """
        Calcula los retornos de la cartera para un día.
        
        Args:
            active_pairs (dict): Pares activos en cartera
            current_date (datetime): Fecha actual
            previous_date (datetime): Fecha anterior
        
        Returns:
            float: Retorno diario de la cartera
        """
        try:
            if not active_pairs:
                return 0.0
            
            daily_returns = []
            position_sizes = []
            
            for pair_id, pair_info in active_pairs.items():
                ticker1, ticker2 = pair_info['ticker1'], pair_info['ticker2']
                beta = pair_info['beta']
                position = pair_info['current_position']
                
                # Verificar que tenemos datos para ambas fechas
                if (previous_date not in self.price_data.index or 
                    current_date not in self.price_data.index):
                    continue
                
                # Calcular retornos diarios
                if ticker1 in self.price_data.columns and ticker2 in self.price_data.columns:
                    price1_prev = self.price_data.loc[previous_date, ticker1]
                    price1_curr = self.price_data.loc[current_date, ticker1]
                    price2_prev = self.price_data.loc[previous_date, ticker2]
                    price2_curr = self.price_data.loc[current_date, ticker2]
                    
                    # Verificar datos válidos
                    if (np.isnan(price1_prev) or np.isnan(price1_curr) or 
                        np.isnan(price2_prev) or np.isnan(price2_curr)):
                        continue
                    
                    # Calcular retornos
                    return1 = price1_curr / price1_prev - 1
                    return2 = price2_curr / price2_prev - 1
                    
                    # Calcular retorno del par según posición
                    pair_return = position * (return1 - beta * return2)
                    
                    # Ajustar tamaño de posición según intensidad de señal y liquidez
                    z_score = pair_info['current_z_score']
                    position_size = min(1.0, abs(z_score) / self.trading_params['z_entry'])
                    
                    daily_returns.append(pair_return)
                    position_sizes.append(position_size)
            
            # Calcular retorno ponderado de la cartera
            if not daily_returns:
                return 0.0
                
            if sum(position_sizes) > 0:
                weighted_return = sum(r * s for r, s in zip(daily_returns, position_sizes)) / sum(position_sizes)
            else:
                weighted_return = np.mean(daily_returns)
            
            # Aplicar costos de transacción
            # Implementar modelo de costos más realista
            spread_cost = 0.0005  # 5 bps de spread
            market_impact = 0.0010 * (len(active_pairs) > self.max_active_pairs / 2)  # 10 bps adicionales para cambios grandes
            transaction_costs = len(active_pairs) * (spread_cost + market_impact) / max(1, len(active_pairs))
            
            net_return = weighted_return - transaction_costs
            
            return net_return
            
        except Exception as e:
            logging.error(f"Error al calcular retornos: {str(e)}", exc_info=True)
            return 0.0
    
    def backtest(self):
        """
        Ejecuta el backtest de la estrategia.
        
        Returns:
            pd.DataFrame: Resultados del backtest
        """
        try:
            # Verificar datos
            if self.price_data is None:
                if not self.load_data():
                    return pd.DataFrame()
            
            # Inicializar resultados
            results = pd.DataFrame(index=self.price_data.index)
            results['return'] = 0.0
            results['equity'] = 1.0
            results['regime'] = np.nan
            results['active_pairs'] = 0
            
            # Inicializar variables
            active_pairs = {}
            previous_regime = 1  # Neutral por defecto
            performance_history = {}
            
            # Definir fecha de inicio efectiva (después del período de lookback)
            effective_start = self.start_date + pd.Timedelta(days=self.trading_params['pair_lookback'])
            
            # Filtrar fechas de trading
            trading_dates = self.price_data.index[self.price_data.index >= effective_start]
            
            # Ejecutar backtest
            for i, current_date in enumerate(tqdm(trading_dates)):
                # Saltar primer día (necesitamos día anterior para calcular retornos)
                if i == 0:
                    continue
                
                previous_date = trading_dates[i-1]
                
                # Detectar régimen de mercado
                current_regime = self.detect_market_regime(current_date, i)
                results.loc[current_date, 'regime'] = current_regime
                
                # Actualizar parámetros de trading
                self.trading_params = self.update_trading_parameters(
                    current_regime, previous_regime, performance_history
                )
                
                # Rebalancear cartera periódicamente
                if i % self.trading_params['rebalance_frequency'] == 0:
                    # Encontrar nuevos pares
                    available_pairs = self.find_cointegrated_pairs(current_date)
                    
                    # Gestionar cartera
                    active_pairs = self.manage_portfolio(current_date, active_pairs, available_pairs)
                
                # Calcular retornos diarios
                daily_return = self.calculate_portfolio_returns(active_pairs, current_date, previous_date)
                results.loc[current_date, 'return'] = daily_return
                
                # Actualizar equity
                if i > 0:
                    results.loc[current_date, 'equity'] = results.loc[previous_date, 'equity'] * (1 + daily_return)
                
                # Guardar número de pares activos
                results.loc[current_date, 'active_pairs'] = len(active_pairs)
                
                # Actualizar historial de rendimiento
                performance_history[current_date] = daily_return
                
                # Actualizar régimen anterior
                previous_regime = current_regime
            
            # Calcular métricas de rendimiento
            self.calculate_performance_metrics(results)
            
            # Guardar resultados
            results.to_csv('./artifacts/results/data/backtest_results.csv')
            
            # Generar gráficos
            self.plot_results(results)
            
            return results
            
        except Exception as e:
            logging.error(f"Error en backtest: {str(e)}", exc_info=True)
            return pd.DataFrame()
    
    def walk_forward_analysis(self, window_size=252, step_size=63):
        """
        Realiza análisis walk-forward para evaluar la robustez de la estrategia.
        
        Args:
            window_size (int): Tamaño de la ventana de análisis en días
            step_size (int): Tamaño del paso entre ventanas en días
        
        Returns:
            pd.DataFrame: Resultados del análisis walk-forward
        """
        try:
            # Verificar datos
            if self.price_data is None:
                if not self.load_data():
                    return pd.DataFrame()
            
            # Definir ventanas de análisis
            trading_dates = self.price_data.index[self.price_data.index >= self.start_date]
            
            if len(trading_dates) < window_size + step_size:
                raise ValueError("Datos insuficientes para análisis walk-forward")
            
            windows = []
            for i in range(0, len(trading_dates) - window_size, step_size):
                train_start = trading_dates[i]
                test_start = trading_dates[i + window_size - step_size]
                test_end = trading_dates[min(i + window_size, len(trading_dates) - 1)]
                
                windows.append({
                    'train_start': train_start,
                    'test_start': test_start,
                    'test_end': test_end
                })
            
            # Inicializar resultados
            wf_results = pd.DataFrame()
            
            # Ejecutar análisis para cada ventana
            for window in tqdm(windows):
                # Filtrar datos para la ventana actual
                window_data = self.price_data.loc[window['train_start']:window['test_end']]
                
                # Encontrar pares cointegrados usando datos de entrenamiento
                train_end = window['test_start'] - pd.Timedelta(days=1)
                cointegrated_pairs = self.find_cointegrated_pairs(train_end)
                
                # Evaluar pares en datos de prueba
                window_returns = []
                
                for pair in cointegrated_pairs[:self.max_active_pairs]:
                    ticker1, ticker2 = pair['ticker1'], pair['ticker2']
                    beta = pair['beta']
                    
                    # Verificar datos disponibles
                    if ticker1 not in window_data.columns or ticker2 not in window_data.columns:
                        continue
                    
                    price_data = {
                        ticker1: window_data[ticker1],
                        ticker2: window_data[ticker2]
                    }
                    
                    # Generar señales usando solo datos hasta test_start
                    signals = self.generate_signals(ticker1, ticker2, beta, 
                                                  {k: v[v.index <= window['test_start']] for k, v in price_data.items()},
                                                  window['test_start'], lookback=window_size)
                    
                    # Filtrar señales para período de prueba
                    test_signals = signals.loc[window['test_start']:window['test_end']]
                    
                    if not test_signals.empty and 'pair_return' in test_signals.columns:
                        window_returns.append(test_signals['pair_return'])
                
                # Calcular retorno combinado para la ventana
                if window_returns:
                    combined_return = pd.concat(window_returns, axis=1).mean(axis=1)
                    combined_return.name = 'return'
                    
                    # Añadir información de la ventana
                    combined_return = pd.DataFrame(combined_return)
                    combined_return['window_start'] = window['test_start']
                    combined_return['window_end'] = window['test_end']
                    
                    # Añadir a resultados globales
                    wf_results = pd.concat([wf_results, combined_return])
            
            # Calcular equity curve
            if not wf_results.empty:
                wf_results['equity'] = (1 + wf_results['return']).cumprod()
                
                # Guardar resultados
                wf_results.to_csv('./artifacts/results/data/walk_forward_results.csv')
                
                # Generar gráficos
                self.plot_walk_forward_results(wf_results)
            
            return wf_results
            
        except Exception as e:
            logging.error(f"Error en análisis walk-forward: {str(e)}", exc_info=True)
            return pd.DataFrame()
    
    def calculate_performance_metrics(self, results):
        """
        Calcula métricas de rendimiento para los resultados del backtest.
        
        Args:
            results (pd.DataFrame): Resultados del backtest
        """
        try:
            # Verificar datos suficientes
            if results.empty or 'return' not in results.columns:
                return
            
            # Calcular métricas básicas
            returns = results['return'].dropna()
            
            if len(returns) < 10:
                return
            
            # Retorno total
            total_return = results['equity'].iloc[-1] / results['equity'].iloc[0] - 1
            
            # Retorno anualizado
            years = len(returns) / 252
            annual_return = (1 + total_return) ** (1 / years) - 1
            
            # Volatilidad anualizada
            annual_vol = returns.std() * np.sqrt(252)
            
            # Sharpe ratio
            risk_free_rate = 0.02  # Tasa libre de riesgo (2%)
            sharpe_ratio = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            downside_vol = downside_returns.std() * np.sqrt(252)
            sortino_ratio = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
            
            # Máximo drawdown
            equity_curve = results['equity']
            rolling_max = equity_curve.cummax()
            drawdown = (equity_curve / rolling_max - 1)
            max_drawdown = drawdown.min()
            
            # Calmar ratio
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
            
            # Métricas por régimen
            regime_metrics = {}
            for regime in results['regime'].dropna().unique():
                regime_returns = returns[results['regime'] == regime]
                if len(regime_returns) > 10:
                    regime_metrics[int(regime)] = {
                        'return': regime_returns.mean() * 252,
                        'volatility': regime_returns.std() * np.sqrt(252),
                        'sharpe': regime_returns.mean() / regime_returns.std() * np.sqrt(252) if regime_returns.std() > 0 else 0,
                        'count': len(regime_returns)
                    }
            
            # Guardar métricas
            self.performance_metrics = {
                'total_return': total_return,
                'annual_return': annual_return,
                'annual_volatility': annual_vol,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'win_rate': len(returns[returns > 0]) / len(returns),
                'regime_metrics': regime_metrics
            }
            
            # Guardar métricas en archivo
            metrics_df = pd.DataFrame({k: [v] for k, v in self.performance_metrics.items() 
                                     if k != 'regime_metrics'})
            metrics_df.to_csv('./artifacts/results/data/performance_metrics.csv', index=False)
            
            # Guardar métricas por régimen
            if regime_metrics:
                regime_df = pd.DataFrame.from_dict(
                    {f"Regime {k}": v for k, v in regime_metrics.items()}, 
                    orient='index'
                )
                regime_df.to_csv('./artifacts/results/data/regime_metrics.csv')
            
            # Imprimir métricas
            print("\nMétricas de Rendimiento:")
            print(f"Retorno Total: {total_return:.2%}")
            print(f"Retorno Anualizado: {annual_return:.2%}")
            print(f"Volatilidad Anualizada: {annual_vol:.2%}")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"Sortino Ratio: {sortino_ratio:.2f}")
            print(f"Máximo Drawdown: {max_drawdown:.2%}")
            print(f"Calmar Ratio: {calmar_ratio:.2f}")
            
        except Exception as e:
            logging.error(f"Error al calcular métricas: {str(e)}", exc_info=True)
    
    def plot_results(self, results):
        """
        Genera gráficos para visualizar los resultados del backtest.
        
        Args:
            results (pd.DataFrame): Resultados del backtest
        """
        try:
            if results.empty:
                return
            
            # Configurar estilo
            plt.style.use('seaborn-v0_8-darkgrid')
            
            # 1. Equity curve
            plt.figure(figsize=(12, 6))
            plt.plot(results.index, results['equity'], linewidth=2)
            plt.title('Equity Curve', fontsize=14)
            plt.xlabel('Fecha')
            plt.ylabel('Equity')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('./artifacts/results/figures/equity_curve.png')
            plt.close()
            
            # 2. Drawdown
            equity_curve = results['equity']
            rolling_max = equity_curve.cummax()
            drawdown = (equity_curve / rolling_max - 1)
            
            plt.figure(figsize=(12, 6))
            plt.plot(results.index, drawdown, linewidth=2, color='red')
            plt.title('Drawdown', fontsize=14)
            plt.xlabel('Fecha')
            plt.ylabel('Drawdown')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('./artifacts/results/figures/drawdown.png')
            plt.close()
            
            # 3. Retornos mensuales
            if len(results) > 30:
                monthly_returns = results['return'].resample('M').apply(
                    lambda x: (1 + x).prod() - 1
                )
                
                plt.figure(figsize=(14, 7))
                monthly_returns.plot(kind='bar', color=np.where(monthly_returns >= 0, 'green', 'red'))
                plt.title('Retornos Mensuales', fontsize=14)
                plt.xlabel('Fecha')
                plt.ylabel('Retorno')
                plt.grid(True, axis='y')
                plt.tight_layout()
                plt.savefig('./artifacts/results/figures/monthly_returns.png')
                plt.close()
            
            # 4. Rendimiento por régimen
            if 'regime' in results.columns and not results['regime'].isna().all():
                plt.figure(figsize=(12, 6))
                
                for regime in sorted(results['regime'].dropna().unique()):
                    regime_equity = results[results['regime'] == regime]['equity']
                    if not regime_equity.empty:
                        # Normalizar equity para cada régimen
                        normalized_equity = regime_equity / regime_equity.iloc[0]
                        plt.plot(regime_equity.index, normalized_equity, 
                                label=f'Régimen {int(regime)}', linewidth=2)
                
                plt.title('Rendimiento por Régimen de Mercado', fontsize=14)
                plt.xlabel('Fecha')
                plt.ylabel('Equity Normalizada')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig('./artifacts/results/figures/regime_performance.png')
                plt.close()
            
            # 5. Número de pares activos
            if 'active_pairs' in results.columns:
                plt.figure(figsize=(12, 6))
                plt.plot(results.index, results['active_pairs'], linewidth=2, color='purple')
                plt.title('Número de Pares Activos', fontsize=14)
                plt.xlabel('Fecha')
                plt.ylabel('Pares Activos')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig('./artifacts/results/figures/active_pairs.png')
                plt.close()
            
            # 6. Distribución de retornos diarios
            plt.figure(figsize=(12, 6))
            sns.histplot(results['return'].dropna(), kde=True, bins=50)
            plt.title('Distribución de Retornos Diarios', fontsize=14)
            plt.xlabel('Retorno')
            plt.ylabel('Frecuencia')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('./artifacts/results/figures/return_distribution.png')
            plt.close()
            
        except Exception as e:
            logging.error(f"Error al generar gráficos: {str(e)}", exc_info=True)
    
    def plot_walk_forward_results(self, results):
        """
        Genera gráficos para visualizar los resultados del análisis walk-forward.
        
        Args:
            results (pd.DataFrame): Resultados del análisis walk-forward
        """
        try:
            if results.empty:
                return
            
            # Configurar estilo
            plt.style.use('seaborn-v0_8-darkgrid')
            
            # 1. Equity curve combinada
            plt.figure(figsize=(12, 6))
            plt.plot(results.index, results['equity'], linewidth=2)
            plt.title('Walk-Forward Equity Curve', fontsize=14)
            plt.xlabel('Fecha')
            plt.ylabel('Equity')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('./artifacts/results/figures/wf_equity_curve.png')
            plt.close()
            
            # 2. Retornos por ventana
            if 'window_start' in results.columns:
                window_groups = results.groupby('window_start')
                
                plt.figure(figsize=(14, 7))
                
                for window_start, group in window_groups:
                    window_equity = (1 + group['return']).cumprod()
                    plt.plot(group.index, window_equity, 
                            label=f'Ventana {window_start.strftime("%Y-%m-%d")}',
                            alpha=0.7)
                
                plt.title('Rendimiento por Ventana de Walk-Forward', fontsize=14)
                plt.xlabel('Fecha')
                plt.ylabel('Equity')
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                plt.grid(True)
                plt.tight_layout()
                plt.savefig('./artifacts/results/figures/wf_window_performance.png')
                plt.close()
            
            # 3. Distribución de retornos por ventana
            plt.figure(figsize=(12, 6))
            
            window_returns = []
            window_labels = []
            
            for window_start, group in results.groupby('window_start'):
                window_return = (1 + group['return']).prod() - 1
                window_returns.append(window_return)
                window_labels.append(window_start.strftime("%Y-%m-%d"))
            
            plt.bar(window_labels, window_returns, color=np.where(np.array(window_returns) >= 0, 'green', 'red'))
            plt.title('Retorno Total por Ventana', fontsize=14)
            plt.xlabel('Fecha de Inicio de Ventana')
            plt.ylabel('Retorno Total')
            plt.xticks(rotation=45)
            plt.grid(True, axis='y')
            plt.tight_layout()
            plt.savefig('./artifacts/results/figures/wf_window_returns.png')
            plt.close()
            
        except Exception as e:
            logging.error(f"Error al generar gráficos de walk-forward: {str(e)}", exc_info=True)

def main():
    """
    Función principal para ejecutar la estrategia.
    """
    try:
        print("Iniciando Estrategia de Arbitraje Estadístico Multi-Régimen")
        
        # Inicializar estrategia
        strategy = StatisticalArbitrageStrategy(
            start_date='2015-01-01',
            end_date='2023-12-31',
            universe_size=100,
            max_active_pairs=20
        )
        
        # Cargar datos
        if not strategy.load_data():
            print("Error al cargar datos. Verificar logs.")
            return
        
        # Ejecutar backtest
        print("\nEjecutando backtest...")
        backtest_results = strategy.backtest()
        
        if backtest_results.empty:
            print("Error en backtest. Verificar logs.")
            return
        
        # Ejecutar análisis walk-forward
        print("\nEjecutando análisis walk-forward...")
        wf_results = strategy.walk_forward_analysis(window_size=252, step_size=63)
        
        print("\nAnálisis completado. Resultados guardados en './artifacts/results/'")
        
    except Exception as e:
        logging.error(f"Error en ejecución principal: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
