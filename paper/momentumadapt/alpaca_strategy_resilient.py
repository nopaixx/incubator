import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
import logging
import time
import json
import os
from datetime import datetime, timedelta
import requests
from io import StringIO

# Ignorar advertencias
warnings.filterwarnings('ignore')

class AlpacaTradingStrategy:
    def __init__(self, api_key, api_secret, base_url='https://paper-api.alpaca.markets'):
        """
        Inicializa la estrategia de trading con la API de Alpaca.
        """
        self.api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
        self.logger = self._setup_logger()
        
        # Parámetros de la estrategia
        self.max_position_size = 0.05  # 5% máximo por posición
        self.max_sector_exposure = 0.25  # 25% máximo por sector
        self.target_volatility = 0.20  # 20% volatilidad objetivo anual
        self.max_drawdown = -0.15  # Stop loss en -15%
        
        # Inicializar variables
        self.portfolio_weights = {}
        self.market_regimes = None
        self.market_volatility = None
        self.last_rebalance_date = None
        
        # Archivo para persistencia
        self.state_file = 'strategy_state.json'
        
        # Cargar estado anterior si existe
        self.load_state()
        
    def _setup_logger(self):
        """Configura y devuelve un logger."""
        logger = logging.getLogger('alpaca_strategy')
        logger.setLevel(logging.INFO)
        
        # Asegurarse de que el directorio de logs existe
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        # Crear manejador para la consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Crear manejador para archivo con rotación diaria
        file_handler = logging.FileHandler(f'logs/alpaca_strategy_{datetime.now().strftime("%Y%m%d")}.log')
        file_handler.setLevel(logging.INFO)
        
        # Definir formato
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Añadir manejadores al logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def save_state(self):
        """
        Guarda el estado actual de la estrategia para recuperarlo después.
        """
        state = {
            'last_rebalance_date': self.last_rebalance_date.isoformat() if self.last_rebalance_date else None,
            'portfolio_weights': self.portfolio_weights
        }
        
        # Asegurarse de que el directorio de datos existe
        if not os.path.exists('data'):
            os.makedirs('data')
        
        # Guardar a archivo
        try:
            with open(os.path.join('data', self.state_file), 'w') as f:
                json.dump(state, f)
            self.logger.info(f"Estado guardado correctamente")
        except Exception as e:
            self.logger.error(f"Error guardando estado: {str(e)}")
    
    def load_state(self):
        """
        Carga el estado guardado de la estrategia.
        """
        state_path = os.path.join('data', self.state_file) if os.path.exists('data') else self.state_file
        
        if not os.path.exists(state_path):
            self.logger.info("No se encontró archivo de estado. Iniciando con valores por defecto.")
            return False
        
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
            
            # Cargar fecha de último rebalanceo
            if state.get('last_rebalance_date'):
                self.last_rebalance_date = datetime.fromisoformat(state['last_rebalance_date'])
                self.logger.info(f"Último rebalanceo cargado: {self.last_rebalance_date}")
            
            # Cargar pesos del portafolio
            if state.get('portfolio_weights'):
                self.portfolio_weights = state['portfolio_weights']
                self.logger.info(f"Cargados {len(self.portfolio_weights)} pesos del portafolio anterior")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error cargando estado: {str(e)}")
            return False
            
    def get_sp500_tickers(self):
        """
        Obtiene la lista de tickers del S&P 500 desde Wikipedia.
        """
        try:
            # Primero intentar cargar desde caché local
            cache_file = os.path.join('data', 'sp500_tickers.json')
            
            # Si el caché existe y tiene menos de 7 días, usarlo
            if os.path.exists(cache_file):
                file_age = datetime.now().timestamp() - os.path.getmtime(cache_file)
                if file_age < 7 * 24 * 3600:  # 7 días en segundos
                    with open(cache_file, 'r') as f:
                        ticker_sector_dict = json.load(f)
                    self.logger.info(f"Cargados {len(ticker_sector_dict)} tickers del S&P 500 desde caché")
                    return ticker_sector_dict
            
            # Si no hay caché válido, descargar
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            df = tables[0]
            
            # Crear diccionario de ticker -> sector
            ticker_sector_dict = dict(zip(df['Symbol'].str.replace('.', '-'), df['GICS Sector']))
            
            # Filtrar tickers que no estén disponibles en Alpaca
            assets = self.api.list_assets()
            tradable_symbols = [asset.symbol for asset in assets if asset.tradable]
            
            filtered_dict = {ticker: sector for ticker, sector in ticker_sector_dict.items()
                            if ticker in tradable_symbols}
            
            # Guardar en caché
            if not os.path.exists('data'):
                os.makedirs('data')
                
            with open(cache_file, 'w') as f:
                json.dump(filtered_dict, f)
            
            self.logger.info(f"Obtenidos y guardados {len(filtered_dict)} tickers negociables del S&P 500")
            return filtered_dict
        
        except Exception as e:
            self.logger.error(f"Error obteniendo tickers del S&P 500: {str(e)}")
            
            # Si hay un error pero existe un caché, intentar usar el caché incluso si es viejo
            cache_file = os.path.join('data', 'sp500_tickers.json')
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        ticker_sector_dict = json.load(f)
                    self.logger.info(f"Usando {len(ticker_sector_dict)} tickers en caché tras error en descarga")
                    return ticker_sector_dict
                except:
                    pass
            
            return {}
    
    def get_defensive_tickers(self):
        """
        Crea una lista de tickers defensivos (ETFs de bonos, oro, etc.)
        """
        defensive_tickers = [
            'TLT',  # iShares 20+ Year Treasury Bond ETF
            'IEF',  # iShares 7-10 Year Treasury Bond ETF
            'LQD',  # iShares iBoxx $ Investment Grade Corporate Bond ETF
            'GLD',  # SPDR Gold Shares
            'XLU',  # Utilities Select Sector SPDR Fund
            'XLP',  # Consumer Staples Select Sector SPDR Fund
            'USMV', # iShares MSCI USA Min Vol Factor ETF
            'SPLV', # Invesco S&P 500 Low Volatility ETF
            'VNQ',  # Vanguard Real Estate ETF
        ]
        
        try:
            # Filtrar tickers que no estén disponibles en Alpaca
            assets = self.api.list_assets()
            tradable_symbols = [asset.symbol for asset in assets if asset.tradable]
            
            filtered_tickers = [ticker for ticker in defensive_tickers if ticker in tradable_symbols]
            defensive_sectors = {ticker: 'Defensive' for ticker in filtered_tickers}
            
            self.logger.info(f"Seleccionados {len(defensive_sectors)} tickers defensivos disponibles")
            return defensive_sectors
        except Exception as e:
            self.logger.error(f"Error obteniendo tickers defensivos: {str(e)}")
            # Devolver todos los tickers como fallback
            return {ticker: 'Defensive' for ticker in defensive_tickers}
    
    def get_alpaca_historical_data(self, tickers, start_date, end_date=None, timeframe='1D'):
        """
        Intenta obtener datos históricos de Alpaca, si falla usa Yahoo Finance.
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        prices_dict = {}
        volume_dict = {}
        
        try:
            # Intentar obtener datos de Alpaca
            self.logger.info("Intentando obtener datos de Alpaca...")
            
            # Obtener datos para cada ticker (Alpaca tiene límites en el número de tickers por llamada)
            chunk_size = 100  # Procesar en grupos de 100 tickers
            for i in range(0, len(tickers), chunk_size):
                chunk = tickers[i:i+chunk_size]
                
                try:
                    # Obtener barras históricas
                    bars = self.api.get_bars(
                        chunk,
                        timeframe,
                        start=start_date,
                        end=end_date,
                        adjustment='all'  # Ajustado por dividendos y splits
                    )
                    
                    # Convertir a DataFrame
                    df = pd.DataFrame([bar._raw for bar in bars])
                    if df.empty:
                        continue
                    
                    # Convertir timestamp a datetime
                    df['timestamp'] = pd.to_datetime(df['t'], unit='s')
                    
                    # Agrupar por símbolo y fecha
                    for ticker in chunk:
                        ticker_data = df[df['symbol'] == ticker]
                        if not ticker_data.empty:
                            # Crear series con índice de fechas
                            ticker_prices = ticker_data.set_index('timestamp')['c']  # Precios de cierre
                            ticker_volume = ticker_data.set_index('timestamp')['v']  # Volumen
                            
                            prices_dict[ticker] = ticker_prices
                            volume_dict[ticker] = ticker_volume
                
                except Exception as e:
                    if "subscription does not permit" in str(e).lower():
                        self.logger.warning(f"No se tienen permisos para descargar datos de Alpaca. Cambiando a Yahoo Finance...")
                        raise Exception("Cambiar a Yahoo Finance")
                    else:
                        self.logger.error(f"Error obteniendo datos para tickers {chunk}: {str(e)}")
            
        except Exception as e:
            # Si Alpaca falla, usar Yahoo Finance como fallback
            self.logger.info("Usando Yahoo Finance como fuente de datos...")
            
            # Descargar datos desde Yahoo Finance
            data = yf.download(tickers, start=start_date, end=end_date, progress=False)
            
            # Extraer precios de cierre y volumen
            if isinstance(data.columns, pd.MultiIndex):
                prices = data['Close']
                volume = data['Volume']
            else:
                prices = data
                volume = pd.DataFrame()
            
            # Convertir a diccionario para mantener consistencia
            for ticker in tickers:
                if ticker in prices.columns:
                    prices_dict[ticker] = prices[ticker]
                if ticker in volume.columns:
                    volume_dict[ticker] = volume[ticker]
        
        # Crear DataFrames combinados
        prices = pd.DataFrame(prices_dict)
        volume = pd.DataFrame(volume_dict)
        
        return prices, volume
    
    def download_data(self, tickers, start_date, end_date=None, include_defensive=True):
        """
        Descarga datos históricos para los tickers especificados.
        """
        try:
            # Obtener tickers defensivos si se solicita
            defensive_sectors = {}
            if include_defensive:
                defensive_sectors = self.get_defensive_tickers()
                all_tickers = list(tickers) + list(defensive_sectors.keys())
            else:
                all_tickers = list(tickers)
            
            # Añadir un margen de tiempo para calcular características que requieren datos históricos
            extended_start = (pd.to_datetime(start_date) - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
            
            # Descargar datos (primero intenta Alpaca, si falla usa Yahoo)
            prices, volume = self.get_alpaca_historical_data(all_tickers, extended_start, end_date)
            
            # Verificar si hay datos
            if prices.empty:
                raise ValueError("No se pudieron obtener datos para los tickers especificados")
            
            # Eliminar columnas con más del 30% de valores NaN
            valid_columns = prices.columns[prices.isna().mean() < 0.3]
            prices = prices[valid_columns]
            
            # Si el volumen está vacío (puede pasar con Yahoo), crear DataFrame con ceros
            if volume.empty:
                volume = pd.DataFrame(0, index=prices.index, columns=prices.columns)
            else:
                volume = volume[valid_columns]
            
            # Llenar valores NaN con el último valor disponible
            prices = prices.fillna(method='ffill')
            volume = volume.fillna(method='ffill')
            
            # Filtrar para el período solicitado
            if end_date is not None:
                prices = prices.loc[start_date:end_date]
                volume = volume.loc[start_date:end_date]
            else:
                prices = prices.loc[start_date:]
                volume = volume.loc[start_date:]
            
            return prices, volume, defensive_sectors
        
        except Exception as e:
            self.logger.error(f"Error descargando datos: {str(e)}")
            # Devolver DataFrames vacíos en caso de error
            return pd.DataFrame(), pd.DataFrame(), {}
    
    def calculate_returns(self, prices, periods):
        """
        Calcula los retornos para diferentes períodos de tiempo.
        """
        try:
            returns = {}
            
            for period_name, days in periods.items():
                # Calcular retornos para el período especificado
                period_returns = prices.pct_change(periods=days).shift(1)
                returns[period_name] = period_returns
            
            return returns
        
        except Exception as e:
            self.logger.error(f"Error calculando retornos: {str(e)}")
            # Devolver un diccionario vacío en caso de error
            return {}
    
    def calculate_features(self, prices, volume, returns):
        """
        Calcula características para cada ticker en cada fecha.
        """
        try:
            # Calcular retornos diarios para volatilidad
            daily_returns = prices.pct_change()
            
            # Calcular volatilidad (ventana de 21 días)
            volatility = daily_returns.rolling(window=21).std() * np.sqrt(252)
            
            # Calcular cambio de volumen (21 días)
            volume_change = volume.pct_change(periods=21)
            
            # Calcular volumen promedio (21 días)
            avg_volume = volume.rolling(window=21).mean()
            
            # Calcular beta (ventana de 63 días)
            spy_returns = daily_returns.mean(axis=1)  # Proxy para el mercado
            beta = pd.DataFrame(index=prices.index, columns=prices.columns)
            
            for ticker in prices.columns:
                # Calcular beta usando regresión con ventana móvil
                for i in range(63, len(daily_returns)):
                    x = spy_returns.iloc[i-63:i].values.reshape(-1, 1)
                    y = daily_returns.iloc[i-63:i][ticker].values
                    
                    # Eliminar valores NaN
                    mask = ~np.isnan(y)
                    if sum(mask) > 30:  # Al menos 30 puntos válidos
                        try:
                            model = LinearRegression().fit(x[mask], y[mask])
                            beta.iloc[i][ticker] = model.coef_[0]
                        except:
                            beta.iloc[i][ticker] = 1.0  # Valor por defecto
            
            # Calcular indicador RSI (14 días)
            rsi = pd.DataFrame(index=prices.index, columns=prices.columns)
            
            for ticker in prices.columns:
                # Calcular cambios diarios
                delta = daily_returns[ticker]
                
                # Separar ganancias y pérdidas
                gains = delta.copy()
                losses = delta.copy()
                gains[gains < 0] = 0
                losses[losses > 0] = 0
                losses = abs(losses)
                
                # Calcular promedio de ganancias y pérdidas
                avg_gain = gains.rolling(window=14).mean()
                avg_loss = losses.rolling(window=14).mean()
                
                # Calcular RS y RSI
                rs = avg_gain / avg_loss
                rsi[ticker] = 100 - (100 / (1 + rs))
            
            # Asegurarse de que tenemos suficientes datos
            valid_dates = prices.index[63:]  # Cambiar de 21 a 63 debido al cálculo de beta
            
            # Crear lista para almacenar características
            features_list = []
            
            # Para cada fecha después de tener suficientes datos
            for date in valid_dates:
                # Para cada ticker
                for ticker in prices.columns:
                    # Verificar si tenemos datos para este ticker en esta fecha
                    if pd.isna(prices.loc[date, ticker]):
                        continue
                    
                    # Extraer características para este ticker en esta fecha
                    features_dict = {
                        'date': date,
                        'ticker': ticker,
                        'momentum_1m': returns['1M'].loc[date, ticker] if not pd.isna(returns['1M'].loc[date, ticker]) else 0,
                        'momentum_3m': returns['3M'].loc[date, ticker] if not pd.isna(returns['3M'].loc[date, ticker]) else 0,
                        'momentum_6m': returns['6M'].loc[date, ticker] if not pd.isna(returns['6M'].loc[date, ticker]) else 0,
                        'momentum_12m': returns['12M'].loc[date, ticker] if not pd.isna(returns['12M'].loc[date, ticker]) else 0,
                        'volatility': volatility.loc[date, ticker] if not pd.isna(volatility.loc[date, ticker]) else np.nan,
                        'avg_volume': avg_volume.loc[date, ticker] if not pd.isna(avg_volume.loc[date, ticker]) else np.nan,
                        'volume_change': volume_change.loc[date, ticker] if not pd.isna(volume_change.loc[date, ticker]) else 0,
                        'beta': beta.loc[date, ticker] if not pd.isna(beta.loc[date, ticker]) else 1.0,
                        'rsi': rsi.loc[date, ticker] if not pd.isna(rsi.loc[date, ticker]) else 50
                    }
                    
                    features_list.append(features_dict)
            
            # Crear DataFrame con características
            features_df = pd.DataFrame(features_list)
            
            # Manejar valores NaN
            features_df = features_df.fillna(0)
            
            return features_df
        
        except Exception as e:
            self.logger.error(f"Error calculando características: {str(e)}")
            # Devolver un DataFrame vacío en caso de error
            return pd.DataFrame()
    
    def detect_market_regimes(self, prices, n_regimes=3):
        """
        Detecta regímenes de mercado utilizando clustering.
        """
        try:
            # Calcular retornos del mercado (promedio de todos los activos)
            market_returns = prices.pct_change().mean(axis=1).dropna()
            
            # Verificar si hay suficientes datos
            if len(market_returns) < 42:  # Necesitamos al menos 42 días para calcular volatilidad
                self.logger.warning("Datos insuficientes para detectar regímenes. Usando régimen por defecto.")
                return pd.Series(0, index=prices.index)
            
            # Calcular volatilidad rodante (21 días)
            rolling_vol = market_returns.rolling(window=21).std().dropna()
            
            # Calcular volatilidad rodante de alta frecuencia (5 días)
            short_vol = market_returns.rolling(window=5).std() * np.sqrt(252)
            
            # Calcular ratio de volatilidad
            vol_ratio = short_vol / rolling_vol.rolling(window=252).mean()
            vol_ratio = vol_ratio[rolling_vol.index]
            
            # Crear características para el modelo
            features = pd.DataFrame({
                'returns': market_returns[rolling_vol.index],
                'volatility': rolling_vol,
                'vol_ratio': vol_ratio[rolling_vol.index].fillna(1)
            })
            
            # Normalizar características
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Aplicar K-means clustering
            kmeans = KMeans(n_clusters=n_regimes, random_state=42)
            regimes = kmeans.fit_predict(features_scaled)
            
            # Crear Serie con regímenes
            regime_series = pd.Series(regimes, index=features.index)
            
            # Calcular y ordenar los centroides para asignar regímenes de manera consistente
            centroids = kmeans.cluster_centers_
            vol_levels = centroids[:, 1]  # Índice 1 corresponde a la volatilidad
            sorted_indices = np.argsort(vol_levels)
            
            # Crear un mapeo de etiquetas originales a etiquetas ordenadas
            label_map = {sorted_indices[i]: i for i in range(n_regimes)}
            
            # Aplicar el mapeo a las etiquetas de régimen
            mapped_regimes = np.array([label_map[r] for r in regimes])
            regime_series = pd.Series(mapped_regimes, index=features.index)
            
            # Propagar regímenes a todas las fechas
            full_regime_series = pd.Series(index=prices.index)
            
            # Para cada fecha en el índice de precios, usar el último régimen disponible
            for date in prices.index:
                if date in regime_series.index:
                    full_regime_series[date] = regime_series[date]
                elif date > regime_series.index[0]:
                    last_date = regime_series.index[regime_series.index < date][-1]
                    full_regime_series[date] = regime_series[last_date]
                else:
                    full_regime_series[date] = regime_series.iloc[0]
            
            return full_regime_series
        
        except Exception as e:
            self.logger.error(f"Error detectando regímenes de mercado: {str(e)}")
            # Devolver una serie con régimen por defecto en caso de error
            return pd.Series(0, index=prices.index)
    
    def generate_signals(self, features, market_regimes):
        """
        Genera señales de trading basadas en características y regímenes de mercado.
        """
        try:
            # Crear DataFrame para almacenar señales
            signals = pd.DataFrame()
            
            # Obtener fechas únicas
            dates = features['date'].unique()
            
            # Para cada fecha
            for date in dates:
                # Obtener características para esta fecha
                date_features = features[features['date'] == date]
                
                # Obtener régimen de mercado para esta fecha
                regime = market_regimes.get(date, 0)  # Usar régimen 0 por defecto
                
                # Calcular señales según el régimen
                if regime == 0:  # Régimen de baja volatilidad
                    momentum_signal = (
                        0.1 * date_features['momentum_1m'] +
                        0.2 * date_features['momentum_3m'] +
                        0.3 * date_features['momentum_6m'] +
                        0.4 * date_features['momentum_12m']
                    )
                elif regime == 1:  # Régimen de volatilidad media
                    momentum_signal = (
                        0.25 * date_features['momentum_1m'] +
                        0.25 * date_features['momentum_3m'] +
                        0.25 * date_features['momentum_6m'] +
                        0.25 * date_features['momentum_12m']
                    )
                else:  # Régimen de alta volatilidad
                    momentum_component = (
                        0.4 * date_features['momentum_1m'] +
                        0.3 * date_features['momentum_3m'] +
                        0.2 * date_features['momentum_6m'] +
                        0.1 * date_features['momentum_12m']
                    )
                    
                    # Añadir componente de mean-reversion basado en RSI
                    rsi_signal = (50 - date_features['rsi']) / 50
                    
                    # Combinar momentum y mean-reversion
                    momentum_signal = 0.5 * momentum_component + 0.5 * rsi_signal
                
                # Ajustar señales por volatilidad (penalizar alta volatilidad)
                vol_adjustment = 1 / (1 + date_features['volatility'] ** 1.5)
                
                # Ajustar señales por volumen
                volume_adjustment = (
                    date_features['avg_volume'] / date_features['avg_volume'].mean() *
                    (1 + date_features['volume_change'])
                )
                volume_adjustment = volume_adjustment / volume_adjustment.mean()
                
                # Ajustar por beta (penalizar beta alta en regímenes de alta volatilidad)
                if regime == 2:  # Alta volatilidad
                    beta_adjustment = 1 / (1 + date_features['beta'])
                elif regime == 1:  # Media volatilidad
                    beta_adjustment = 1 / (1 + 0.5 * date_features['beta'])
                else:  # Baja volatilidad
                    beta_adjustment = 1  # Sin ajuste en baja volatilidad
                
                # Combinar señales
                combined_signal = momentum_signal * vol_adjustment * volume_adjustment * beta_adjustment
                
                # Crear DataFrame con señales para esta fecha
                date_signals = pd.DataFrame({
                    'date': date,
                    'ticker': date_features['ticker'],
                    'signal': combined_signal,
                    'regime': regime,
                    'volatility': date_features['volatility'],
                    'beta': date_features['beta'],
                    'rsi': date_features['rsi']
                })
                
                # Añadir a DataFrame de señales
                signals = pd.concat([signals, date_signals], ignore_index=True)
            
            return signals
        
        except Exception as e:
            self.logger.error(f"Error generando señales: {str(e)}")
            # Devolver un DataFrame vacío en caso de error
            return pd.DataFrame()
    
    def calculate_volatility_target_multiplier(self, market_volatility, target_vol=None, max_leverage=1.5, min_leverage=0.25):
        """
        Calcula el multiplicador de exposición basado en la volatilidad del mercado.
        """
        if target_vol is None:
            target_vol = self.target_volatility
            
        # Evitar división por cero
        if market_volatility <= 0:
            return max_leverage
        
        # Calcular multiplicador basado en volatilidad objetivo
        vol_multiplier = target_vol / market_volatility
        
        # Limitar el multiplicador dentro del rango permitido
        vol_multiplier = max(min(vol_multiplier, max_leverage), min_leverage)
        
        return vol_multiplier
    
    def implement_stop_loss(self, prices, portfolio_weights, current_date):
        """
        Implementa stop loss para reducir drawdowns.
        """
        try:
            # Si no hay pesos, devolver diccionario vacío
            if not portfolio_weights:
                return {}
            
            # Copia de los pesos para no modificar el original
            current_weights = portfolio_weights.copy()
            
            # Obtener fechas anteriores (último mes)
            one_month_ago = current_date - pd.Timedelta(days=30)
            past_dates = prices.index[(prices.index >= one_month_ago) & (prices.index < current_date)]
            
            if len(past_dates) < 5:  # Necesitamos al menos algunos días de historia
                return current_weights
            
            # Calcular drawdowns para cada activo
            for ticker in list(current_weights.keys()):
                if ticker in prices.columns:
                    # Obtener precios para este ticker
                    ticker_prices = prices.loc[past_dates, ticker]
                    
                    if len(ticker_prices) < 5:
                        continue
                    
                    # Calcular drawdown
                    peak = ticker_prices.max()
                    current_price = prices.loc[current_date, ticker]
                    drawdown = (current_price / peak) - 1
                    
                    # Si drawdown excede el máximo, eliminar posición
                    if drawdown < self.max_drawdown:
                        self.logger.info(f"Stop Loss activado para {ticker}: drawdown {drawdown:.2%} excede límite {self.max_drawdown:.2%}")
                        del current_weights[ticker]
            
            # Si se eliminaron todas las posiciones, devolver diccionario vacío
            if not current_weights:
                return {}
            
            # Normalizar pesos restantes
            total_weight = sum(current_weights.values())
            if total_weight > 0:
                current_weights = {ticker: weight / total_weight for ticker, weight in current_weights.items()}
            
            return current_weights
        
        except Exception as e:
            self.logger.error(f"Error implementando stop loss para {current_date}: {str(e)}")
            # Devolver los pesos originales en caso de error
            return portfolio_weights
    
    def construct_portfolio(self, signals, sectors, date, top_pct=0.1):
        """
        Construye un portafolio basado en señales para una fecha específica.
        """
        try:
            # Obtener señales para la fecha especificada
            date_signals = signals[signals['date'] == date].copy()
            
            # Si no hay señales para esta fecha, devolver diccionario vacío
            if date_signals.empty:
                return {}
            
            # Añadir sector a cada ticker
            date_signals['sector'] = date_signals['ticker'].map(lambda x: sectors.get(x, 'Unknown'))
            
            # Obtener el régimen actual
            current_regime = date_signals['regime'].iloc[0]
            
            # Separar tickers defensivos y no defensivos
            defensive_tickers = date_signals[date_signals['sector'] == 'Defensive']
            non_defensive_tickers = date_signals[date_signals['sector'] != 'Defensive']
            
            # Ajustar porcentaje de exposición a activos defensivos según régimen
            defensive_allocation = 0.0  # Por defecto, sin activos defensivos
            
            if current_regime == 2:  # Alta volatilidad
                defensive_allocation = 0.40  # 40% a defensivos en alta volatilidad
            elif current_regime == 1:  # Media volatilidad
                defensive_allocation = 0.20  # 20% a defensivos en media volatilidad
            elif current_regime == 0:  # Baja volatilidad
                defensive_allocation = 0.05  # 5% a defensivos en baja volatilidad
            
            # Asignar pesos iniciales
            weights = {}
            
            # Procesar activos no defensivos
            non_defensive_tickers = non_defensive_tickers.sort_values('signal', ascending=False)
            n_non_defensive = int(len(non_defensive_tickers) * top_pct)
            top_non_defensive = non_defensive_tickers.head(n_non_defensive)
            
            # Calcular exposición por sector para activos no defensivos
            sector_exposure = top_non_defensive.groupby('sector').size() / n_non_defensive
            
            # Incorporar risk parity para activos no defensivos
            # Inversamente proporcional a la volatilidad
            top_non_defensive['risk_weight'] = 1 / top_non_defensive['volatility'].replace(0, 0.01)
            top_non_defensive['risk_weight'] = top_non_defensive['risk_weight'] / top_non_defensive['risk_weight'].sum()
            
            # Combinar señal y riesgo (70% señal, 30% risk parity)
            top_non_defensive['combined_weight'] = 0.7 * top_non_defensive['signal'] + 0.3 * top_non_defensive['risk_weight']
            
            # Para cada ticker no defensivo
            for _, row in top_non_defensive.iterrows():
                ticker = row['ticker']
                sector = row['sector']
                
                # Si la exposición del sector excede el máximo, reducir peso
                if sector in sector_exposure and sector_exposure[sector] > self.max_sector_exposure:
                    weight = row['combined_weight'] * (self.max_sector_exposure / sector_exposure[sector])
                else:
                    weight = row['combined_weight']
                
                weights[ticker] = weight * (1 - defensive_allocation)
            
            # Procesar activos defensivos
            if not defensive_tickers.empty and defensive_allocation > 0:
                defensive_tickers = defensive_tickers.sort_values('signal', ascending=False)
                
                # Tomar los mejores activos defensivos
                n_defensive = min(5, len(defensive_tickers))
                top_defensive = defensive_tickers.head(n_defensive)
                
                # Asignar pesos equitativos a los activos defensivos
                defensive_weight_per_asset = defensive_allocation / n_defensive
                
                for ticker in top_defensive['ticker']:
                    weights[ticker] = defensive_weight_per_asset
            
            # Limitar tamaño máximo por posición
            for ticker in list(weights.keys()):
                if weights[ticker] > self.max_position_size:
                    excess = weights[ticker] - self.max_position_size
                    weights[ticker] = self.max_position_size
                    
                    # Redistribuir exceso proporcionalmente entre otros activos
                    total_other_weights = sum(w for t, w in weights.items() if t != ticker)
                    if total_other_weights > 0:
                        for t in weights:
                            if t != ticker:
                                weights[t] += excess * (weights[t] / total_other_weights)
            
            # Normalizar pesos para que sumen 1
            total_weight = sum(weights.values())
            
            if total_weight > 0:
                weights = {ticker: weight / total_weight for ticker, weight in weights.items()}
            
            return weights
        
        except Exception as e:
            self.logger.error(f"Error construyendo portafolio para {date}: {str(e)}")
            # Devolver diccionario vacío en caso de error
            return {}
    
    def run_strategy(self, initial_cash=None, rebalance_freq='M', lookback_days=365):
        """
        Ejecuta la estrategia de trading en tiempo real.
        
        Args:
            initial_cash: Cantidad inicial a invertir (si None, usa todo el efectivo disponible)
            rebalance_freq: Frecuencia de rebalanceo ('D'-diario, 'W'-semanal, 'M'-mensual)
            lookback_days: Número de días de datos históricos para analizar
        """
        try:
            # Obtener fecha actual
            current_date = datetime.now()
            
            # Verificar si es momento de rebalancear
            if self.last_rebalance_date is not None:
                if rebalance_freq == 'D':  # Diario
                    days_diff = (current_date - self.last_rebalance_date).days
                    if days_diff < 1:
                        self.logger.info("No es momento de rebalancear (siguiente rebalanceo mañana)")
                        return False
                elif rebalance_freq == 'W':  # Semanal
                    days_diff = (current_date - self.last_rebalance_date).days
                    if days_diff < 7:
                        self.logger.info(f"No es momento de rebalancear (siguiente rebalanceo en {7-days_diff} días)")
                        return False
                elif rebalance_freq == 'M':  # Mensual
                    current_month = current_date.month
                    current_year = current_date.year
                    last_month = self.last_rebalance_date.month
                    last_year = self.last_rebalance_date.year
                    
                    if (current_year == last_year and current_month == last_month):
                        self.logger.info("No es momento de rebalancear (siguiente rebalanceo el próximo mes)")
                        return False
            
            # Obtener tickers del S&P 500 y sus sectores
            self.logger.info("Obteniendo tickers del S&P 500...")
            sectors = self.get_sp500_tickers()
            tickers = list(sectors.keys())
            
            # Si hay demasiados tickers, limitarlos para evitar problemas
            if len(tickers) > 2000:  # Alpaca tiene límites en el número de tickers para datos históricos
                self.logger.info(f"Limitando número de tickers de {len(tickers)} a 200")
                # Ordenar alfabéticamente para consistencia
                tickers = sorted(tickers)[:2000]
                sectors = {ticker: sectors[ticker] for ticker in tickers}
            
            # Definir fechas para datos históricos
            end_date = current_date.strftime('%Y-%m-%d')
            start_date = (current_date - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            
            # Descargar datos históricos
            self.logger.info(f"Descargando datos históricos desde {start_date} hasta {end_date}...")
            prices, volume, defensive_sectors = self.download_data(tickers, start_date, end_date)
            
            # Actualizar sectores con activos defensivos
            sectors.update(defensive_sectors)
            
            # Verificar si hay datos
            if prices.empty or volume.empty:
                self.logger.error("No se pudieron obtener datos históricos")
                return False
            
            self.logger.info(f"Datos descargados: {len(prices.columns)} tickers con {len(prices)} días de datos")
            
            # Calcular retornos para diferentes períodos
            self.logger.info("Calculando retornos para diferentes períodos...")
            periods = {
                '1M': 21,
                '3M': 63,
                '6M': 126,
                '12M': 252
            }
            returns = self.calculate_returns(prices, periods)
            
            # Calcular características
            self.logger.info("Calculando características...")
            features = self.calculate_features(prices, volume, returns)
            
            # Detectar regímenes de mercado
            self.logger.info("Detectando regímenes de mercado...")
            market_regimes = self.detect_market_regimes(prices)
            self.market_regimes = market_regimes  # Guardar para persistencia
            
            # Generar señales
            self.logger.info("Generando señales...")
            signals = self.generate_signals(features, market_regimes)
            
            # Obtener última fecha con datos
            latest_date = prices.index[-1]
            
            # Construir portafolio para la fecha actual
            self.logger.info(f"Construyendo portafolio para la fecha {latest_date}...")
            weights = self.construct_portfolio(signals, sectors, latest_date)
            
            # Implementar stop loss
            self.logger.info("Aplicando stop loss...")
            weights = self.implement_stop_loss(prices, weights, latest_date)
            
            # Calcular volatilidad del mercado para volatility targeting
            market_returns = prices.pct_change().mean(axis=1)
            market_volatility = market_returns.rolling(window=21).std() * np.sqrt(252)
            latest_volatility = market_volatility.iloc[-1] if not market_volatility.empty else 0.20
            
            # Calcular multiplicador de exposición
            vol_multiplier = self.calculate_volatility_target_multiplier(latest_volatility)
            self.logger.info(f"Volatilidad actual del mercado: {latest_volatility:.2%}, Multiplicador: {vol_multiplier:.2f}")
            
            # Aplicar volatility targeting
            if vol_multiplier < 1:
                # Crear posición en efectivo (no invertida)
                weights['CASH'] = 1 - vol_multiplier
                
                # Reducir todas las demás posiciones proporcionalmente
                for ticker in list(weights.keys()):
                    if ticker != 'CASH':
                        weights[ticker] *= vol_multiplier
            
            # Guardar los pesos calculados antes de implementarlos
            self.portfolio_weights = weights
            
            # Guardar estado actual antes de implementar para evitar pérdida de datos
            self.save_state()
            
            # Implementar el portafolio en Alpaca
            self.logger.info("Implementando portafolio en Alpaca...")
            success = self.rebalance_portfolio(weights, initial_cash)
            
            if success:
                self.logger.info("Portafolio rebalanceado exitosamente")
                self.last_rebalance_date = current_date
                self.save_state()  # Actualizar estado con fecha de rebalanceo
                return True
            else:
                self.logger.error("Error rebalanceando portafolio")
                return False
        
        except Exception as e:
            self.logger.error(f"Error ejecutando estrategia: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def rebalance_portfolio(self, target_weights, initial_cash=None):
        """
        Rebalancea el portafolio para alcanzar los pesos objetivo.
        
        Args:
            target_weights: Diccionario con pesos objetivo para cada ticker
            initial_cash: Cantidad inicial a invertir (si None, usa todo el efectivo disponible)
        """
        try:
            # Obtener posiciones actuales
            positions = self.api.list_positions()
            current_positions = {p.symbol: float(p.market_value) for p in positions}
            
            # Obtener valor total del portafolio
            account = self.api.get_account()
            portfolio_value = float(account.portfolio_value)
            cash = float(account.cash)
            
            # Si se especifica una cantidad inicial, limitar el valor del portafolio
            if initial_cash is not None and initial_cash < portfolio_value:
                portfolio_value = initial_cash
                # Ajustar efectivo disponible
                cash = min(cash, initial_cash)
            
            self.logger.info(f"Valor del portafolio: ${portfolio_value:.2f}, Efectivo disponible: ${cash:.2f}")
            
            # Calcular valores objetivo para cada ticker
            target_values = {}
            for ticker, weight in target_weights.items():
                if ticker != 'CASH':  # Ignorar "CASH", que es solo conceptual
                    target_values[ticker] = portfolio_value * weight
            
            # Calcular operaciones necesarias
            orders = []
            
            # Liquidar posiciones que ya no están en el portafolio objetivo
            for symbol in current_positions:
                if symbol not in target_values:
                    orders.append({
                        'symbol': symbol,
                        'qty': 0,  # Liquidar toda la posición
                        'side': 'sell'
                    })
                    self.logger.info(f"Liquidar posición en {symbol}")
            
            # Calcular órdenes para alcanzar los pesos objetivo
            for symbol, target_value in target_values.items():
                # Obtener último precio
                try:
                    # Intentar obtener el último trade
                    latest_trade = self.api.get_latest_trade(symbol)
                    last_price = float(latest_trade.price)
                except:
                    try:
                        # Si falla, obtener el último precio de cierre
                        latest_bar = self.api.get_latest_bar(symbol)
                        last_price = float(latest_bar.c)
                    except:
                        # Como último recurso, usar Yahoo Finance
                        try:
                            ticker = yf.Ticker(symbol)
                            last_price = ticker.info.get('regularMarketPrice', ticker.info.get('currentPrice'))
                            if last_price is None:
                                last_price = ticker.history(period="1d")['Close'].iloc[-1]
                        except:
                            self.logger.error(f"No se pudo obtener precio para {symbol}")
                            continue
                
                # Calcular cantidad objetivo
                target_qty = int(target_value / last_price)
                
                # Calcular cantidad actual
                current_qty = 0
                if symbol in current_positions:
                    current_position = next((p for p in positions if p.symbol == symbol), None)
                    if current_position:
                        current_qty = int(current_position.qty)
                
                # Calcular diferencia
                qty_diff = target_qty - current_qty
                
                # Si hay una diferencia significativa, crear orden
                if abs(qty_diff) > 0:
                    side = 'buy' if qty_diff > 0 else 'sell'
                    abs_qty = abs(qty_diff)
                    
                    # Limitar cantidad para ventas a la cantidad actual
                    if side == 'sell':
                        abs_qty = min(abs_qty, current_qty)
                    
                    orders.append({
                        'symbol': symbol,
                        'qty': abs_qty,
                        'side': side
                    })
                    
                    self.logger.info(f"{side.upper()} {abs_qty} acciones de {symbol} a ${last_price:.2f} -- {last_price*abs_qty}")
            
            # Ejecutar órdenes
            for order in orders:
                symbol = order['symbol']
                qty = order['qty']
                side = order['side']
                
                try:
                    # Para liquidaciones, cancelar órdenes pendientes y cerrar posición
                    if side == 'sell' and qty == 0:
                        # Cancelar órdenes pendientes para este símbolo
                        open_orders = self.api.list_orders(symbol=symbol, status='open')
                        for open_order in open_orders:
                            self.api.cancel_order(open_order.id)
                        
                        # Cerrar posición
                        self.api.close_position(symbol)
                    else:
                        # Orden normal
                        self.api.submit_order(
                            symbol=symbol,
                            qty=qty,
                            side=side,
                            type='market',
                            time_in_force='day'
                        )
                except Exception as e:
                    self.logger.error(f"Error enviando orden para {symbol}: {str(e)}")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error rebalanceando portafolio: {str(e)}")
            return False
    
    def run_scheduled_strategy(self, interval_minutes=60, rebalance_freq='M'):
        """
        Ejecuta la estrategia de forma programada a intervalos regulares.
        
        Args:
            interval_minutes: Intervalo en minutos para verificar si es momento de rebalancear
            rebalance_freq: Frecuencia de rebalanceo ('D', 'W', 'M')
        """
        self.logger.info(f"Iniciando estrategia programada con frecuencia de rebalanceo {rebalance_freq}")
        self.logger.info(f"Verificando cada {interval_minutes} minutos si es momento de rebalancear")
        
        while True:
            try:
                # Ejecutar estrategia
                self.run_strategy(rebalance_freq=rebalance_freq)
                
                # Esperar hasta el próximo intervalo
                self.logger.info(f"Esperando {interval_minutes} minutos hasta la próxima verificación...")
                time.sleep(interval_minutes * 60)
                
                # Guardar estado después de cada ciclo completo
                self.save_state()
            
            except KeyboardInterrupt:
                self.logger.info("Estrategia detenida manualmente")
                self.save_state()  # Guardar estado antes de salir
                break
            
            except Exception as e:
                self.logger.error(f"Error en ejecución programada: {str(e)}")
                # Esperar antes de reintentar
                time.sleep(60)  # Esperar 1 minuto antes de reintentar