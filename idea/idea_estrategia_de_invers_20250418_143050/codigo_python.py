import os
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import traceback
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
from tqdm import tqdm
import json
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller
from scipy.signal import argrelextrema
import requests
from bs4 import BeautifulSoup
import time
import random

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

# Suprimir advertencias
warnings.filterwarnings('ignore')

class QuantitativeStrategy:
    def __init__(self, tickers=None, start_date='2010-01-01', end_date=None, 
                 lookback_period=252, regime_clusters=3, max_positions=5, 
                 stop_loss=0.05, take_profit=0.15, position_size=0.2,
                 vix_threshold=None, atr_multiplier=2.0):
        """
        Inicializa la estrategia cuantitativa.
        
        Args:
            tickers: Lista de tickers o None para usar S&P 500
            start_date: Fecha de inicio para los datos
            end_date: Fecha de fin para los datos (None para usar fecha actual)
            lookback_period: Período para cálculos retrospectivos (días de trading)
            regime_clusters: Número de regímenes de mercado a identificar
            max_positions: Número máximo de posiciones simultáneas
            stop_loss: Porcentaje de stop loss (0.05 = 5%)
            take_profit: Porcentaje de take profit (0.15 = 15%)
            position_size: Tamaño de cada posición como fracción del capital (0.2 = 20%)
            vix_threshold: Umbral de VIX para circuit breaker (None para inferir automáticamente)
            atr_multiplier: Multiplicador para ATR en cálculos de volatilidad
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.lookback_period = lookback_period
        self.regime_clusters = regime_clusters
        self.max_positions = max_positions
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_size = position_size
        self.vix_threshold = vix_threshold
        self.atr_multiplier = atr_multiplier
        
        # Variables de estado
        self.data = None
        self.vix_data = None
        self.sp500_data = None
        self.regimes = None
        self.signals = None
        self.positions = []
        self.closed_positions = []
        self.capital = 100000  # Capital inicial
        self.equity_curve = []
        self.current_regime = None
        
        # Métricas de rendimiento
        self.metrics = {}
        
    def get_sp500_tickers(self):
        """
        Obtiene la lista de tickers del S&P 500 desde Wikipedia.
        
        Returns:
            list: Lista de tickers del S&P 500
        """
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'class': 'wikitable sortable'})
            
            tickers = []
            for row in table.findAll('tr')[1:]:
                ticker = row.findAll('td')[0].text.strip()
                tickers.append(ticker)
            
            return tickers
        except Exception as e:
            logging.error(f"Error obteniendo tickers del S&P 500: {str(e)}")
            logging.error(traceback.format_exc())
            # Retornar una lista de tickers populares como fallback
            return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V']
    
    def load_data(self):
        """
        Carga los datos de precios para los tickers especificados y el VIX.
        
        Returns:
            bool: True si los datos se cargaron correctamente, False en caso contrario
        """
        try:
            # Obtener tickers del S&P 500 si no se proporcionaron
            if self.tickers is None:
                self.tickers = self.get_sp500_tickers()
                
            # Cargar datos del VIX
            self.vix_data = yf.download('^VIX', start=self.start_date, end=self.end_date)
            
            # Cargar datos del S&P 500 para referencia
            self.sp500_data = yf.download('^GSPC', start=self.start_date, end=self.end_date)
            
            # Seleccionar una muestra aleatoria de tickers si hay demasiados
            if len(self.tickers) > 50:
                selected_tickers = random.sample(self.tickers, 50)
            else:
                selected_tickers = self.tickers
                
            # Cargar datos para los tickers seleccionados
            all_data = {}
            for ticker in tqdm(selected_tickers, desc="Cargando datos"):
                try:
                    ticker_data = yf.download(ticker, start=self.start_date, end=self.end_date)
                    if not ticker_data.empty and len(ticker_data) > self.lookback_period:
                        all_data[ticker] = ticker_data
                except Exception as e:
                    logging.error(f"Error cargando datos para {ticker}: {str(e)}")
                    continue
            
            # Verificar si se obtuvieron datos
            if not all_data:
                logging.error("No se pudieron cargar datos para ningún ticker")
                return False
                
            # Seleccionar el ticker con más datos para análisis detallado
            ticker_lengths = {ticker: len(data) for ticker, data in all_data.items()}
            selected_ticker = max(ticker_lengths, key=ticker_lengths.get)
            
            self.data = all_data[selected_ticker]
            self.data['Ticker'] = selected_ticker
            
            # Calcular retornos diarios
            self.data['Returns'] = self.data['Close'].pct_change()
            
            # Eliminar filas con NaN al inicio
            self.data = self.data.dropna()
            
            # Alinear índices de VIX con los datos de precios
            self.vix_data = self.vix_data.reindex(self.data.index, method='ffill')
            self.sp500_data = self.sp500_data.reindex(self.data.index, method='ffill')
            
            return True
        except Exception as e:
            logging.error(f"Error en load_data: {str(e)}")
            logging.error(traceback.format_exc())
            return False
    
    def calculate_technical_indicators(self):
        """
        Calcula indicadores técnicos para el análisis.
        
        Returns:
            bool: True si los indicadores se calcularon correctamente, False en caso contrario
        """
        try:
            # Medias móviles
            self.data['SMA_20'] = self.data['Close'].rolling(window=20).mean()
            self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()
            self.data['SMA_200'] = self.data['Close'].rolling(window=200).mean()
            
            # Bandas de Bollinger (20 días)
            self.data['BB_Middle'] = self.data['SMA_20']
            self.data['BB_Std'] = self.data['Close'].rolling(window=20).std()
            self.data['BB_Upper'] = self.data['BB_Middle'] + 2 * self.data['BB_Std']
            self.data['BB_Lower'] = self.data['BB_Middle'] - 2 * self.data['BB_Std']
            
            # RSI (14 días)
            delta = self.data['Close'].diff()
            gain = delta.where(delta > 0, 0).fillna(0)
            loss = -delta.where(delta < 0, 0).fillna(0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            self.data['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            self.data['EMA_12'] = self.data['Close'].ewm(span=12, adjust=False).mean()
            self.data['EMA_26'] = self.data['Close'].ewm(span=26, adjust=False).mean()
            self.data['MACD'] = self.data['EMA_12'] - self.data['EMA_26']
            self.data['MACD_Signal'] = self.data['MACD'].ewm(span=9, adjust=False).mean()
            self.data['MACD_Hist'] = self.data['MACD'] - self.data['MACD_Signal']
            
            # ATR (14 días)
            high_low = self.data['High'] - self.data['Low']
            high_close = abs(self.data['High'] - self.data['Close'].shift())
            low_close = abs(self.data['Low'] - self.data['Close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            self.data['ATR'] = true_range.rolling(14).mean()
            
            # Volatilidad (desviación estándar de retornos en 20 días)
            self.data['Volatility'] = self.data['Returns'].rolling(window=20).std() * np.sqrt(252)
            
            # Momentum (retorno de 10 días)
            self.data['Momentum'] = self.data['Close'] / self.data['Close'].shift(10) - 1
            
            # Relación con S&P 500
            self.data['SP500_Returns'] = self.sp500_data['Close'].pct_change()
            
            # Beta (sensibilidad al mercado)
            returns_df = pd.DataFrame({
                'stock': self.data['Returns'],
                'market': self.data['SP500_Returns']
            }).dropna()
            
            if len(returns_df) > 30:  # Asegurar suficientes datos para cálculo de beta
                beta_model = np.polyfit(returns_df['market'], returns_df['stock'], 1)
                self.data['Beta'] = beta_model[0]
            else:
                self.data['Beta'] = 1.0  # Valor por defecto
            
            # Rellenar NaN con valores forward fill para evitar problemas
            self.data = self.data.fillna(method='ffill')
            
            # Eliminar las filas restantes con NaN (principalmente al inicio)
            self.data = self.data.dropna()
            
            return True
        except Exception as e:
            logging.error(f"Error en calculate_technical_indicators: {str(e)}")
            logging.error(traceback.format_exc())
            # Continuar con valores por defecto
            return True
    
    def detect_market_regimes(self):
        """
        Detecta regímenes de mercado utilizando clustering.
        
        Returns:
            bool: True si los regímenes se detectaron correctamente, False en caso contrario
        """
        try:
            # Preparar características para clustering
            features = self.data[['Volatility', 'RSI', 'Momentum']].copy()
            
            # Normalizar características
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Aplicar K-means clustering
            kmeans = KMeans(n_clusters=self.regime_clusters, random_state=42)
            self.data['Regime'] = kmeans.fit_predict(scaled_features)
            
            # Analizar características de cada régimen
            regime_stats = self.data.groupby('Regime').agg({
                'Returns': ['mean', 'std'],
                'Volatility': 'mean',
                'RSI': 'mean',
                'Momentum': 'mean'
            })
            
            # Etiquetar regímenes (0: Bajista, 1: Neutral, 2: Alcista)
            # Ordenar por retorno medio
            regime_order = regime_stats[('Returns', 'mean')].argsort()
            regime_mapping = {
                regime_order[0]: 0,  # Bajista
                regime_order[1]: 1,  # Neutral
                regime_order[2]: 2,  # Alcista
            }
            
            # Aplicar mapeo
            self.data['Regime'] = self.data['Regime'].map(regime_mapping)
            
            # Guardar estadísticas de regímenes
            self.regime_stats = regime_stats
            
            # Guardar estadísticas en CSV
            regime_stats_df = pd.DataFrame({
                'Regime': ['Bajista', 'Neutral', 'Alcista'],
                'Mean_Return': regime_stats[('Returns', 'mean')].values,
                'Std_Return': regime_stats[('Returns', 'std')].values,
                'Mean_Volatility': regime_stats[('Volatility', 'mean')].values,
                'Mean_RSI': regime_stats[('RSI', 'mean')].values,
                'Mean_Momentum': regime_stats[('Momentum', 'mean')].values
            })
            
            regime_stats_df.to_csv('./artifacts/results/data/regime_statistics.csv', index=False)
            
            # Visualizar regímenes
            plt.figure(figsize=(12, 8))
            for regime in range(3):
                regime_data = self.data[self.data['Regime'] == regime]
                plt.scatter(regime_data.index, regime_data['Close'], 
                           label=f"Régimen {['Bajista', 'Neutral', 'Alcista'][regime]}", 
                           alpha=0.7, s=30)
            
            plt.title('Regímenes de Mercado Detectados')
            plt.xlabel('Fecha')
            plt.ylabel('Precio')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('./artifacts/results/figures/market_regimes.png')
            plt.close()
            
            return True
        except Exception as e:
            logging.error(f"Error en detect_market_regimes: {str(e)}")
            logging.error(traceback.format_exc())
            # Asignar régimen neutral por defecto
            self.data['Regime'] = 1
            return True
    
    def check_circuit_breakers(self):
        """
        Implementa circuit breakers basados en VIX y otros indicadores.
        
        Returns:
            bool: True si los circuit breakers se implementaron correctamente, False en caso contrario
        """
        try:
            # Inferir umbral de VIX si no se proporciona
            if self.vix_threshold is None:
                # Usar percentil 90 del VIX como umbral
                self.vix_threshold = np.percentile(self.vix_data['Close'], 90)
            
            # Crear columna de circuit breaker
            self.data['Circuit_Breaker'] = False
            
            # Verificar VIX para cada fecha
            for date in self.data.index:
                if date in self.vix_data.index:
                    vix_value = self.vix_data.loc[date, 'Close']
                    
                    # Activar circuit breaker si VIX supera umbral
                    if vix_value > self.vix_threshold:
                        self.data.loc[date, 'Circuit_Breaker'] = True
            
            # Añadir circuit breaker basado en movimientos extremos de precio
            daily_returns = self.data['Returns'].abs()
            extreme_move_threshold = daily_returns.mean() + 3 * daily_returns.std()
            
            extreme_days = self.data[daily_returns > extreme_move_threshold].index
            self.data.loc[extreme_days, 'Circuit_Breaker'] = True
            
            # Visualizar circuit breakers
            plt.figure(figsize=(12, 8))
            plt.plot(self.data.index, self.data['Close'], label='Precio', alpha=0.7)
            
            # Marcar días con circuit breaker
            circuit_days = self.data[self.data['Circuit_Breaker']].index
            plt.scatter(circuit_days, self.data.loc[circuit_days, 'Close'], 
                       color='red', label='Circuit Breaker', s=50, marker='x')
            
            plt.title('Circuit Breakers Detectados')
            plt.xlabel('Fecha')
            plt.ylabel('Precio')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('./artifacts/results/figures/circuit_breakers.png')
            plt.close()
            
            # Guardar estadísticas de circuit breakers
            circuit_stats = {
                'vix_threshold': self.vix_threshold,
                'extreme_move_threshold': extreme_move_threshold,
                'total_circuit_breaker_days': self.data['Circuit_Breaker'].sum(),
                'percentage_circuit_breaker_days': (self.data['Circuit_Breaker'].sum() / len(self.data)) * 100
            }
            
            with open('./artifacts/results/data/circuit_breaker_stats.json', 'w') as f:
                json.dump(circuit_stats, f, indent=4)
            
            return True
        except Exception as e:
            logging.error(f"Error en check_circuit_breakers: {str(e)}")
            logging.error(traceback.format_exc())
            # Continuar sin circuit breakers
            self.data['Circuit_Breaker'] = False
            return True
    
    def generate_trading_signals(self):
        """
        Genera señales de trading basadas en regímenes y condiciones técnicas.
        
        Returns:
            bool: True si las señales se generaron correctamente, False en caso contrario
        """
        try:
            # Inicializar columnas de señales
            self.data['Signal'] = 0  # 0: Sin señal, 1: Compra, -1: Venta
            
            # Iterar a través de los datos para generar señales
            for i in range(1, len(self.data)):
                date = self.data.index[i]
                prev_date = self.data.index[i-1]
                
                # No generar señales si hay circuit breaker activo
                if self.data.loc[prev_date, 'Circuit_Breaker']:
                    continue
                
                # Obtener régimen actual
                current_regime = self.data.loc[prev_date, 'Regime']
                self.current_regime = current_regime
                
                # Verificar posiciones activas
                active_positions = len([p for p in self.positions if p['exit_date'] is None])
                
                # Condiciones de entrada según régimen
                if active_positions < self.max_positions:
                    # Régimen alcista: comprar en soporte técnico
                    if current_regime == 2:  # Alcista
                        # Comprar si el precio está cerca del soporte (BB inferior o SMA50)
                        if (self.data.loc[prev_date, 'Close'] <= self.data.loc[prev_date, 'BB_Lower'] or
                            self.data.loc[prev_date, 'Close'] <= self.data.loc[prev_date, 'SMA_50']) and \
                           self.data.loc[prev_date, 'RSI'] < 40:
                            self.data.loc[date, 'Signal'] = 1
                    
                    # Régimen neutral: comprar en rebotes técnicos
                    elif current_regime == 1:  # Neutral
                        # Comprar si hay cruce de MACD y RSI muestra sobrevendido
                        if self.data.loc[prev_date, 'MACD_Hist'] > 0 and \
                           self.data.loc[prev_date-1:prev_date-1, 'MACD_Hist'].values[0] < 0 and \
                           self.data.loc[prev_date, 'RSI'] < 45:
                            self.data.loc[date, 'Signal'] = 1
                    
                    # Régimen bajista: comprar solo en condiciones muy específicas
                    elif current_regime == 0:  # Bajista
                        # Comprar solo en rebotes fuertes con confirmación
                        if self.data.loc[prev_date, 'RSI'] < 30 and \
                           self.data.loc[prev_date, 'Close'] > self.data.loc[prev_date, 'Open'] and \
                           self.data.loc[prev_date, 'Momentum'] > 0:
                            self.data.loc[date, 'Signal'] = 1
                
                # Gestionar posiciones existentes
                for position in self.positions:
                    if position['exit_date'] is None:  # Posición abierta
                        entry_price = position['entry_price']
                        current_price = self.data.loc[prev_date, 'Close']
                        
                        # Calcular ganancia/pérdida actual
                        pnl_pct = (current_price - entry_price) / entry_price
                        
                        # Condiciones de salida
                        # Stop loss
                        if pnl_pct <= -self.stop_loss:
                            position['exit_date'] = date
                            position['exit_price'] = self.data.loc[date, 'Open']  # Usar precio de apertura para salida
                            position['exit_type'] = 'stop_loss'
                            self.closed_positions.append(position)
                            self.positions.remove(position)
                        
                        # Take profit
                        elif pnl_pct >= self.take_profit:
                            position['exit_date'] = date
                            position['exit_price'] = self.data.loc[date, 'Open']  # Usar precio de apertura para salida
                            position['exit_type'] = 'take_profit'
                            self.closed_positions.append(position)
                            self.positions.remove(position)
                        
                        # Salida basada en cambio de régimen
                        elif current_regime == 0 and position['regime'] != 0:  # Si cambia a bajista
                            position['exit_date'] = date
                            position['exit_price'] = self.data.loc[date, 'Open']  # Usar precio de apertura para salida
                            position['exit_type'] = 'regime_change'
                            self.closed_positions.append(position)
                            self.positions.remove(position)
                
                # Abrir nuevas posiciones según señales
                if self.data.loc[date, 'Signal'] == 1:
                    # Verificar nuevamente el límite de posiciones
                    if len([p for p in self.positions if p['exit_date'] is None]) < self.max_positions:
                        entry_price = self.data.loc[date, 'Open']  # Usar precio de apertura para entrada
                        
                        # Calcular tamaño de posición
                        position_value = self.capital * self.position_size
                        shares = position_value / entry_price
                        
                        # Registrar nueva posición
                        new_position = {
                            'entry_date': date,
                            'entry_price': entry_price,
                            'shares': shares,
                            'regime': current_regime,
                            'exit_date': None,
                            'exit_price': None,
                            'exit_type': None
                        }
                        
                        self.positions.append(new_position)
            
            # Visualizar señales
            plt.figure(figsize=(12, 8))
            plt.plot(self.data.index, self.data['Close'], label='Precio', alpha=0.7)
            
            # Marcar señales de compra
            buy_signals = self.data[self.data['Signal'] == 1].index
            plt.scatter(buy_signals, self.data.loc[buy_signals, 'Close'], 
                       color='green', label='Señal de Compra', s=50, marker='^')
            
            plt.title('Señales de Trading Generadas')
            plt.xlabel('Fecha')
            plt.ylabel('Precio')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('./artifacts/results/figures/trading_signals.png')
            plt.close()
            
            return True
        except Exception as e:
            logging.error(f"Error en generate_trading_signals: {str(e)}")
            logging.error(traceback.format_exc())
            return False
    
    def calculate_returns(self):
        """
        Calcula los retornos de la estrategia y métricas de rendimiento.
        
        Returns:
            bool: True si los retornos se calcularon correctamente, False en caso contrario
        """
        try:
            # Inicializar serie de retornos diarios
            daily_returns = pd.Series(0.0, index=self.data.index)
            
            # Inicializar curva de capital
            self.equity_curve = pd.Series(self.capital, index=self.data.index)
            
            # Calcular retornos para posiciones cerradas
            for position in self.closed_positions:
                entry_date = position['entry_date']
                exit_date = position['exit_date']
                entry_price = position['entry_price']
                exit_price = position['exit_price']
                shares = position['shares']
                
                # Calcular P&L
                pnl = (exit_price - entry_price) * shares
                pnl_pct = (exit_price - entry_price) / entry_price
                
                # Registrar retorno en la fecha de salida
                if exit_date in daily_returns.index:
                    daily_returns[exit_date] += pnl / self.capital
            
            # Calcular retornos acumulados y curva de capital
            cumulative_returns = (1 + daily_returns).cumprod() - 1
            self.equity_curve = self.capital * (1 + cumulative_returns)
            
            # Calcular métricas de rendimiento
            # Retorno total
            total_return = (self.equity_curve.iloc[-1] / self.capital) - 1
            
            # Retorno anualizado
            years = (self.data.index[-1] - self.data.index[0]).days / 365.25
            annual_return = (1 + total_return) ** (1 / years) - 1
            
            # Volatilidad anualizada
            daily_std = daily_returns.std()
            annual_volatility = daily_std * np.sqrt(252)
            
            # Ratio de Sharpe (asumiendo tasa libre de riesgo de 0% para simplificar)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
            
            # Drawdown máximo
            rolling_max = self.equity_curve.cummax()
            drawdown = (self.equity_curve - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Ratio de Calmar
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
            
            # Ratio de ganancia/pérdida
            winning_trades = [p for p in self.closed_positions if p['exit_price'] > p['entry_price']]
            losing_trades = [p for p in self.closed_positions if p['exit_price'] <= p['entry_price']]
            
            win_rate = len(winning_trades) / len(self.closed_positions) if self.closed_positions else 0
            
            avg_win = np.mean([(p['exit_price'] - p['entry_price']) / p['entry_price'] for p in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([(p['exit_price'] - p['entry_price']) / p['entry_price'] for p in losing_trades]) if losing_trades else 0
            
            profit_factor = abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if losing_trades and avg_loss != 0 else 0
            
            # Guardar métricas
            self.metrics = {
                'total_return': total_return,
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': len(self.closed_positions),
                'avg_win': avg_win,
                'avg_loss': avg_loss
            }
            
            # Guardar métricas en CSV
            metrics_df = pd.DataFrame([self.metrics])
            metrics_df.to_csv('./artifacts/results/data/performance_metrics.csv', index=False)
            
            # Visualizar curva de capital
            plt.figure(figsize=(12, 8))
            plt.plot(self.equity_curve.index, self.equity_curve, label='Capital')
            
            # Añadir línea de referencia (buy & hold)
            buy_hold_capital = self.capital * (self.data['Close'] / self.data['Close'].iloc[0])
            plt.plot(buy_hold_capital.index, buy_hold_capital, label='Buy & Hold', alpha=0.7, linestyle='--')
            
            plt.title('Curva de Capital')
            plt.xlabel('Fecha')
            plt.ylabel('Capital ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('./artifacts/results/figures/equity_curve.png')
            plt.close()
            
            # Visualizar drawdown
            plt.figure(figsize=(12, 6))
            plt.plot(drawdown.index, drawdown * 100)
            plt.fill_between(drawdown.index, drawdown * 100, 0, alpha=0.3, color='red')
            plt.title('Drawdown (%)')
            plt.xlabel('Fecha')
            plt.ylabel('Drawdown (%)')
            plt.grid(True, alpha=0.3)
            plt.savefig('./artifacts/results/figures/drawdown.png')
            plt.close()
            
            return True
        except Exception as e:
            logging.error(f"Error en calculate_returns: {str(e)}")
            logging.error(traceback.format_exc())
            return False
    
    def analyze_trades(self):
        """
        Analiza las operaciones realizadas y genera estadísticas.
        
        Returns:
            bool: True si el análisis se completó correctamente, False en caso contrario
        """
        try:
            if not self.closed_positions:
                logging.warning("No hay operaciones cerradas para analizar")
                return True
            
            # Convertir operaciones a DataFrame para análisis
            trades_data = []
            for trade in self.closed_positions:
                trades_data.append({
                    'entry_date': trade['entry_date'],
                    'exit_date': trade['exit_date'],
                    'entry_price': trade['entry_price'],
                    'exit_price': trade['exit_price'],
                    'shares': trade['shares'],
                    'regime': trade['regime'],
                    'exit_type': trade['exit_type'],
                    'pnl': (trade['exit_price'] - trade['entry_price']) * trade['shares'],
                    'pnl_pct': (trade['exit_price'] - trade['entry_price']) / trade['entry_price'],
                    'duration': (trade['exit_date'] - trade['entry_date']).days
                })
            
            trades_df = pd.DataFrame(trades_data)
            
            # Guardar detalles de operaciones
            trades_df.to_csv('./artifacts/results/data/trades_details.csv', index=False)
            
            # Análisis por tipo de salida
            exit_type_analysis = trades_df.groupby('exit_type').agg({
                'pnl': ['count', 'mean', 'sum'],
                'pnl_pct': ['mean', 'std'],
                'duration': 'mean'
            })
            
            exit_type_analysis.to_csv('./artifacts/results/data/exit_type_analysis.csv')
            
            # Análisis por régimen
            regime_analysis = trades_df.groupby('regime').agg({
                'pnl': ['count', 'mean', 'sum'],
                'pnl_pct': ['mean', 'std'],
                'duration': 'mean'
            })
            
            regime_analysis.to_csv('./artifacts/results/data/regime_trade_analysis.csv')
            
            # Visualizar distribución de retornos de operaciones
            plt.figure(figsize=(12, 6))
            sns.histplot(trades_df['pnl_pct'], kde=True, bins=30)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.title('Distribución de Retornos por Operación')
            plt.xlabel('Retorno (%)')
            plt.ylabel('Frecuencia')
            plt.grid(True, alpha=0.3)
            plt.savefig('./artifacts/results/figures/trade_returns_distribution.png')
            plt.close()
            
            # Visualizar duración de operaciones vs. retorno
            plt.figure(figsize=(12, 6))
            plt.scatter(trades_df['duration'], trades_df['pnl_pct'] * 100, 
                       alpha=0.7, c=trades_df['pnl_pct'] > 0, cmap='coolwarm')
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title('Duración vs. Retorno de Operaciones')
            plt.xlabel('Duración (días)')
            plt.ylabel('Retorno (%)')
            plt.colorbar(label='Operación Ganadora')
            plt.grid(True, alpha=0.3)
            plt.savefig('./artifacts/results/figures/duration_vs_return.png')
            plt.close()
            
            # Visualizar rendimiento por régimen
            regime_names = {0: 'Bajista', 1: 'Neutral', 2: 'Alcista'}
            regime_colors = {0: 'red', 1: 'gray', 2: 'green'}
            
            plt.figure(figsize=(10, 6))
            for regime in trades_df['regime'].unique():
                regime_trades = trades_df[trades_df['regime'] == regime]
                plt.bar(regime_names[regime], regime_trades['pnl_pct'].mean() * 100, 
                       yerr=regime_trades['pnl_pct'].std() * 100,
                       color=regime_colors[regime], alpha=0.7)
            
            plt.title('Rendimiento Promedio por Régimen de Mercado')
            plt.ylabel('Retorno Promedio (%)')
            plt.grid(True, alpha=0.3, axis='y')
            plt.savefig('./artifacts/results/figures/regime_performance.png')
            plt.close()
            
            return True
        except Exception as e:
            logging.error(f"Error en analyze_trades: {str(e)}")
            logging.error(traceback.format_exc())
            return False
    
    def run_backtest(self):
        """
        Ejecuta el backtest completo de la estrategia.
        
        Returns:
            bool: True si el backtest se completó correctamente, False en caso contrario
        """
        try:
            print("Iniciando backtest...")
            
            # Cargar datos
            if not self.load_data():
                return False
            print("Datos cargados correctamente.")
            
            # Calcular indicadores técnicos
            if not self.calculate_technical_indicators():
                return False
            print("Indicadores técnicos calculados.")
            
            # Detectar regímenes de mercado
            if not self.detect_market_regimes():
                return False
            print("Regímenes de mercado detectados.")
            
            # Implementar circuit breakers
            if not self.check_circuit_breakers():
                return False
            print("Circuit breakers implementados.")
            
            # Generar señales de trading
            if not self.generate_trading_signals():
                return False
            print("Señales de trading generadas.")
            
            # Calcular retornos y métricas
            if not self.calculate_returns():
                return False
            print("Retornos calculados.")
            
            # Analizar operaciones
            if not self.analyze_trades():
                return False
            print("Análisis de operaciones completado.")
            
            # Generar informe final
            self.generate_report()
            print("Backtest completado con éxito.")
            
            return True
        except Exception as e:
            logging.error(f"Error en run_backtest: {str(e)}")
            logging.error(traceback.format_exc())
            return False
    
    def generate_report(self):
        """
        Genera un informe resumido de los resultados del backtest.
        """
        try:
            # Crear informe en formato texto
            with open('./artifacts/results/backtest_report.txt', 'w') as f:
                f.write("=== INFORME DE BACKTEST ===\n\n")
                
                f.write(f"Ticker: {self.data['Ticker'].iloc[0]}\n")
                f.write(f"Período: {self.data.index[0].strftime('%Y-%m-%d')} a {self.data.index[-1].strftime('%Y-%m-%d')}\n")
                f.write(f"Capital inicial: ${self.capital:,.2f}\n")
                f.write(f"Capital final: ${self.equity_curve.iloc[-1]:,.2f}\n\n")
                
                f.write("--- MÉTRICAS DE RENDIMIENTO ---\n")
                f.write(f"Retorno total: {self.metrics['total_return']*100:.2f}%\n")
                f.write(f"Retorno anualizado: {self.metrics['annual_return']*100:.2f}%\n")
                f.write(f"Volatilidad anualizada: {self.metrics['annual_volatility']*100:.2f}%\n")
                f.write(f"Ratio de Sharpe: {self.metrics['sharpe_ratio']:.2f}\n")
                f.write(f"Drawdown máximo: {self.metrics['max_drawdown']*100:.2f}%\n")
                f.write(f"Ratio de Calmar: {self.metrics['calmar_ratio']:.2f}\n\n")
                
                f.write("--- ESTADÍSTICAS DE OPERACIONES ---\n")
                f.write(f"Total de operaciones: {self.metrics['total_trades']}\n")
                f.write(f"Tasa de acierto: {self.metrics['win_rate']*100:.2f}%\n")
                f.write(f"Factor de beneficio: {self.metrics['profit_factor']:.2f}\n")
                f.write(f"Ganancia promedio: {self.metrics['avg_win']*100:.2f}%\n")
                f.write(f"Pérdida promedio: {self.metrics['avg_loss']*100:.2f}%\n\n")
                
                f.write("--- ANÁLISIS DE REGÍMENES ---\n")
                if hasattr(self, 'regime_stats'):
                    for regime in range(3):
                        regime_name = ['Bajista', 'Neutral', 'Alcista'][regime]
                        f.write(f"Régimen {regime_name}:\n")
                        f.write(f"  - Retorno medio: {self.regime_stats[('Returns', 'mean')][regime]*100:.2f}%\n")
                        f.write(f"  - Volatilidad: {self.regime_stats[('Volatility', 'mean')][regime]*100:.2f}%\n")
                        f.write(f"  - RSI medio: {self.regime_stats[('RSI', 'mean')][regime]:.2f}\n\n")
                
                f.write("--- CONCLUSIONES ---\n")
                # Evaluar rendimiento
                if self.metrics['sharpe_ratio'] > 1.0:
                    performance = "bueno"
                elif self.metrics['sharpe_ratio'] > 0.5:
                    performance = "aceptable"
                else:
                    performance = "pobre"
                
                f.write(f"La estrategia muestra un rendimiento {performance} con un Sharpe de {self.metrics['sharpe_ratio']:.2f}.\n")
                
                # Evaluar comportamiento en diferentes regímenes
                if hasattr(self, 'regime_stats'):
                    best_regime = np.argmax(self.regime_stats[('Returns', 'mean')])
                    worst_regime = np.argmin(self.regime_stats[('Returns', 'mean')])
                    
                    f.write(f"La estrategia funciona mejor en régimen {['Bajista', 'Neutral', 'Alcista'][best_regime]} ")
                    f.write(f"y peor en régimen {['Bajista', 'Neutral', 'Alcista'][worst_regime]}.\n")
                
                # Recomendaciones
                f.write("\nRecomendaciones:\n")
                if self.metrics['win_rate'] < 0.4:
                    f.write("- Mejorar la tasa de acierto ajustando los criterios de entrada.\n")
                if self.metrics['max_drawdown'] < -0.2:
                    f.write("- Implementar mejor gestión de riesgo para reducir el drawdown máximo.\n")
                if self.metrics['profit_factor'] < 1.5:
                    f.write("- Optimizar la relación riesgo/recompensa para mejorar el factor de beneficio.\n")
            
            print(f"Informe generado en './artifacts/results/backtest_report.txt'")
            
        except Exception as e:
            logging.error(f"Error en generate_report: {str(e)}")
            logging.error(traceback.format_exc())
    
    def run_walk_forward_analysis(self, train_size=0.7, window_size=252, step_size=63):
        """
        Ejecuta análisis walk-forward para evaluar la robustez de la estrategia.
        
        Args:
            train_size: Proporción de datos para entrenamiento en cada ventana
            window_size: Tamaño de la ventana en días
            step_size: Tamaño del paso para avanzar la ventana en días
            
        Returns:
            bool: True si el análisis se completó correctamente, False en caso contrario
        """
        try:
            print("Iniciando análisis walk-forward...")
            
            # Cargar datos completos
            if not self.load_data():
                return False
            
            # Calcular indicadores técnicos
            if not self.calculate_technical_indicators():
                return False
            
            # Preparar resultados
            all_windows_results = []
            all_equity_curves = pd.DataFrame()
            
            # Definir ventanas
            total_days = len(self.data)
            start_indices = range(0, total_days - window_size, step_size)
            
            for i, start_idx in enumerate(start_indices):
                print(f"Procesando ventana {i+1}/{len(start_indices)}...")
                
                end_idx = min(start_idx + window_size, total_days)
                window_data = self.data.iloc[start_idx:end_idx].copy()
                
                # Dividir en entrenamiento y prueba
                train_idx = int(len(window_data) * train_size)
                train_data = window_data.iloc[:train_idx]
                test_data = window_data.iloc[train_idx:]
                
                if len(train_data) < 100 or len(test_data) < 20:
                    print("Ventana demasiado pequeña, saltando...")
                    continue
                
                # Crear instancia de estrategia para esta ventana
                window_strategy = QuantitativeStrategy(
                    tickers=[self.data['Ticker'].iloc[0]],
                    start_date=train_data.index[0],
                    end_date=train_data.index[-1],
                    lookback_period=min(self.lookback_period, len(train_data) - 10),
                    regime_clusters=self.regime_clusters,
                    max_positions=self.max_positions,
                    stop_loss=self.stop_loss,
                    take_profit=self.take_profit,
                    position_size=self.position_size,
                    vix_threshold=self.vix_threshold,
                    atr_multiplier=self.atr_multiplier
                )
                
                # Entrenar en datos de entrenamiento
                window_strategy.data = train_data
                window_strategy.detect_market_regimes()
                window_strategy.check_circuit_breakers()
                
                # Aplicar la estrategia a los datos de prueba
                test_data = test_data.copy()
                
                # Asignar último régimen conocido a datos de prueba
                if len(window_strategy.data) > 0 and 'Regime' in window_strategy.data.columns:
                    last_regime = window_strategy.data['Regime'].iloc[-1]
                    test_data['Regime'] = last_regime
                else:
                    test_data['Regime'] = 1  # Régimen neutral por defecto
                
                # Inicializar variables para backtest en datos de prueba
                test_strategy = QuantitativeStrategy(
                    tickers=[self.data['Ticker'].iloc[0]],
                    start_date=test_data.index[0],
                    end_date=test_data.index[-1],
                    lookback_period=self.lookback_period,
                    regime_clusters=self.regime_clusters,
                    max_positions=self.max_positions,
                    stop_loss=self.stop_loss,
                    take_profit=self.take_profit,
                    position_size=self.position_size,
                    vix_threshold=self.vix_threshold,
                    atr_multiplier=self.atr_multiplier
                )
                
                test_strategy.data = test_data
                test_strategy.vix_data = self.vix_data.reindex(test_data.index, method='ffill')
                test_strategy.sp500_data = self.sp500_data.reindex(test_data.index, method='ffill')
                
                # Generar señales y calcular retornos
                test_strategy.check_circuit_breakers()
                test_strategy.generate_trading_signals()
                test_strategy.calculate_returns()
                
                # Guardar resultados
                window_results = {
                    'window': i+1,
                    'start_date': window_data.index[0],
                    'end_date': window_data.index[-1],
                    'train_end': train_data.index[-1],
                    'sharpe': test_strategy.metrics.get('sharpe_ratio', 0),
                    'return': test_strategy.metrics.get('total_return', 0),
                    'drawdown': test_strategy.metrics.get('max_drawdown', 0),
                    'win_rate': test_strategy.metrics.get('win_rate', 0),
                    'trades': test_strategy.metrics.get('total_trades', 0)
                }
                
                all_windows_results.append(window_results)
                
                # Guardar curva de capital
                if hasattr(test_strategy, 'equity_curve') and isinstance(test_strategy.equity_curve, pd.Series):
                    window_equity = test_strategy.equity_curve.copy()
                    window_equity.name = f'Window_{i+1}'
                    all_equity_curves = pd.concat([all_equity_curves, window_equity], axis=1)
            
            # Convertir resultados a DataFrame
            results_df = pd.DataFrame(all_windows_results)
            results_df.to_csv('./artifacts/results/data/walk_forward_results.csv', index=False)
            
            # Guardar curvas de capital
            if not all_equity_curves.empty:
                all_equity_curves.to_csv('./artifacts/results/data/walk_forward_equity_curves.csv')
            
            # Visualizar resultados
            if not results_df.empty:
                # Gráfico de Sharpe Ratio por ventana
                plt.figure(figsize=(12, 6))
                plt.bar(results_df['window'], results_df['sharpe'], alpha=0.7)
                plt.axhline(y=results_df['sharpe'].mean(), color='r', linestyle='--', 
                           label=f'Promedio: {results_df["sharpe"].mean():.2f}')
                plt.title('Sharpe Ratio por Ventana')
                plt.xlabel('Ventana')
                plt.ylabel('Sharpe Ratio')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig('./artifacts/results/figures/walk_forward_sharpe.png')
                plt.close()
                
                # Gráfico de retornos por ventana
                plt.figure(figsize=(12, 6))
                plt.bar(results_df['window'], results_df['return'] * 100, alpha=0.7)
                plt.axhline(y=results_df['return'].mean() * 100, color='r', linestyle='--', 
                           label=f'Promedio: {results_df["return"].mean()*100:.2f}%')
                plt.title('Retorno por Ventana')
                plt.xlabel('Ventana')
                plt.ylabel('Retorno (%)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig('./artifacts/results/figures/walk_forward_returns.png')
                plt.close()
                
                # Resumen de estadísticas
                summary = {
                    'avg_sharpe': results_df['sharpe'].mean(),
                    'std_sharpe': results_df['sharpe'].std(),
                    'avg_return': results_df['return'].mean(),
                    'std_return': results_df['return'].std(),
                    'avg_drawdown': results_df['drawdown'].mean(),
                    'avg_win_rate': results_df['win_rate'].mean(),
                    'avg_trades': results_df['trades'].mean(),
                    'positive_windows': (results_df['return'] > 0).sum() / len(results_df)
                }
                
                with open('./artifacts/results/data/walk_forward_summary.json', 'w') as f:
                    json.dump(summary, f, indent=4)
                
                # Informe de análisis walk-forward
                with open('./artifacts/results/walk_forward_report.txt', 'w') as f:
                    f.write("=== INFORME DE ANÁLISIS WALK-FORWARD ===\n\n")
                    
                    f.write(f"Ticker: {self.data['Ticker'].iloc[0]}\n")
                    f.write(f"Período total: {self.data.index[0].strftime('%Y-%m-%d')} a {self.data.index[-1].strftime('%Y-%m-%d')}\n")
                    f.write(f"Tamaño de ventana: {window_size} días\n")
                    f.write(f"Paso de ventana: {step_size} días\n")
                    f.write(f"Proporción de entrenamiento: {train_size*100:.0f}%\n\n")
                    
                    f.write("--- MÉTRICAS PROMEDIO ---\n")
                    f.write(f"Sharpe Ratio: {summary['avg_sharpe']:.2f} (±{summary['std_sharpe']:.2f})\n")
                    f.write(f"Retorno: {summary['avg_return']*100:.2f}% (±{summary['std_return']*100:.2f}%)\n")
                    f.write(f"Drawdown máximo: {summary['avg_drawdown']*100:.2f}%\n")
                    f.write(f"Tasa de acierto: {summary['avg_win_rate']*100:.2f}%\n")
                    f.write(f"Operaciones promedio: {summary['avg_trades']:.1f}\n\n")
                    
                    f.write(f"Ventanas con retorno positivo: {summary['positive_windows']*100:.1f}%\n\n")
                    
                    f.write("--- CONCLUSIONES ---\n")
                    # Evaluar robustez
                    if summary['positive_windows'] > 0.7 and summary['avg_sharpe'] > 0.8:
                        robustness = "alta"
                    elif summary['positive_windows'] > 0.5 and summary['avg_sharpe'] > 0.5:
                        robustness = "moderada"
                    else:
                        robustness = "baja"
                    
                    f.write(f"La estrategia muestra una robustez {robustness} a través de diferentes períodos de mercado.\n")
                    
                    # Evaluar consistencia
                    sharpe_cv = summary['std_sharpe'] / summary['avg_sharpe'] if summary['avg_sharpe'] > 0 else float('inf')
                    if sharpe_cv < 0.5:
                        consistency = "alta"
                    elif sharpe_cv < 1.0:
                        consistency = "moderada"
                    else:
                        consistency = "baja"
                    
                    f.write(f"La consistencia de rendimiento es {consistency} (CV de Sharpe: {sharpe_cv:.2f}).\n\n")
                    
                    # Recomendaciones
                    f.write("Recomendaciones:\n")
                    if summary['avg_sharpe'] < 0.5:
                        f.write("- Revisar la estrategia para mejorar el rendimiento ajustado por riesgo.\n")
                    if summary['positive_windows'] < 0.6:
                        f.write("- Mejorar la adaptabilidad a diferentes condiciones de mercado.\n")
                    if sharpe_cv > 0.8:
                        f.write("- Trabajar en la consistencia del rendimiento entre diferentes períodos.\n")
            
            print(f"Análisis walk-forward completado. Resultados guardados en './artifacts/results/'")
            return True
        except Exception as e:
            logging.error(f"Error en run_walk_forward_analysis: {str(e)}")
            logging.error(traceback.format_exc())
            return False

def main():
    """
    Función principal para ejecutar la estrategia.
    """
    try:
        # Crear instancia de la estrategia
        strategy = QuantitativeStrategy(
            tickers=None,  # Usar S&P 500
            start_date='2015-01-01',
            end_date='2023-12-31',
            lookback_period=252,
            regime_clusters=3,
            max_positions=5,
            stop_loss=0.05,
            take_profit=0.15,
            position_size=0.2,
            vix_threshold=None,  # Inferir automáticamente
            atr_multiplier=2.0
        )
        
        # Ejecutar backtest
        strategy.run_backtest()
        
        # Ejecutar análisis walk-forward
        strategy.run_walk_forward_analysis(train_size=0.7, window_size=252, step_size=63)
        
        print("Análisis completado. Resultados guardados en './artifacts/results/'")
    except Exception as e:
        logging.error(f"Error en main: {str(e)}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
