import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
from tqdm import tqdm
from datetime import datetime, timedelta
import warnings
from sklearn.linear_model import LinearRegression
import requests
from io import StringIO

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

def get_sp500_tickers():
    """
    Obtiene la lista de tickers del S&P 500 desde Wikipedia.
    
    Returns:
        dict: Diccionario con tickers como claves y sectores como valores
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        
        # Crear diccionario de ticker -> sector
        ticker_sector_dict = dict(zip(df['Symbol'], df['GICS Sector']))
        
        # Limpiar tickers (algunos tienen puntos que yfinance no maneja bien)
        ticker_sector_dict = {ticker.replace('.', '-'): sector 
                             for ticker, sector in ticker_sector_dict.items()}
        
        return ticker_sector_dict
    
    except Exception as e:
        logging.error(f"Error obteniendo tickers del S&P 500: {str(e)}")
        # Devolver un diccionario vacío en caso de error
        return {}

def download_data(tickers, start_date, end_date):
    """
    Descarga datos históricos para los tickers especificados.
    
    Args:
        tickers (list): Lista de tickers a descargar
        start_date (str): Fecha de inicio en formato 'YYYY-MM-DD'
        end_date (str): Fecha de fin en formato 'YYYY-MM-DD'
        
    Returns:
        tuple: (precios, volumen)
    """
    try:
        # Añadir un margen de tiempo para calcular características que requieren datos históricos
        extended_start = (pd.to_datetime(start_date) - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
        
        # Descargar datos
        data = yf.download(tickers, start=extended_start, end=end_date, progress=False)
        
        # Extraer precios de cierre y volumen
        prices = data['Close']
        volume = data['Volume']
        
        # Verificar si hay datos
        if prices.empty or volume.empty:
            raise ValueError("No se pudieron obtener datos para los tickers especificados")
        
        # Eliminar columnas con más del 30% de valores NaN
        valid_columns = prices.columns[prices.isna().mean() < 0.3]
        prices = prices[valid_columns]
        volume = volume[valid_columns]
        
        # Llenar valores NaN con el último valor disponible
        prices = prices.fillna(method='ffill')
        volume = volume.fillna(method='ffill')
        
        # Filtrar para el período solicitado
        prices = prices.loc[start_date:end_date]
        volume = volume.loc[start_date:end_date]
        
        return prices, volume
    
    except Exception as e:
        logging.error(f"Error descargando datos: {str(e)}")
        # Devolver DataFrames vacíos en caso de error
        return pd.DataFrame(), pd.DataFrame()

def calculate_returns(prices, periods):
    """
    Calcula los retornos para diferentes períodos de tiempo.
    
    Args:
        prices (DataFrame): DataFrame con precios de cierre
        periods (dict): Diccionario con nombres de períodos y número de días
        
    Returns:
        dict: Diccionario con retornos para cada período
    """
    try:
        returns = {}
        
        for period_name, days in periods.items():
            # Calcular retornos para el período especificado
            period_returns = prices.pct_change(periods=days).shift(1)
            returns[period_name] = period_returns
        
        return returns
    
    except Exception as e:
        logging.error(f"Error calculando retornos: {str(e)}")
        # Devolver un diccionario vacío en caso de error
        return {}

def calculate_features(prices, volume, returns):
    """
    Calcula características para cada ticker en cada fecha.
    
    Args:
        prices (DataFrame): DataFrame con precios de cierre
        volume (DataFrame): DataFrame con volumen
        returns (dict): Diccionario con retornos para diferentes períodos
        
    Returns:
        DataFrame: DataFrame con características
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
        
        # Calcular autocorrelación (21 días)
        autocorr = pd.DataFrame(index=prices.index, columns=prices.columns)
        
        # Necesitamos al menos 22 días de datos para calcular autocorrelación
        min_required_days = 22
        if len(daily_returns) >= min_required_days:
            for ticker in prices.columns:
                for i in range(min_required_days, len(daily_returns)):
                    window = daily_returns.iloc[i-21:i][ticker].dropna()
                    if len(window) > 5:  # Necesitamos al menos algunos puntos para la autocorrelación
                        try:
                            autocorr.iloc[i][ticker] = window.autocorr(lag=1)
                        except:
                            autocorr.iloc[i][ticker] = 0
        
        # Asegurarse de que tenemos suficientes datos
        valid_dates = prices.index[21:]
        
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
                    'autocorr': autocorr.loc[date, ticker] if not pd.isna(autocorr.loc[date, ticker]) else 0
                }
                
                features_list.append(features_dict)
        
        # Crear DataFrame con características
        features_df = pd.DataFrame(features_list)
        
        # Manejar valores NaN
        features_df = features_df.fillna(0)
        
        return features_df
    
    except Exception as e:
        logging.error(f"Error calculando características: {str(e)}")
        # Devolver un DataFrame vacío en caso de error
        return pd.DataFrame()

def detect_market_regimes(prices, n_regimes=3):
    """
    Detecta regímenes de mercado utilizando clustering.
    
    Args:
        prices (DataFrame): DataFrame con precios de cierre
        n_regimes (int): Número de regímenes a detectar
        
    Returns:
        Series: Serie con regímenes para cada fecha
    """
    try:
        # Calcular retornos del mercado (promedio de todos los activos)
        market_returns = prices.pct_change().mean(axis=1).dropna()
        
        # Verificar si hay suficientes datos
        if len(market_returns) < 42:  # Necesitamos al menos 42 días para calcular volatilidad
            logging.warning("Datos insuficientes para detectar regímenes. Usando régimen por defecto.")
            return pd.Series(0, index=prices.index)
        
        # Calcular volatilidad rodante (21 días)
        rolling_vol = market_returns.rolling(window=21).std().dropna()
        
        # Crear características para el modelo
        features = pd.DataFrame({
            'returns': market_returns[rolling_vol.index],
            'volatility': rolling_vol
        })
        
        # Normalizar características
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Aplicar K-means clustering
        kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        regimes = kmeans.fit_predict(features_scaled)
        
        # Crear Serie con regímenes
        regime_series = pd.Series(regimes, index=features.index)
        
        # Propagar regímenes a todas las fechas
        full_regime_series = pd.Series(index=prices.index)
        
        # Para cada fecha en el índice de precios
        for date in prices.index:
            # Si la fecha está en el índice de regímenes, usar ese régimen
            if date in regime_series.index:
                full_regime_series[date] = regime_series[date]
            # Si no, usar el último régimen disponible
            elif date > regime_series.index[0]:
                # Encontrar la fecha más reciente en el índice de regímenes
                last_date = regime_series.index[regime_series.index < date][-1]
                full_regime_series[date] = regime_series[last_date]
            # Si la fecha es anterior al primer régimen, usar el primer régimen
            else:
                full_regime_series[date] = regime_series.iloc[0]
        
        return full_regime_series
    
    except Exception as e:
        logging.error(f"Error detectando regímenes de mercado: {str(e)}")
        # Devolver una serie con régimen por defecto en caso de error
        return pd.Series(0, index=prices.index)

def generate_signals(features, market_regimes):
    """
    Genera señales de trading basadas en características y regímenes de mercado.
    
    Args:
        features (DataFrame): DataFrame con características
        market_regimes (Series): Serie con regímenes de mercado
        
    Returns:
        DataFrame: DataFrame con señales para cada ticker en cada fecha
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
            regime = market_regimes.get(date, 0)  # Usar régimen 0 por defecto si no hay dato
            
            # Calcular señales según el régimen
            if regime == 0:  # Régimen de baja volatilidad
                # En régimen de baja volatilidad, dar más peso a momentum de largo plazo
                momentum_signal = (
                    0.1 * date_features['momentum_1m'] +
                    0.2 * date_features['momentum_3m'] +
                    0.3 * date_features['momentum_6m'] +
                    0.4 * date_features['momentum_12m']
                )
            elif regime == 1:  # Régimen de volatilidad media
                # En régimen de volatilidad media, equilibrar pesos
                momentum_signal = (
                    0.25 * date_features['momentum_1m'] +
                    0.25 * date_features['momentum_3m'] +
                    0.25 * date_features['momentum_6m'] +
                    0.25 * date_features['momentum_12m']
                )
            else:  # Régimen de alta volatilidad
                # En régimen de alta volatilidad, dar más peso a momentum de corto plazo
                momentum_signal = (
                    0.4 * date_features['momentum_1m'] +
                    0.3 * date_features['momentum_3m'] +
                    0.2 * date_features['momentum_6m'] +
                    0.1 * date_features['momentum_12m']
                )
            
            # Ajustar señales por volatilidad (penalizar alta volatilidad)
            vol_adjustment = 1 / (1 + date_features['volatility'])
            
            # Ajustar señales por volumen (favorecer alto volumen y cambios positivos)
            volume_adjustment = (
                date_features['avg_volume'] / date_features['avg_volume'].mean() *
                (1 + date_features['volume_change'])
            )
            volume_adjustment = volume_adjustment / volume_adjustment.mean()  # Normalizar
            
            # Combinar señales
            combined_signal = momentum_signal * vol_adjustment * volume_adjustment
            
            # Crear DataFrame con señales para esta fecha
            date_signals = pd.DataFrame({
                'date': date,
                'ticker': date_features['ticker'],
                'signal': combined_signal,
                'autocorr': date_features['autocorr'],
                'regime': regime
            })
            
            # Añadir a DataFrame de señales
            signals = pd.concat([signals, date_signals], ignore_index=True)
        
        return signals
    
    except Exception as e:
        logging.error(f"Error generando señales: {str(e)}")
        # Devolver un DataFrame vacío en caso de error
        return pd.DataFrame()

def adjust_for_autocorrelation(signals):
    """
    Ajusta las señales basándose en la autocorrelación de los retornos.
    
    Args:
        signals (DataFrame): DataFrame con señales
        
    Returns:
        DataFrame: DataFrame con señales ajustadas
    """
    try:
        # Crear copia para no modificar el original
        adjusted_signals = signals.copy()
        
        # Para cada fecha
        for date in adjusted_signals['date'].unique():
            # Obtener señales para esta fecha
            date_signals = adjusted_signals[adjusted_signals['date'] == date]
            
            # Para cada ticker
            for idx, row in date_signals.iterrows():
                # Obtener autocorrelación
                autocorr = row['autocorr']
                
                # Ajustar señal según autocorrelación
                if not pd.isna(autocorr):
                    # Si autocorrelación es positiva, reducir la señal
                    if autocorr > 0:
                        adjustment_factor = 1 / (1 + 2 * autocorr)
                    # Si autocorrelación es negativa, aumentar la señal
                    else:
                        adjustment_factor = 1 - 2 * autocorr
                    
                    # Aplicar ajuste
                    adjusted_signals.loc[idx, 'signal'] *= adjustment_factor
        
        return adjusted_signals
    
    except Exception as e:
        logging.error(f"Error ajustando por autocorrelación: {str(e)}")
        # Devolver señales sin ajustar en caso de error
        return signals

def combine_signals(signals, lookback_window=63):
    """
    Combina señales utilizando pesos adaptativos basados en rendimiento reciente.
    
    Args:
        signals (DataFrame): DataFrame con señales
        lookback_window (int): Ventana para evaluar rendimiento de señales
        
    Returns:
        DataFrame: DataFrame con señales combinadas
    """
    try:
        # Crear copia para no modificar el original
        combined_signals = signals.copy()
        
        # Obtener fechas únicas ordenadas
        dates = sorted(combined_signals['date'].unique())
        
        # Si no hay suficientes fechas, devolver señales sin combinar
        if len(dates) <= lookback_window:
            return combined_signals
        
        # Para cada fecha después de la ventana de lookback
        for i in range(lookback_window, len(dates)):
            current_date = dates[i]
            
            # Obtener fechas en la ventana de lookback
            lookback_dates = dates[i-lookback_window:i]
            
            # Obtener señales para la fecha actual
            current_signals = combined_signals[combined_signals['date'] == current_date]
            
            # Para cada ticker en las señales actuales
            for ticker in current_signals['ticker'].unique():
                # Obtener señales históricas para este ticker
                ticker_history = combined_signals[
                    (combined_signals['ticker'] == ticker) &
                    (combined_signals['date'].isin(lookback_dates))
                ]
                
                # Si no hay suficiente historia, continuar con el siguiente ticker
                if len(ticker_history) < lookback_window / 2:
                    continue
                
                # Calcular correlación entre señal y régimen
                signal_regime_corr = np.corrcoef(
                    ticker_history['signal'],
                    ticker_history['regime']
                )[0, 1]
                
                # Ajustar señal según correlación con régimen
                if not pd.isna(signal_regime_corr):
                    # Obtener índice de la señal actual
                    idx = combined_signals[
                        (combined_signals['date'] == current_date) &
                        (combined_signals['ticker'] == ticker)
                    ].index
                    
                    # Si correlación es positiva, aumentar señal en regímenes altos
                    if signal_regime_corr > 0:
                        regime_factor = 1 + 0.2 * current_signals.loc[
                            current_signals['ticker'] == ticker, 'regime'
                        ].values[0]
                    # Si correlación es negativa, reducir señal en regímenes altos
                    else:
                        regime_factor = 1 - 0.2 * current_signals.loc[
                            current_signals['ticker'] == ticker, 'regime'
                        ].values[0]
                    
                    # Aplicar ajuste
                    combined_signals.loc[idx, 'signal'] *= regime_factor
        
        return combined_signals
    
    except Exception as e:
        logging.error(f"Error combinando señales: {str(e)}")
        # Devolver señales sin combinar en caso de error
        return signals

def construct_portfolio(signals, sectors, date, top_pct=0.1, max_sector_exposure=0.25):
    """
    Construye un portafolio basado en señales para una fecha específica.
    
    Args:
        signals (DataFrame): DataFrame con señales
        sectors (dict): Diccionario con sectores para cada ticker
        date (datetime): Fecha para la cual construir el portafolio
        top_pct (float): Porcentaje de tickers con mejores señales a incluir
        max_sector_exposure (float): Exposición máxima por sector
        
    Returns:
        dict: Diccionario con pesos para cada ticker
    """
    try:
        # Obtener señales para la fecha especificada
        date_signals = signals[signals['date'] == date].copy()
        
        # Si no hay señales para esta fecha, devolver diccionario vacío
        if date_signals.empty:
            return {}
        
        # Añadir sector a cada ticker
        date_signals['sector'] = date_signals['ticker'].map(lambda x: sectors.get(x, 'Unknown'))
        
        # Ordenar por señal (de mayor a menor)
        date_signals = date_signals.sort_values('signal', ascending=False)
        
        # Seleccionar top_pct% de tickers
        n_tickers = int(len(date_signals) * top_pct)
        top_tickers = date_signals.head(n_tickers)
        
        # Calcular exposición por sector
        sector_exposure = top_tickers.groupby('sector').size() / n_tickers
        
        # Ajustar pesos para limitar exposición por sector
        weights = {}
        
        # Para cada ticker en top_tickers
        for _, row in top_tickers.iterrows():
            ticker = row['ticker']
            sector = row['sector']
            
            # Si la exposición del sector excede el máximo, reducir peso
            if sector_exposure[sector] > max_sector_exposure:
                weight = row['signal'] * (max_sector_exposure / sector_exposure[sector])
            else:
                weight = row['signal']
            
            weights[ticker] = weight
        
        # Normalizar pesos para que sumen 1
        total_weight = sum(weights.values())
        
        if total_weight > 0:
            weights = {ticker: weight / total_weight for ticker, weight in weights.items()}
        
        return weights
    
    except Exception as e:
        logging.error(f"Error construyendo portafolio para {date}: {str(e)}")
        # Devolver diccionario vacío en caso de error
        return {}

def calculate_portfolio_returns(prices, portfolio_weights, start_date, end_date):
    """
    Calcula los retornos del portafolio.
    
    Args:
        prices (DataFrame): DataFrame con precios de cierre
        portfolio_weights (dict): Diccionario con pesos para cada fecha
        start_date (datetime): Fecha de inicio
        end_date (datetime): Fecha de fin
        
    Returns:
        Series: Serie con retornos del portafolio
    """
    try:
        # Crear serie para almacenar retornos
        strategy_returns = pd.Series(index=pd.date_range(start=start_date, end=end_date, freq='B'))
        
        # Filtrar fechas de trading disponibles
        trading_dates = prices.index
        trading_dates = trading_dates[(trading_dates >= start_date) & (trading_dates <= end_date)]
        
        if len(trading_dates) < 2:
            logging.warning("Insuficientes fechas de trading para calcular retornos")
            return pd.Series(index=pd.date_range(start=start_date, end=end_date, freq='B'))
        
        # Verificar si hay pesos para alguna fecha
        valid_dates = [d for d in trading_dates if d in portfolio_weights]
        if not valid_dates:
            logging.warning("No hay pesos de portafolio para ninguna fecha en el período")
            return pd.Series(index=pd.date_range(start=start_date, end=end_date, freq='B'))
        
        # Calcular retornos diarios
        daily_returns = prices.pct_change()
        
        # Inicializar pesos actuales
        current_weights = None
        last_rebalance_date = None
        
        # Para cada fecha de trading
        for i in range(1, len(trading_dates)):
            current_date = trading_dates[i]
            previous_date = trading_dates[i-1]
            
            # Si es fecha de rebalanceo o primera fecha, actualizar pesos
            if current_date in portfolio_weights:
                current_weights = portfolio_weights[current_date]
                last_rebalance_date = current_date
            
            # Si no hay pesos actuales, continuar
            if current_weights is None:
                continue
            
            # Calcular retorno del portafolio para esta fecha
            portfolio_return = 0
            
            for ticker, weight in current_weights.items():
                # Verificar si el ticker está en los datos
                if ticker in daily_returns.columns:
                    # Obtener retorno para este ticker
                    ticker_return = daily_returns.loc[current_date, ticker]
                    
                    # Si no es NaN, añadir al retorno del portafolio
                    if not pd.isna(ticker_return):
                        portfolio_return += weight * ticker_return
            
            # Guardar retorno del portafolio
            strategy_returns[current_date] = portfolio_return
        
        # Eliminar NaN
        strategy_returns = strategy_returns.dropna()
        
        return strategy_returns
    
    except Exception as e:
        logging.error(f"Error calculando retornos del portafolio: {str(e)}")
        # Devolver serie vacía en caso de error
        return pd.Series()

def calculate_performance_metrics(returns, benchmark_returns=None):
    """
    Calcula métricas de rendimiento para una serie de retornos.
    
    Args:
        returns (Series): Serie con retornos
        benchmark_returns (Series, optional): Serie con retornos del benchmark
        
    Returns:
        dict: Diccionario con métricas de rendimiento
    """
    try:
        # Verificar si hay retornos
        if returns.empty:
            return {
                'annualized_return': 0,
                'annualized_volatility': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'information_ratio': 0
            }
        
        # Calcular retorno acumulado
        cumulative_return = (1 + returns).cumprod() - 1
        
        # Calcular retorno anualizado
        n_years = len(returns) / 252
        annualized_return = (1 + cumulative_return.iloc[-1]) ** (1 / n_years) - 1
        
        # Calcular volatilidad anualizada
        annualized_volatility = returns.std() * np.sqrt(252)
        
        # Calcular Sharpe ratio
        risk_free_rate = 0.02  # Tasa libre de riesgo (2%)
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
        
        # Calcular máximo drawdown
        peak = cumulative_return.cummax()
        drawdown = (cumulative_return - peak) / (1 + peak)
        max_drawdown = drawdown.min()
        
        # Calcular win rate
        win_rate = (returns > 0).mean()
        
        # Calcular Information Ratio si hay benchmark
        information_ratio = 0
        if benchmark_returns is not None:
            # Alinear fechas
            aligned_returns = returns.reindex(benchmark_returns.index).dropna()
            aligned_benchmark = benchmark_returns.reindex(aligned_returns.index)
            
            # Calcular tracking error
            tracking_error = (aligned_returns - aligned_benchmark).std() * np.sqrt(252)
            
            # Calcular Information Ratio
            if tracking_error > 0:
                information_ratio = (aligned_returns.mean() - aligned_benchmark.mean()) * 252 / tracking_error
        
        return {
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'information_ratio': information_ratio
        }
    
    except Exception as e:
        logging.error(f"Error calculando métricas de rendimiento: {str(e)}")
        # Devolver métricas vacías en caso de error
        return {
            'annualized_return': 0,
            'annualized_volatility': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'information_ratio': 0
        }

def plot_performance(strategy_returns, benchmark_returns=None, title='Strategy Performance'):
    """
    Genera gráfico de rendimiento.
    
    Args:
        strategy_returns (Series): Serie con retornos de la estrategia
        benchmark_returns (Series, optional): Serie con retornos del benchmark
        title (str): Título del gráfico
        
    Returns:
        None
    """
    try:
        plt.figure(figsize=(12, 6))
        
        # Calcular retorno acumulado
        strategy_cumulative = (1 + strategy_returns).cumprod() - 1
        
        # Graficar retorno acumulado de la estrategia
        plt.plot(strategy_cumulative.index, strategy_cumulative.values, label='Strategy')
        
        # Si hay benchmark, graficar también
        if benchmark_returns is not None:
            # Alinear fechas
            aligned_benchmark = benchmark_returns.reindex(strategy_returns.index)
            
            # Calcular retorno acumulado del benchmark
            benchmark_cumulative = (1 + aligned_benchmark).cumprod() - 1
            
            # Graficar retorno acumulado del benchmark
            plt.plot(benchmark_cumulative.index, benchmark_cumulative.values, label='Benchmark')
        
        # Añadir título y etiquetas
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        
        # Guardar gráfico
        plt.savefig(f'./artifacts/results/figures/{title.replace(" ", "_")}.png')
        plt.close()
    
    except Exception as e:
        logging.error(f"Error generando gráfico de rendimiento: {str(e)}")

def plot_drawdown(returns, title='Drawdown Analysis'):
    """
    Genera gráfico de drawdown.
    
    Args:
        returns (Series): Serie con retornos
        title (str): Título del gráfico
        
    Returns:
        None
    """
    try:
        # Calcular retorno acumulado
        cumulative_return = (1 + returns).cumprod() - 1
        
        # Calcular drawdown
        peak = cumulative_return.cummax()
        drawdown = (cumulative_return - peak) / (1 + peak)
        
        plt.figure(figsize=(12, 6))
        
        # Graficar drawdown
        plt.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
        plt.plot(drawdown.index, drawdown.values, color='red', alpha=0.5)
        
        # Añadir título y etiquetas
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.grid(True)
        
        # Guardar gráfico
        plt.savefig(f'./artifacts/results/figures/{title.replace(" ", "_")}.png')
        plt.close()
    
    except Exception as e:
        logging.error(f"Error generando gráfico de drawdown: {str(e)}")

def plot_regime_performance(returns, regimes, title='Performance by Regime'):
    """
    Genera gráfico de rendimiento por régimen.
    
    Args:
        returns (Series): Serie con retornos
        regimes (Series): Serie con regímenes
        title (str): Título del gráfico
        
    Returns:
        None
    """
    try:
        # Alinear fechas
        aligned_regimes = regimes.reindex(returns.index)
        
        # Crear DataFrame con retornos y regímenes
        df = pd.DataFrame({
            'returns': returns,
            'regime': aligned_regimes
        })
        
        # Calcular retorno promedio por régimen
        regime_returns = df.groupby('regime')['returns'].mean() * 252  # Anualizado
        
        plt.figure(figsize=(10, 6))
        
        # Graficar retorno por régimen
        bars = plt.bar(regime_returns.index, regime_returns.values)
        
        # Colorear barras según régimen
        colors = ['green', 'yellow', 'red']
        for i, bar in enumerate(bars):
            if i < len(colors):
                bar.set_color(colors[i])
        
        # Añadir título y etiquetas
        plt.title(title)
        plt.xlabel('Regime')
        plt.ylabel('Annualized Return')
        plt.xticks(regime_returns.index)
        plt.grid(True, axis='y')
        
        # Guardar gráfico
        plt.savefig(f'./artifacts/results/figures/{title.replace(" ", "_")}.png')
        plt.close()
    
    except Exception as e:
        logging.error(f"Error generando gráfico de rendimiento por régimen: {str(e)}")

def plot_sector_exposure(portfolio_weights, sectors, date, title='Sector Exposure'):
    """
    Genera gráfico de exposición por sector.
    
    Args:
        portfolio_weights (dict): Diccionario con pesos para una fecha
        sectors (dict): Diccionario con sectores para cada ticker
        date (datetime): Fecha para la cual mostrar exposición
        title (str): Título del gráfico
        
    Returns:
        None
    """
    try:
        # Calcular exposición por sector
        sector_exposure = {}
        
        for ticker, weight in portfolio_weights.items():
            sector = sectors.get(ticker, 'Unknown')
            sector_exposure[sector] = sector_exposure.get(sector, 0) + weight
        
        # Ordenar sectores por exposición
        sorted_sectors = sorted(sector_exposure.items(), key=lambda x: x[1], reverse=True)
        
        # Extraer sectores y exposiciones
        sector_names = [s[0] for s in sorted_sectors]
        exposures = [s[1] for s in sorted_sectors]
        
        plt.figure(figsize=(12, 6))
        
        # Graficar exposición por sector
        bars = plt.barh(sector_names, exposures)
        
        # Añadir título y etiquetas
        plt.title(f'{title} - {date.strftime("%Y-%m-%d")}')
        plt.xlabel('Exposure')
        plt.ylabel('Sector')
        plt.grid(True, axis='x')
        
        # Guardar gráfico
        plt.savefig(f'./artifacts/results/figures/{title.replace(" ", "_")}_{date.strftime("%Y%m%d")}.png')
        plt.close()
    
    except Exception as e:
        logging.error(f"Error generando gráfico de exposición por sector: {str(e)}")

def backtest_strategy(tickers, sectors, start_date, end_date, rebalance_freq='M'):
    """
    Realiza un backtest de la estrategia.
    
    Args:
        tickers (list): Lista de tickers
        sectors (dict): Diccionario con sectores para cada ticker
        start_date (str): Fecha de inicio en formato 'YYYY-MM-DD'
        end_date (str): Fecha de fin en formato 'YYYY-MM-DD'
        rebalance_freq (str): Frecuencia de rebalanceo ('D', 'W', 'M', etc.)
        
    Returns:
        tuple: (retornos de la estrategia, retornos del benchmark, métricas)
    """
    try:
        print("Iniciando backtest...")
        
        # Descargar datos
        prices, volume = download_data(tickers, start_date, end_date)
        
        # Verificar si hay datos
        if prices.empty or volume.empty:
            raise ValueError("No se pudieron obtener datos para los tickers especificados")
        
        # Descargar datos del benchmark (S&P 500)
        benchmark_data = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
        benchmark_returns = benchmark_data['Close'].pct_change().dropna()
        
        # Calcular retornos para diferentes períodos
        periods = {
            '1M': 21,
            '3M': 63,
            '6M': 126,
            '12M': 252
        }
        returns = calculate_returns(prices, periods)
        
        # Calcular características
        print("Calculando características...")
        features = calculate_features(prices, volume, returns)
        
        # Detectar regímenes de mercado
        print("Detectando regímenes de mercado...")
        market_regimes = detect_market_regimes(prices)
        
        # Generar señales
        print("Generando señales...")
        signals = generate_signals(features, market_regimes)
        
        # Ajustar señales por autocorrelación
        print("Ajustando señales por autocorrelación...")
        adjusted_signals = adjust_for_autocorrelation(signals)
        
        # Combinar señales
        print("Combinando señales...")
        combined_signals = combine_signals(adjusted_signals)
        
        # Determinar fechas de rebalanceo
        rebalance_dates = pd.date_range(start=start_date, end=end_date, freq=rebalance_freq)
        rebalance_dates = rebalance_dates[rebalance_dates.isin(prices.index)]
        
        # Construir portafolios para cada fecha de rebalanceo
        print("Construyendo portafolios...")
        portfolio_weights = {}
        
        for date in rebalance_dates:
            # Verificar si hay señales para esta fecha
            date_signals = combined_signals[combined_signals['date'] == date]
            
            if not date_signals.empty:
                # Construir portafolio
                weights = construct_portfolio(combined_signals, sectors, date)
                
                # Guardar pesos
                portfolio_weights[date] = weights
        
        # Calcular retornos del portafolio
        print("Calculando retornos...")
        strategy_returns = calculate_portfolio_returns(
            prices,
            portfolio_weights,
            pd.to_datetime(start_date),
            pd.to_datetime(end_date)
        )
        
        # Calcular métricas de rendimiento
        print("Calculando métricas de rendimiento...")
        metrics = calculate_performance_metrics(strategy_returns, benchmark_returns)
        
        # Generar gráficos
        print("Generando gráficos...")
        plot_performance(strategy_returns, benchmark_returns, title='Strategy vs Benchmark')
        plot_drawdown(strategy_returns, title='Strategy Drawdown')
        plot_regime_performance(strategy_returns, market_regimes, title='Performance by Regime')
        
        # Guardar métricas en CSV
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv('./artifacts/results/data/performance_metrics.csv', index=False)
        
        # Guardar retornos en CSV
        strategy_returns.to_csv('./artifacts/results/data/strategy_returns.csv')
        benchmark_returns.to_csv('./artifacts/results/data/benchmark_returns.csv')
        
        # Guardar exposición sectorial para la última fecha de rebalanceo
        if rebalance_dates.size > 0:
            last_rebalance = rebalance_dates[-1]
            if last_rebalance in portfolio_weights:
                plot_sector_exposure(
                    portfolio_weights[last_rebalance],
                    sectors,
                    last_rebalance,
                    title='Last Rebalance Sector Exposure'
                )
        
        print("Backtest completado.")
        
        return strategy_returns, benchmark_returns, metrics
    
    except Exception as e:
        logging.error(f"Error en backtest: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        # Devolver valores vacíos en caso de error
        return pd.Series(), pd.Series(), {}

def walk_forward_validation(tickers, sectors, start_date, end_date, train_window=252, test_window=63):
    """
    Realiza validación walk-forward de la estrategia.
    
    Args:
        tickers (list): Lista de tickers
        sectors (dict): Diccionario con sectores para cada ticker
        start_date (str): Fecha de inicio en formato 'YYYY-MM-DD'
        end_date (str): Fecha de fin en formato 'YYYY-MM-DD'
        train_window (int): Tamaño de la ventana de entrenamiento en días
        test_window (int): Tamaño de la ventana de prueba en días
        
    Returns:
        tuple: (retornos de la estrategia, retornos del benchmark, métricas)
    """
    try:
        print("Iniciando validación walk-forward...")
        
        # Convertir fechas a datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Descargar datos para todo el período
        prices, volume = download_data(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        # Verificar si hay datos
        if prices.empty or volume.empty:
            raise ValueError("No se pudieron obtener datos para los tickers especificados")
        
        # Descargar datos del benchmark (S&P 500)
        benchmark_data = yf.download('^GSPC', start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
        benchmark_returns = benchmark_data['Close'].pct_change().dropna()
        
        # Obtener fechas de trading
        trading_dates = prices.index
        
        # Inicializar variables para almacenar resultados
        all_strategy_returns = pd.Series()
        all_portfolio_weights = {}
        
        # Para cada ventana de validación
        current_start = start_date
        
        while current_start + pd.Timedelta(days=train_window + test_window) <= end_date:
            # Definir ventanas de entrenamiento y prueba
            train_end = current_start + pd.Timedelta(days=train_window)
            test_end = train_end + pd.Timedelta(days=test_window)
            
            # Ajustar a fechas de trading disponibles
            train_end = trading_dates[trading_dates <= train_end][-1]
            test_end = trading_dates[trading_dates <= test_end][-1]
            
            print(f"Entrenando: {current_start.strftime('%Y-%m-%d')} a {train_end.strftime('%Y-%m-%d')}")
            print(f"Probando: {train_end.strftime('%Y-%m-%d')} a {test_end.strftime('%Y-%m-%d')}")
            
            # Calcular retornos para diferentes períodos
            periods = {
                '1M': 21,
                '3M': 63,
                '6M': 126,
                '12M': 252
            }
            returns = calculate_returns(prices, periods)
            
            # Calcular características para el período de entrenamiento
            train_features = calculate_features(
                prices.loc[:train_end],
                volume.loc[:train_end],
                {k: v.loc[:train_end] for k, v in returns.items()}
            )
            
            # Detectar regímenes de mercado para el período de entrenamiento
            market_regimes = detect_market_regimes(prices.loc[:train_end])
            
            # Generar señales para el período de entrenamiento
            signals = generate_signals(train_features, market_regimes)
            
            # Ajustar señales por autocorrelación
            adjusted_signals = adjust_for_autocorrelation(signals)
            
            # Combinar señales
            combined_signals = combine_signals(adjusted_signals)
            
            # Determinar fechas de rebalanceo para el período de prueba
            test_dates = trading_dates[(trading_dates > train_end) & (trading_dates <= test_end)]
            
            # Construir portafolios para cada fecha de prueba
            for date in test_dates:
                # Obtener último régimen conocido
                last_regime = market_regimes.iloc[-1] if not market_regimes.empty else 0
                
                # Calcular características para esta fecha
                date_features = calculate_features(
                    prices.loc[:date],
                    volume.loc[:date],
                    {k: v.loc[:date] for k, v in returns.items()}
                )
                
                # Filtrar características para esta fecha
                date_features = date_features[date_features['date'] == date]
                
                # Si no hay características, continuar con la siguiente fecha
                if date_features.empty:
                    continue
                
                # Generar señales para esta fecha
                date_signals = pd.DataFrame({
                    'date': date,
                    'ticker': date_features['ticker'],
                    'signal': (
                        0.25 * date_features['momentum_1m'] +
                        0.25 * date_features['momentum_3m'] +
                        0.25 * date_features['momentum_6m'] +
                        0.25 * date_features['momentum_12m']
                    ),
                    'autocorr': date_features['autocorr'],
                    'regime': last_regime
                })
                
                # Ajustar señales por autocorrelación
                for idx, row in date_signals.iterrows():
                    autocorr = row['autocorr']
                    if not pd.isna(autocorr) and autocorr != 0:
                        if autocorr > 0:
                            adjustment_factor = 1 / (1 + 2 * autocorr)
                        else:
                            adjustment_factor = 1 - 2 * autocorr
                        date_signals.loc[idx, 'signal'] *= adjustment_factor
                
                # Construir portafolio
                weights = construct_portfolio(date_signals, sectors, date)
                
                # Guardar pesos
                all_portfolio_weights[date] = weights
            
            # Calcular retornos del portafolio para el período de prueba
            test_returns = calculate_portfolio_returns(
                prices,
                all_portfolio_weights,
                train_end,
                test_end
            )
            
            # Añadir a los retornos totales
            all_strategy_returns = pd.concat([all_strategy_returns, test_returns])
            
            # Avanzar a la siguiente ventana
            current_start = train_end
        
        # Calcular métricas de rendimiento
        metrics = calculate_performance_metrics(all_strategy_returns, benchmark_returns)
        
        # Generar gráficos
        plot_performance(all_strategy_returns, benchmark_returns, title='Walk-Forward Strategy vs Benchmark')
        plot_drawdown(all_strategy_returns, title='Walk-Forward Strategy Drawdown')
        
        # Guardar métricas en CSV
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv('./artifacts/results/data/walk_forward_metrics.csv', index=False)
        
        # Guardar retornos en CSV
        all_strategy_returns.to_csv('./artifacts/results/data/walk_forward_returns.csv')
        
        print("Validación walk-forward completada.")
        
        return all_strategy_returns, benchmark_returns, metrics
    
    except Exception as e:
        logging.error(f"Error en validación walk-forward: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        # Devolver valores vacíos en caso de error
        return pd.Series(), pd.Series(), {}

def main():
    """
    Función principal que ejecuta la estrategia.
    """
    try:
        print("Iniciando estrategia de momentum multi-horizonte...")
        
        # Obtener tickers y sectores del S&P 500
        sectors = get_sp500_tickers()
        tickers = list(sectors.keys())
        
        # Si hay demasiados tickers, limitar para evitar errores de API
        if len(tickers) > 100:
            tickers = tickers[:100]
            sectors = {ticker: sectors[ticker] for ticker in tickers}
        
        # Definir fechas
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')  # 3 años
        
        # Realizar backtest
        print("\n=== Ejecutando Backtest ===")
        strategy_returns, benchmark_returns, metrics = backtest_strategy(
            tickers,
            sectors,
            start_date,
            end_date,
            rebalance_freq='M'  # Rebalanceo mensual
        )
        
        # Mostrar métricas
        print("\nMétricas de rendimiento del backtest:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Realizar validación walk-forward
        print("\n=== Ejecutando Validación Walk-Forward ===")
        wf_returns, wf_benchmark, wf_metrics = walk_forward_validation(
            tickers,
            sectors,
            start_date,
            end_date,
            train_window=252,  # 1 año de entrenamiento
            test_window=63     # 3 meses de prueba
        )
        
        # Mostrar métricas
        print("\nMétricas de rendimiento de la validación walk-forward:")
        for metric, value in wf_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        print("\nEstrategia completada. Resultados guardados en ./artifacts/results/")
    
    except Exception as e:
        logging.error(f"Error en función principal: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
