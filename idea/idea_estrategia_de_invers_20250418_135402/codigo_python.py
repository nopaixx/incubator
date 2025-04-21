import os
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime, timedelta
import warnings
import requests
from bs4 import BeautifulSoup
import time
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

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

class AdaptiveFactorStrategy:
    def __init__(self, start_date='2010-01-01', end_date=None, lookback_period=252, 
                 rebalance_freq=21, max_stock_weight=0.05, max_sector_weight=0.20):
        """
        Inicializa la estrategia de factores adaptativos market-neutral.
        
        Args:
            start_date: Fecha de inicio para los datos
            end_date: Fecha de fin para los datos (None = hoy)
            lookback_period: Período de lookback para cálculos (días de trading)
            rebalance_freq: Frecuencia de rebalanceo (días de trading)
            max_stock_weight: Peso máximo por acción
            max_sector_weight: Peso máximo por sector
        """
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.lookback_period = lookback_period
        self.rebalance_freq = rebalance_freq
        self.max_stock_weight = max_stock_weight
        self.max_sector_weight = max_sector_weight
        
        # Parámetros para identificación de regímenes
        self.regime_lookback = 63  # ~3 meses
        self.vix_high_threshold = 25
        self.vol_high_threshold = 0.20  # Anualizado
        
        # Parámetros para circuit breakers
        self.max_factor_drawdown = 0.15
        self.drawdown_recovery_threshold = 0.05
        
        # Inicializar datos
        self.sp500_tickers = None
        self.market_data = None
        self.stock_data = None
        self.sector_data = None
        self.vix_data = None
        self.factor_data = {}
        self.factor_performance = {}
        self.factor_weights = {}
        self.regime_history = None
        self.portfolio_history = None
        
        # Factores a utilizar
        self.factors = [
            'momentum', 'value', 'quality', 'low_vol', 
            'size', 'growth', 'dividend', 'profitability'
        ]
        
        # Inicializar pesos de factores por régimen
        self.init_factor_weights()
    
    def init_factor_weights(self):
        """Inicializa los pesos de factores por régimen con valores predeterminados."""
        # Pesos iniciales por régimen (serán optimizados)
        regimes = ['low_vol', 'high_vol', 'transition']
        
        for regime in regimes:
            self.factor_weights[regime] = {factor: 1/len(self.factors) for factor in self.factors}
    
    def get_sp500_tickers(self):
        """Obtiene la lista de tickers del S&P 500 desde Wikipedia."""
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'class': 'wikitable sortable'})
            
            tickers = []
            sectors = {}
            
            for row in table.findAll('tr')[1:]:
                cells = row.findAll('td')
                ticker = cells[0].text.strip()
                sector = cells[1].text.strip()
                
                tickers.append(ticker)
                sectors[ticker] = sector
            
            self.sp500_tickers = tickers
            self.sector_data = pd.Series(sectors)
            
            # Guardar datos para referencia
            pd.DataFrame({'Ticker': tickers, 'Sector': [sectors[t] for t in tickers]}).to_csv(
                './artifacts/results/data/sp500_components.csv', index=False)
            
            return tickers
        except Exception as e:
            logging.error(f"Error obteniendo tickers del S&P 500: {str(e)}")
            raise
    
    def download_data_in_batches(self, tickers, start_date, end_date, batch_size=100):
        """Descarga datos en lotes para evitar limitaciones de yfinance."""
        all_data = {}
        
        # Añadir SPY y VIX a la primera descarga
        first_batch = tickers[:batch_size-2] + ['^GSPC', '^VIX']
        
        for i in range(0, len(tickers), batch_size):
            batch = first_batch if i == 0 else tickers[i:i+batch_size]
            
            # Intentar hasta 3 veces con backoff exponencial
            for attempt in range(3):
                try:
                    data = yf.download(batch, start=start_date, end=end_date, progress=False)
                    
                    # Si los datos están vacíos, reintenta
                    if data.empty:
                        if attempt < 2:
                            time.sleep(2 ** attempt)  # Backoff exponencial
                            continue
                        else:
                            logging.warning(f"No se pudieron obtener datos para el lote {i//batch_size+1}")
                    
                    # Procesar datos por ticker
                    if len(batch) > 1:
                        for ticker in batch:
                            try:
                                ticker_data = data.xs(ticker, level=1, axis=1)
                                if not ticker_data.empty:
                                    all_data[ticker] = ticker_data
                            except:
                                continue
                    else:
                        all_data[batch[0]] = data
                    
                    break  # Salir del bucle de intentos si fue exitoso
                
                except Exception as e:
                    if attempt < 2:
                        time.sleep(2 ** attempt)  # Backoff exponencial
                    else:
                        logging.error(f"Error descargando datos para el lote {i//batch_size+1}: {str(e)}")
            
            # Pequeña pausa entre lotes para evitar límites de API
            time.sleep(1)
        
        return all_data
    
    def load_data(self):
        """Carga todos los datos necesarios para la estrategia."""
        try:
            # Obtener tickers del S&P 500
            if self.sp500_tickers is None:
                self.get_sp500_tickers()
            
            # Descargar datos en lotes
            data_dict = self.download_data_in_batches(
                self.sp500_tickers, self.start_date, self.end_date)
            
            # Extraer datos del mercado (S&P 500) y VIX
            if '^GSPC' in data_dict:
                self.market_data = data_dict['^GSPC']
                del data_dict['^GSPC']
            else:
                logging.error("No se pudieron obtener datos del S&P 500")
                raise ValueError("No se pudieron obtener datos del S&P 500")
            
            if '^VIX' in data_dict:
                self.vix_data = data_dict['^VIX']['Close']
                del data_dict['^VIX']
            else:
                logging.error("No se pudieron obtener datos del VIX")
                raise ValueError("No se pudieron obtener datos del VIX")
            
            # Crear DataFrame con precios de cierre
            close_prices = pd.DataFrame({ticker: data['Close'] 
                                         for ticker, data in data_dict.items()
                                         if 'Close' in data})
            
            # Crear DataFrame con volúmenes
            volumes = pd.DataFrame({ticker: data['Volume'] 
                                   for ticker, data in data_dict.items()
                                   if 'Volume' in data})
            
            # Crear DataFrame con datos fundamentales (para factores)
            fundamentals = {}
            for ticker, data in data_dict.items():
                if all(col in data for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                    fundamentals[ticker] = {
                        'close': data['Close'],
                        'high': data['High'],
                        'low': data['Low'],
                        'open': data['Open'],
                        'volume': data['Volume']
                    }
            
            self.stock_data = {
                'close': close_prices,
                'volume': volumes,
                'fundamentals': fundamentals
            }
            
            # Calcular retornos diarios
            self.stock_data['returns'] = self.stock_data['close'].pct_change()
            
            # Guardar datos de mercado para referencia
            self.market_data['Close'].to_csv('./artifacts/results/data/sp500_prices.csv')
            self.vix_data.to_csv('./artifacts/results/data/vix_data.csv')
            
            print(f"Datos cargados: {len(self.stock_data['close'].columns)} acciones")
            
        except Exception as e:
            logging.error(f"Error cargando datos: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            raise
    
    def calculate_factors(self):
        """Calcula todos los factores para cada acción."""
        try:
            # Asegurarse de que los datos estén cargados
            if self.stock_data is None:
                self.load_data()
            
            close_prices = self.stock_data['close']
            returns = self.stock_data['returns']
            volumes = self.stock_data['volume']
            
            # 1. Factor Momentum (retornos de 12 meses excluyendo el último mes)
            momentum = pd.DataFrame(index=close_prices.index, columns=close_prices.columns)
            for date in close_prices.index:
                # Obtener fecha hace 12 meses y hace 1 mes
                year_ago = close_prices.index[close_prices.index < date]
                if len(year_ago) >= 252:  # ~1 año de trading
                    year_ago = year_ago[-252]
                    month_ago = close_prices.index[close_prices.index < date][-21]  # ~1 mes de trading
                    
                    # Calcular retornos desde hace 12 meses hasta hace 1 mes
                    prices_year_ago = close_prices.loc[year_ago]
                    prices_month_ago = close_prices.loc[month_ago]
                    
                    momentum.loc[date] = (prices_month_ago / prices_year_ago) - 1
            
            # 2. Factor Value (inverso del P/E, simulado con precio/volumen como proxy)
            # En una implementación real, usaríamos datos fundamentales reales
            value = pd.DataFrame(index=close_prices.index, columns=close_prices.columns)
            for date in close_prices.index:
                if date in volumes.index:
                    # Usar precio/volumen como proxy inverso de value
                    # Valores más bajos = más value
                    price_to_volume = close_prices.loc[date] / (volumes.loc[date] + 1)
                    value.loc[date] = -price_to_volume  # Invertir para que valores altos = más value
            
            # 3. Factor Quality (estabilidad de retornos, menor volatilidad = mayor calidad)
            quality = pd.DataFrame(index=close_prices.index, columns=close_prices.columns)
            for date in close_prices.index:
                past_dates = returns.index[returns.index < date]
                if len(past_dates) >= 63:  # ~3 meses de trading
                    past_dates = past_dates[-63:]
                    # Calcular volatilidad de retornos (menor = mejor calidad)
                    vol = returns.loc[past_dates].std()
                    quality.loc[date] = -vol  # Invertir para que valores altos = más calidad
            
            # 4. Factor Low Volatility
            low_vol = pd.DataFrame(index=close_prices.index, columns=close_prices.columns)
            for date in close_prices.index:
                past_dates = returns.index[returns.index < date]
                if len(past_dates) >= 126:  # ~6 meses de trading
                    past_dates = past_dates[-126:]
                    # Calcular volatilidad de retornos
                    vol = returns.loc[past_dates].std() * np.sqrt(252)  # Anualizar
                    low_vol.loc[date] = -vol  # Invertir para que valores altos = menor volatilidad
            
            # 5. Factor Size (inverso de la capitalización de mercado, proxy con volumen)
            size = pd.DataFrame(index=close_prices.index, columns=close_prices.columns)
            for date in close_prices.index:
                if date in volumes.index:
                    # Usar volumen como proxy de tamaño
                    size.loc[date] = -volumes.loc[date]  # Invertir para que valores altos = menor tamaño
            
            # 6. Factor Growth (tasa de crecimiento de precios)
            growth = pd.DataFrame(index=close_prices.index, columns=close_prices.columns)
            for date in close_prices.index:
                past_dates = close_prices.index[close_prices.index < date]
                if len(past_dates) >= 252:  # ~1 año de trading
                    # Fechas para 1 año, 6 meses y 3 meses atrás
                    year_ago = past_dates[-252]
                    six_months_ago = past_dates[-126] if len(past_dates) >= 126 else past_dates[0]
                    three_months_ago = past_dates[-63] if len(past_dates) >= 63 else past_dates[0]
                    
                    # Calcular tasas de crecimiento
                    growth_1y = (close_prices.loc[date] / close_prices.loc[year_ago]) - 1
                    growth_6m = (close_prices.loc[date] / close_prices.loc[six_months_ago]) - 1
                    growth_3m = (close_prices.loc[date] / close_prices.loc[three_months_ago]) - 1
                    
                    # Promedio ponderado de tasas de crecimiento
                    growth.loc[date] = 0.5 * growth_1y + 0.3 * growth_6m + 0.2 * growth_3m
            
            # 7. Factor Dividend (simulado con volatilidad baja y retornos estables)
            # En una implementación real, usaríamos datos de dividendos reales
            dividend = pd.DataFrame(index=close_prices.index, columns=close_prices.columns)
            for date in close_prices.index:
                past_dates = returns.index[returns.index < date]
                if len(past_dates) >= 252:  # ~1 año de trading
                    past_dates = past_dates[-252:]
                    # Combinar baja volatilidad y retornos positivos como proxy de dividendos
                    vol = returns.loc[past_dates].std()
                    avg_return = returns.loc[past_dates].mean()
                    dividend.loc[date] = avg_return - vol  # Mayor retorno y menor vol = mejor
            
            # 8. Factor Profitability (simulado con consistencia de retornos positivos)
            # En una implementación real, usaríamos datos fundamentales reales
            profitability = pd.DataFrame(index=close_prices.index, columns=close_prices.columns)
            for date in close_prices.index:
                past_dates = returns.index[returns.index < date]
                if len(past_dates) >= 126:  # ~6 meses de trading
                    past_dates = past_dates[-126:]
                    # Porcentaje de días con retornos positivos
                    positive_days = (returns.loc[past_dates] > 0).mean()
                    profitability.loc[date] = positive_days
            
            # Almacenar factores calculados
            self.factor_data = {
                'momentum': momentum,
                'value': value,
                'quality': quality,
                'low_vol': low_vol,
                'size': size,
                'growth': growth,
                'dividend': dividend,
                'profitability': profitability
            }
            
            # Normalizar factores (z-score por fecha)
            for factor_name, factor_df in self.factor_data.items():
                for date in factor_df.index:
                    if not factor_df.loc[date].isna().all():
                        factor_values = factor_df.loc[date].dropna()
                        if len(factor_values) > 0:
                            mean = factor_values.mean()
                            std = factor_values.std()
                            if std > 0:
                                factor_df.loc[date] = (factor_df.loc[date] - mean) / std
            
            # Guardar datos de factores para referencia
            for factor_name, factor_df in self.factor_data.items():
                factor_df.iloc[-252:].to_csv(f'./artifacts/results/data/factor_{factor_name}.csv')
            
            print("Factores calculados correctamente")
            
        except Exception as e:
            logging.error(f"Error calculando factores: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            raise
    
    def identify_market_regime(self, date):
        """
        Identifica el régimen de mercado actual basado en VIX, volatilidad y tendencias.
        
        Args:
            date: Fecha para la cual identificar el régimen
        
        Returns:
            str: Régimen identificado ('low_vol', 'high_vol', 'transition')
        """
        try:
            # Obtener fechas anteriores para el lookback
            past_dates = self.market_data.index[self.market_data.index < date]
            if len(past_dates) < self.regime_lookback:
                return 'transition'  # Por defecto si no hay suficientes datos
            
            past_dates = past_dates[-self.regime_lookback:]
            
            # 1. Nivel del VIX
            current_vix = self.vix_data.loc[date] if date in self.vix_data.index else None
            vix_high = current_vix is not None and current_vix > self.vix_high_threshold
            
            # 2. Volatilidad del mercado
            market_returns = self.market_data['Close'].pct_change().loc[past_dates]
            market_vol = market_returns.std() * np.sqrt(252)  # Anualizada
            vol_high = market_vol > self.vol_high_threshold
            
            # 3. Tendencia del mercado
            market_trend = self.market_data['Close'].loc[date] / self.market_data['Close'].loc[past_dates[0]] - 1
            trend_up = market_trend > 0.05  # 5% de subida en el período
            trend_down = market_trend < -0.05  # 5% de bajada en el período
            
            # Determinar régimen
            if vix_high or vol_high:
                if trend_down:
                    return 'high_vol'  # Alta volatilidad con tendencia bajista
                else:
                    return 'transition'  # Alta volatilidad sin tendencia clara
            elif trend_up and not vol_high:
                return 'low_vol'  # Baja volatilidad con tendencia alcista
            else:
                return 'transition'  # Caso por defecto
                
        except Exception as e:
            logging.error(f"Error identificando régimen de mercado: {str(e)}")
            return 'transition'  # Valor por defecto en caso de error
    
    def calculate_regime_history(self):
        """Calcula el historial de regímenes de mercado para todo el período."""
        try:
            regimes = {}
            
            for date in self.market_data.index:
                regimes[date] = self.identify_market_regime(date)
            
            self.regime_history = pd.Series(regimes)
            
            # Guardar historial de regímenes
            self.regime_history.to_csv('./artifacts/results/data/regime_history.csv')
            
            # Visualizar distribución de regímenes
            regime_counts = self.regime_history.value_counts()
            plt.figure(figsize=(10, 6))
            sns.barplot(x=regime_counts.index, y=regime_counts.values)
            plt.title('Distribución de Regímenes de Mercado')
            plt.ylabel('Número de días')
            plt.tight_layout()
            plt.savefig('./artifacts/results/figures/regime_distribution.png')
            plt.close()
            
            print(f"Historial de regímenes calculado: {regime_counts.to_dict()}")
            
        except Exception as e:
            logging.error(f"Error calculando historial de regímenes: {str(e)}")
            raise
    
    def calculate_factor_performance(self):
        """Calcula el rendimiento histórico de cada factor."""
        try:
            # Inicializar diccionario para almacenar rendimientos
            factor_returns = {factor: pd.Series(index=self.stock_data['returns'].index) 
                             for factor in self.factors}
            
            # Para cada fecha, calcular el rendimiento de cada factor
            for date in self.stock_data['returns'].index[1:]:  # Empezar desde el segundo día
                prev_date = self.stock_data['returns'].index[self.stock_data['returns'].index < date][-1]
                
                for factor in self.factors:
                    if factor in self.factor_data and prev_date in self.factor_data[factor].index:
                        # Obtener scores del factor para la fecha anterior
                        factor_scores = self.factor_data[factor].loc[prev_date].dropna()
                        
                        if len(factor_scores) > 0:
                            # Seleccionar top y bottom 10% de acciones por factor
                            num_stocks = max(10, int(len(factor_scores) * 0.1))
                            top_stocks = factor_scores.nlargest(num_stocks).index
                            bottom_stocks = factor_scores.nsmallest(num_stocks).index
                            
                            # Calcular retornos para estas acciones
                            if date in self.stock_data['returns'].index:
                                top_returns = self.stock_data['returns'].loc[date, top_stocks].mean()
                                bottom_returns = self.stock_data['returns'].loc[date, bottom_stocks].mean()
                                
                                # Factor return = long top stocks, short bottom stocks
                                factor_returns[factor].loc[date] = top_returns - bottom_returns
            
            # Calcular rendimiento acumulado para cada factor
            factor_cumulative = {factor: (1 + factor_returns[factor].fillna(0)).cumprod() 
                                for factor in self.factors}
            
            # Calcular drawdowns para cada factor
            factor_drawdowns = {}
            for factor in self.factors:
                cumulative = factor_cumulative[factor]
                running_max = cumulative.cummax()
                drawdown = (cumulative / running_max) - 1
                factor_drawdowns[factor] = drawdown
            
            # Calcular Sharpe ratio para cada factor
            factor_sharpe = {}
            for factor in self.factors:
                returns_series = factor_returns[factor].dropna()
                if len(returns_series) > 0:
                    annual_return = returns_series.mean() * 252
                    annual_vol = returns_series.std() * np.sqrt(252)
                    if annual_vol > 0:
                        factor_sharpe[factor] = annual_return / annual_vol
                    else:
                        factor_sharpe[factor] = 0
                else:
                    factor_sharpe[factor] = 0
            
            # Almacenar resultados
            self.factor_performance = {
                'returns': factor_returns,
                'cumulative': factor_cumulative,
                'drawdowns': factor_drawdowns,
                'sharpe': factor_sharpe
            }
            
            # Guardar rendimiento de factores
            factor_perf_df = pd.DataFrame({f: factor_cumulative[f] for f in self.factors})
            factor_perf_df.to_csv('./artifacts/results/data/factor_performance.csv')
            
            # Visualizar rendimiento de factores
            plt.figure(figsize=(12, 8))
            for factor in self.factors:
                plt.plot(factor_cumulative[factor].index, factor_cumulative[factor].values, label=factor)
            plt.title('Rendimiento Acumulado de Factores')
            plt.xlabel('Fecha')
            plt.ylabel('Rendimiento Acumulado')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('./artifacts/results/figures/factor_performance.png')
            plt.close()
            
            print("Rendimiento de factores calculado correctamente")
            
        except Exception as e:
            logging.error(f"Error calculando rendimiento de factores: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            raise
    
    def optimize_factor_weights(self, train_start, train_end):
        """
        Optimiza los pesos de los factores para cada régimen basado en datos históricos.
        
        Args:
            train_start: Fecha de inicio para entrenamiento
            train_end: Fecha de fin para entrenamiento
        """
        try:
            # Filtrar datos de entrenamiento
            train_dates = self.market_data.index[(self.market_data.index >= train_start) & 
                                                (self.market_data.index <= train_end)]
            
            if len(train_dates) < 63:  # Mínimo ~3 meses de datos
                # Usar pesos predeterminados si no hay suficientes datos
                return
            
            # Optimizar pesos para cada régimen
            for regime in ['low_vol', 'high_vol', 'transition']:
                # Filtrar fechas por régimen
                regime_dates = [date for date in train_dates 
                               if date in self.regime_history.index and self.regime_history[date] == regime]
                
                if len(regime_dates) < 21:  # Mínimo ~1 mes de datos para el régimen
                    continue
                
                # Preparar datos para optimización
                factor_returns_regime = {}
                for factor in self.factors:
                    if factor in self.factor_performance['returns']:
                        returns = self.factor_performance['returns'][factor].loc[regime_dates].fillna(0)
                        if len(returns) > 0:
                            factor_returns_regime[factor] = returns
                
                if len(factor_returns_regime) < 2:
                    continue
                
                # Convertir a DataFrame para cálculos
                returns_df = pd.DataFrame(factor_returns_regime)
                
                # Calcular matriz de covarianza y vector de retornos esperados
                cov_matrix = returns_df.cov() * 252  # Anualizada
                exp_returns = returns_df.mean() * 252  # Anualizados
                
                # Aplicar penalización a factores con drawdowns significativos
                for factor in self.factors:
                    if factor in self.factor_performance['drawdowns']:
                        max_drawdown = self.factor_performance['drawdowns'][factor].loc[regime_dates].min()
                        if max_drawdown < -self.max_factor_drawdown:
                            exp_returns[factor] *= (1 + max_drawdown)  # Reducir retorno esperado
                
                # Optimización simple: maximizar Sharpe ratio
                # En una implementación real, usaríamos optimización cuadrática con restricciones
                
                # Generar combinaciones de pesos
                num_factors = len(exp_returns)
                best_sharpe = -np.inf
                best_weights = None
                
                # Usar validación cruzada para evitar overfitting
                tscv = TimeSeriesSplit(n_splits=5)
                for train_idx, test_idx in tscv.split(returns_df):
                    train_returns = returns_df.iloc[train_idx]
                    test_returns = returns_df.iloc[test_idx]
                    
                    # Calcular retornos esperados y covarianza en datos de entrenamiento
                    train_exp_returns = train_returns.mean() * 252
                    train_cov_matrix = train_returns.cov() * 252
                    
                    # Generar 1000 combinaciones aleatorias de pesos
                    for _ in range(1000):
                        weights = np.random.random(num_factors)
                        weights /= weights.sum()  # Normalizar para que sumen 1
                        
                        # Calcular Sharpe ratio en datos de prueba
                        portfolio_return = (test_returns @ weights).mean() * 252
                        portfolio_vol = np.sqrt(weights @ train_cov_matrix @ weights)
                        
                        if portfolio_vol > 0:
                            sharpe = portfolio_return / portfolio_vol
                            if sharpe > best_sharpe:
                                best_sharpe = sharpe
                                best_weights = weights
                
                if best_weights is not None:
                    # Actualizar pesos de factores para el régimen
                    self.factor_weights[regime] = {factor: weight 
                                                 for factor, weight in zip(exp_returns.index, best_weights)}
            
            # Guardar pesos optimizados
            weights_df = pd.DataFrame(self.factor_weights)
            weights_df.to_csv('./artifacts/results/data/factor_weights.csv')
            
            print("Pesos de factores optimizados correctamente")
            
        except Exception as e:
            logging.error(f"Error optimizando pesos de factores: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            raise
    
    def calculate_combined_factor_score(self, date):
        """
        Calcula el score combinado de factores para cada acción en una fecha específica.
        
        Args:
            date: Fecha para calcular los scores
        
        Returns:
            pd.Series: Score combinado para cada acción
        """
        try:
            # Identificar régimen actual
            regime = self.identify_market_regime(date)
            
            # Obtener pesos de factores para el régimen
            factor_weights = self.factor_weights.get(regime, {})
            
            # Inicializar scores combinados
            all_stocks = set()
            for factor in self.factors:
                if factor in self.factor_data and date in self.factor_data[factor].index:
                    stocks = self.factor_data[factor].loc[date].dropna().index
                    all_stocks.update(stocks)
            
            combined_scores = pd.Series(0, index=list(all_stocks))
            
            # Aplicar circuit breakers: verificar drawdowns de factores
            active_factors = []
            for factor in self.factors:
                if factor in self.factor_performance['drawdowns'] and date in self.factor_performance['drawdowns'][factor].index:
                    current_drawdown = self.factor_performance['drawdowns'][factor].loc[date]
                    if current_drawdown > -self.max_factor_drawdown:
                        active_factors.append(factor)
            
            if not active_factors:
                active_factors = self.factors  # Si todos están en drawdown, usar todos
            
            # Calcular score combinado
            for factor in active_factors:
                if factor in factor_weights and factor in self.factor_data and date in self.factor_data[factor].index:
                    weight = factor_weights[factor]
                    factor_scores = self.factor_data[factor].loc[date].dropna()
                    
                    # Aplicar peso del factor a los scores
                    for stock in factor_scores.index:
                        if stock in combined_scores.index:
                            combined_scores[stock] += weight * factor_scores[stock]
            
            return combined_scores
            
        except Exception as e:
            logging.error(f"Error calculando scores combinados: {str(e)}")
            return pd.Series()
    
    def calculate_stock_betas(self, date, lookback=126):
        """
        Calcula las betas de las acciones respecto al mercado.
        
        Args:
            date: Fecha para calcular las betas
            lookback: Período de lookback para el cálculo
            
        Returns:
            pd.Series: Beta para cada acción
        """
        try:
            # Obtener fechas anteriores para el lookback
            past_dates = self.stock_data['returns'].index[self.stock_data['returns'].index < date]
            if len(past_dates) < lookback:
                return pd.Series()
            
            past_dates = past_dates[-lookback:]
            
            # Obtener retornos del mercado
            market_returns = self.market_data['Close'].pct_change().loc[past_dates].fillna(0)
            
            # Calcular beta para cada acción
            betas = {}
            for stock in self.stock_data['returns'].columns:
                stock_returns = self.stock_data['returns'].loc[past_dates, stock].fillna(0)
                
                if len(stock_returns) == len(market_returns) and not stock_returns.isna().all():
                    # Usar regresión lineal para calcular beta
                    model = LinearRegression()
                    X = market_returns.values.reshape(-1, 1)
                    y = stock_returns.values
                    model.fit(X, y)
                    beta = model.coef_[0]
                    betas[stock] = beta
            
            return pd.Series(betas)
            
        except Exception as e:
            logging.error(f"Error calculando betas: {str(e)}")
            return pd.Series()
    
    def ensure_sector_neutrality(self, long_stocks, short_stocks, date):
        """
        Asegura la neutralidad sectorial entre posiciones long y short.
        
        Args:
            long_stocks: Lista de acciones en posición long
            short_stocks: Lista de acciones en posición short
            date: Fecha actual
            
        Returns:
            tuple: Listas ajustadas de acciones long y short
        """
        try:
            # Obtener sectores para las acciones
            sectors = {}
            for stock in long_stocks + short_stocks:
                if stock in self.sector_data.index:
                    sectors[stock] = self.sector_data[stock]
            
            # Calcular exposición sectorial
            long_sector_exposure = {}
            short_sector_exposure = {}
            
            for stock in long_stocks:
                if stock in sectors:
                    sector = sectors[stock]
                    long_sector_exposure[sector] = long_sector_exposure.get(sector, 0) + 1
            
            for stock in short_stocks:
                if stock in sectors:
                    sector = sectors[stock]
                    short_sector_exposure[sector] = short_sector_exposure.get(sector, 0) + 1
            
            # Identificar sectores desbalanceados
            all_sectors = set(long_sector_exposure.keys()) | set(short_sector_exposure.keys())
            
            for sector in all_sectors:
                long_count = long_sector_exposure.get(sector, 0)
                short_count = short_sector_exposure.get(sector, 0)
                
                # Si hay desbalance significativo
                if abs(long_count - short_count) > 2:
                    if long_count > short_count:
                        # Reducir posiciones long en este sector
                        sector_long_stocks = [s for s in long_stocks if s in sectors and sectors[s] == sector]
                        excess = min(len(sector_long_stocks), long_count - short_count - 2)
                        if excess > 0:
                            for _ in range(excess):
                                if sector_long_stocks:
                                    long_stocks.remove(sector_long_stocks.pop())
                    else:
                        # Reducir posiciones short en este sector
                        sector_short_stocks = [s for s in short_stocks if s in sectors and sectors[s] == sector]
                        excess = min(len(sector_short_stocks), short_count - long_count - 2)
                        if excess > 0:
                            for _ in range(excess):
                                if sector_short_stocks:
                                    short_stocks.remove(sector_short_stocks.pop())
            
            return long_stocks, short_stocks
            
        except Exception as e:
            logging.error(f"Error asegurando neutralidad sectorial: {str(e)}")
            return long_stocks, short_stocks
    
    def select_portfolio(self, date, num_stocks=50):
        """
        Selecciona el portfolio para una fecha específica.
        
        Args:
            date: Fecha para seleccionar el portfolio
            num_stocks: Número de acciones a seleccionar (long + short)
            
        Returns:
            dict: Portfolio con pesos para cada acción
        """
        try:
            # Calcular scores combinados
            combined_scores = self.calculate_combined_factor_score(date)
            
            if combined_scores.empty:
                return {}
            
            # Calcular betas
            stock_betas = self.calculate_stock_betas(date)
            
            # Filtrar acciones con datos completos
            valid_stocks = combined_scores.index.intersection(stock_betas.index)
            combined_scores = combined_scores.loc[valid_stocks]
            stock_betas = stock_betas.loc[valid_stocks]
            
            if len(combined_scores) < num_stocks:
                return {}
            
            # Seleccionar acciones long y short basadas en scores
            num_each_side = num_stocks // 2
            long_candidates = combined_scores.nlargest(num_each_side * 2).index.tolist()
            short_candidates = combined_scores.nsmallest(num_each_side * 2).index.tolist()
            
            # Asegurar neutralidad sectorial
            long_stocks, short_stocks = self.ensure_sector_neutrality(
                long_candidates[:num_each_side], 
                short_candidates[:num_each_side],
                date
            )
            
            # Calcular volatilidades para ponderación inversa
            volatilities = {}
            past_dates = self.stock_data['returns'].index[self.stock_data['returns'].index < date]
            if len(past_dates) >= 63:  # ~3 meses
                past_dates = past_dates[-63:]
                for stock in long_stocks + short_stocks:
                    if stock in self.stock_data['returns'].columns:
                        vol = self.stock_data['returns'].loc[past_dates, stock].std()
                        if vol > 0:
                            volatilities[stock] = vol
            
            # Si no hay volatilidades, usar pesos iguales
            if not volatilities:
                long_weights = {stock: 1/len(long_stocks) for stock in long_stocks}
                short_weights = {stock: -1/len(short_stocks) for stock in short_stocks}
            else:
                # Ponderación inversa a la volatilidad
                long_inv_vol = {stock: 1/volatilities.get(stock, 1) for stock in long_stocks}
                short_inv_vol = {stock: 1/volatilities.get(stock, 1) for stock in short_stocks}
                
                # Normalizar pesos
                long_sum = sum(long_inv_vol.values())
                short_sum = sum(short_inv_vol.values())
                
                if long_sum > 0 and short_sum > 0:
                    long_weights = {stock: weight/long_sum for stock, weight in long_inv_vol.items()}
                    short_weights = {stock: -weight/short_sum for stock, weight in short_inv_vol.items()}
                else:
                    long_weights = {stock: 1/len(long_stocks) for stock in long_stocks}
                    short_weights = {stock: -1/len(short_stocks) for stock in short_stocks}
            
            # Combinar pesos
            portfolio_weights = {**long_weights, **short_weights}
            
            # Ajustar para neutralidad beta
            portfolio_beta = sum(portfolio_weights.get(stock, 0) * stock_betas.get(stock, 0) 
                                for stock in portfolio_weights)
            
            if portfolio_beta != 0:
                # Ajustar pesos para neutralizar beta
                beta_adjustment = -portfolio_beta
                
                # Aplicar ajuste a todas las posiciones
                for stock in portfolio_weights:
                    if stock in stock_betas:
                        portfolio_weights[stock] += beta_adjustment * stock_betas[stock] / len(portfolio_weights)
            
            # Aplicar límites de concentración
            for stock in list(portfolio_weights.keys()):
                if abs(portfolio_weights[stock]) > self.max_stock_weight:
                    if portfolio_weights[stock] > 0:
                        portfolio_weights[stock] = self.max_stock_weight
                    else:
                        portfolio_weights[stock] = -self.max_stock_weight
            
            # Normalizar para que la suma de valores absolutos sea 2 (1 long, 1 short)
            abs_sum = sum(abs(w) for w in portfolio_weights.values())
            if abs_sum > 0:
                portfolio_weights = {stock: 2 * weight / abs_sum for stock, weight in portfolio_weights.items()}
            
            return portfolio_weights
            
        except Exception as e:
            logging.error(f"Error seleccionando portfolio: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return {}
    
    def calculate_portfolio_returns(self, portfolio_weights, date):
        """
        Calcula el retorno del portfolio para una fecha específica.
        
        Args:
            portfolio_weights: Diccionario con pesos del portfolio
            date: Fecha para calcular el retorno
            
        Returns:
            float: Retorno del portfolio
        """
        try:
            if not portfolio_weights or date not in self.stock_data['returns'].index:
                return 0.0
            
            # Obtener retornos para la fecha
            date_returns = self.stock_data['returns'].loc[date]
            
            # Calcular retorno ponderado
            portfolio_return = 0.0
            for stock, weight in portfolio_weights.items():
                if stock in date_returns.index and not pd.isna(date_returns[stock]):
                    portfolio_return += weight * date_returns[stock]
            
            return portfolio_return
            
        except Exception as e:
            logging.error(f"Error calculando retorno del portfolio: {str(e)}")
            return 0.0
    
    def run_backtest(self, start_date=None, end_date=None):
        """
        Ejecuta un backtest de la estrategia.
        
        Args:
            start_date: Fecha de inicio del backtest (None = usar start_date de la clase)
            end_date: Fecha de fin del backtest (None = usar end_date de la clase)
            
        Returns:
            pd.Series: Serie con valores del portfolio
        """
        try:
            # Usar fechas predeterminadas si no se especifican
            if start_date is None:
                start_date = self.start_date
            if end_date is None:
                end_date = self.end_date
            
            # Asegurarse de que los datos estén cargados y procesados
            if self.stock_data is None:
                self.load_data()
            
            if not self.factor_data:
                self.calculate_factors()
            
            if self.regime_history is None:
                self.calculate_regime_history()
            
            if not self.factor_performance:
                self.calculate_factor_performance()
            
            # Optimizar pesos de factores
            self.optimize_factor_weights(start_date, end_date)
            
            # Filtrar fechas para el backtest
            backtest_dates = self.stock_data['returns'].index[
                (self.stock_data['returns'].index >= start_date) & 
                (self.stock_data['returns'].index <= end_date)
            ]
            
            # Inicializar variables para el backtest
            portfolio_values = pd.Series(100.0, index=[backtest_dates[0]])
            current_portfolio = {}
            last_rebalance_date = backtest_dates[0]
            
            # Ejecutar backtest
            for i, date in enumerate(backtest_dates[1:], 1):
                # Verificar si es necesario rebalancear
                days_since_rebalance = len(self.stock_data['returns'].index[
                    (self.stock_data['returns'].index > last_rebalance_date) & 
                    (self.stock_data['returns'].index <= date)
                ])
                
                # Rebalancear cada rebalance_freq días o si es el primer día
                if days_since_rebalance >= self.rebalance_freq or not current_portfolio:
                    # Seleccionar nuevo portfolio
                    current_portfolio = self.select_portfolio(date)
                    last_rebalance_date = date
                
                # Calcular retorno del portfolio
                daily_return = self.calculate_portfolio_returns(current_portfolio, date)
                
                # Actualizar valor del portfolio
                portfolio_values[date] = portfolio_values.iloc[-1] * (1 + daily_return)
            
            # Guardar resultados del backtest
            portfolio_values.to_csv('./artifacts/results/data/backtest_results.csv')
            
            # Calcular métricas de rendimiento
            returns = portfolio_values.pct_change().dropna()
            
            annual_return = returns.mean() * 252
            annual_vol = returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
            
            # Calcular drawdown
            drawdown = (portfolio_values / portfolio_values.cummax()) - 1
            max_drawdown = drawdown.min()
            
            # Calcular retorno acumulado
            cumulative_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
            
            # Guardar métricas
            metrics = {
                'Annual Return': annual_return,
                'Annual Volatility': annual_vol,
                'Sharpe Ratio': sharpe_ratio,
                'Max Drawdown': max_drawdown,
                'Cumulative Return': cumulative_return
            }
            
            pd.Series(metrics).to_csv('./artifacts/results/data/backtest_metrics.csv')
            
            # Visualizar resultados
            plt.figure(figsize=(12, 8))
            plt.plot(portfolio_values.index, portfolio_values.values)
            plt.title('Backtest: Valor del Portfolio')
            plt.xlabel('Fecha')
            plt.ylabel('Valor')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('./artifacts/results/figures/backtest_performance.png')
            plt.close()
            
            # Visualizar drawdown
            plt.figure(figsize=(12, 6))
            plt.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
            plt.title('Backtest: Drawdown')
            plt.xlabel('Fecha')
            plt.ylabel('Drawdown')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('./artifacts/results/figures/backtest_drawdown.png')
            plt.close()
            
            print(f"Backtest completado. Sharpe Ratio: {sharpe_ratio:.2f}, Max Drawdown: {max_drawdown:.2%}")
            
            return portfolio_values
            
        except Exception as e:
            logging.error(f"Error ejecutando backtest: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            raise
    
    def run_walkforward_analysis(self, train_window=252*2, test_window=63, num_folds=5):
        """
        Ejecuta un análisis walkforward para evaluar la robustez de la estrategia.
        
        Args:
            train_window: Tamaño de la ventana de entrenamiento (días)
            test_window: Tamaño de la ventana de prueba (días)
            num_folds: Número de folds para el análisis
            
        Returns:
            pd.Series: Serie con valores del portfolio combinados
        """
        try:
            # Asegurarse de que los datos estén cargados y procesados
            if self.stock_data is None:
                self.load_data()
            
            if not self.factor_data:
                self.calculate_factors()
            
            if self.regime_history is None:
                self.calculate_regime_history()
            
            if not self.factor_performance:
                self.calculate_factor_performance()
            
            # Obtener todas las fechas disponibles
            all_dates = self.stock_data['returns'].index
            
            if len(all_dates) < train_window + test_window:
                raise ValueError("No hay suficientes datos para el análisis walkforward")
            
            # Calcular fechas de inicio para cada fold
            total_window = train_window + test_window
            available_range = len(all_dates) - total_window
            
            if available_range <= 0 or num_folds <= 0:
                raise ValueError("No hay suficiente rango de fechas para los folds especificados")
            
            step_size = max(1, available_range // num_folds)
            fold_start_indices = [i * step_size for i in range(num_folds)]
            
            # Inicializar resultados
            walkforward_values = pd.Series(dtype=float)
            fold_metrics = []
            
            # Ejecutar cada fold
            for fold, start_idx in enumerate(fold_start_indices):
                # Definir ventanas de entrenamiento y prueba
                train_start_idx = start_idx
                train_end_idx = train_start_idx + train_window
                test_start_idx = train_end_idx
                test_end_idx = test_start_idx + test_window
                
                # Asegurarse de que los índices estén dentro del rango
                if test_end_idx >= len(all_dates):
                    test_end_idx = len(all_dates) - 1
                
                # Convertir índices a fechas
                train_start = all_dates[train_start_idx]
                train_end = all_dates[train_end_idx]
                test_start = all_dates[test_start_idx]
                test_end = all_dates[test_end_idx]
                
                print(f"Fold {fold+1}/{num_folds}: Train {train_start} to {train_end}, Test {test_start} to {test_end}")
                
                # Optimizar pesos de factores con datos de entrenamiento
                self.optimize_factor_weights(train_start, train_end)
                
                # Ejecutar backtest en período de prueba
                fold_portfolio = self.run_backtest(test_start, test_end)
                
                # Normalizar valores para que comiencen en 100
                if not fold_portfolio.empty:
                    first_value = fold_portfolio.iloc[0]
                    normalized_portfolio = fold_portfolio / first_value * 100
                    
                    # Agregar a resultados combinados
                    walkforward_values = pd.concat([walkforward_values, normalized_portfolio])
                
                # Calcular métricas para este fold
                if not fold_portfolio.empty:
                    returns = fold_portfolio.pct_change().dropna()
                    
                    annual_return = returns.mean() * 252
                    annual_vol = returns.std() * np.sqrt(252)
                    sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
                    
                    drawdown = (fold_portfolio / fold_portfolio.cummax()) - 1
                    max_drawdown = drawdown.min()
                    
                    fold_metrics.append({
                        'Fold': fold + 1,
                        'Train Start': train_start,
                        'Train End': train_end,
                        'Test Start': test_start,
                        'Test End': test_end,
                        'Annual Return': annual_return,
                        'Annual Volatility': annual_vol,
                        'Sharpe Ratio': sharpe_ratio,
                        'Max Drawdown': max_drawdown
                    })
            
            # Eliminar duplicados y ordenar por fecha
            walkforward_values = walkforward_values[~walkforward_values.index.duplicated(keep='first')]
            walkforward_values = walkforward_values.sort_index()
            
            # Guardar resultados
            walkforward_values.to_csv('./artifacts/results/data/walkforward_results.csv')
            pd.DataFrame(fold_metrics).to_csv('./artifacts/results/data/walkforward_metrics.csv', index=False)
            
            # Visualizar resultados
            plt.figure(figsize=(12, 8))
            plt.plot(walkforward_values.index, walkforward_values.values)
            plt.title('Análisis Walkforward: Valor del Portfolio')
            plt.xlabel('Fecha')
            plt.ylabel('Valor')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('./artifacts/results/figures/walkforward_performance.png')
            plt.close()
            
            # Visualizar métricas por fold
            metrics_df = pd.DataFrame(fold_metrics)
            
            plt.figure(figsize=(12, 6))
            plt.bar(metrics_df['Fold'], metrics_df['Sharpe Ratio'])
            plt.title('Sharpe Ratio por Fold')
            plt.xlabel('Fold')
            plt.ylabel('Sharpe Ratio')
            plt.grid(True, axis='y')
            plt.tight_layout()
            plt.savefig('./artifacts/results/figures/walkforward_sharpe.png')
            plt.close()
            
            # Calcular métricas agregadas
            if walkforward_values.size > 1:
                returns = walkforward_values.pct_change().dropna()
                
                annual_return = returns.mean() * 252
                annual_vol = returns.std() * np.sqrt(252)
                sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
                
                drawdown = (walkforward_values / walkforward_values.cummax()) - 1
                max_drawdown = drawdown.min()
                
                print(f"Análisis Walkforward completado. Sharpe Ratio: {sharpe_ratio:.2f}, Max Drawdown: {max_drawdown:.2%}")
            
            return walkforward_values
            
        except Exception as e:
            logging.error(f"Error ejecutando análisis walkforward: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            raise
    
    def run_sensitivity_analysis(self, parameter_ranges):
        """
        Ejecuta un análisis de sensibilidad para identificar parámetros críticos.
        
        Args:
            parameter_ranges: Diccionario con rangos de parámetros a probar
            
        Returns:
            pd.DataFrame: Resultados del análisis de sensibilidad
        """
        try:
            # Asegurarse de que los datos estén cargados y procesados
            if self.stock_data is None:
                self.load_data()
            
            if not self.factor_data:
                self.calculate_factors()
            
            # Definir período para análisis de sensibilidad (último año)
            end_date = self.stock_data['returns'].index[-1]
            start_idx = max(0, len(self.stock_data['returns'].index) - 252)
            start_date = self.stock_data['returns'].index[start_idx]
            
            # Guardar parámetros originales
            original_params = {
                'lookback_period': self.lookback_period,
                'rebalance_freq': self.rebalance_freq,
                'max_stock_weight': self.max_stock_weight,
                'max_sector_weight': self.max_sector_weight,
                'vix_high_threshold': self.vix_high_threshold,
                'vol_high_threshold': self.vol_high_threshold,
                'max_factor_drawdown': self.max_factor_drawdown
            }
            
            # Inicializar resultados
            sensitivity_results = []
            
            # Ejecutar análisis para cada parámetro
            for param_name, param_values in parameter_ranges.items():
                for param_value in param_values:
                    # Restaurar parámetros originales
                    for name, value in original_params.items():
                        setattr(self, name, value)
                    
                    # Modificar el parámetro a analizar
                    setattr(self, param_name, param_value)
                    
                    # Ejecutar backtest con el parámetro modificado
                    portfolio_values = self.run_backtest(start_date, end_date)
                    
                    # Calcular métricas
                    if not portfolio_values.empty and len(portfolio_values) > 1:
                        returns = portfolio_values.pct_change().dropna()
                        
                        annual_return = returns.mean() * 252
                        annual_vol = returns.std() * np.sqrt(252)
                        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
                        
                        drawdown = (portfolio_values / portfolio_values.cummax()) - 1
                        max_drawdown = drawdown.min()
                        
                        sensitivity_results.append({
                            'Parameter': param_name,
                            'Value': param_value,
                            'Annual Return': annual_return,
                            'Annual Volatility': annual_vol,
                            'Sharpe Ratio': sharpe_ratio,
                            'Max Drawdown': max_drawdown
                        })
            
            # Restaurar parámetros originales
            for name, value in original_params.items():
                setattr(self, name, value)
            
            # Convertir resultados a DataFrame
            sensitivity_df = pd.DataFrame(sensitivity_results)
            
            # Guardar resultados
            sensitivity_df.to_csv('./artifacts/results/data/sensitivity_analysis.csv', index=False)
            
            # Visualizar resultados
            for param in parameter_ranges.keys():
                param_data = sensitivity_df[sensitivity_df['Parameter'] == param]
                
                if not param_data.empty:
                    plt.figure(figsize=(10, 6))
                    plt.plot(param_data['Value'], param_data['Sharpe Ratio'], marker='o')
                    plt.title(f'Sensibilidad de Sharpe Ratio a {param}')
                    plt.xlabel(param)
                    plt.ylabel('Sharpe Ratio')
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(f'./artifacts/results/figures/sensitivity_{param}.png')
                    plt.close()
            
            print("Análisis de sensibilidad completado")
            
            return sensitivity_df
            
        except Exception as e:
            logging.error(f"Error ejecutando análisis de sensibilidad: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            raise

# Función principal para ejecutar la estrategia
def main():
    try:
        print("Iniciando estrategia de factores adaptativos market-neutral...")
        
        # Crear instancia de la estrategia
        strategy = AdaptiveFactorStrategy(
            start_date='2015-01-01',
            end_date='2023-12-31',
            lookback_period=252,
            rebalance_freq=21
        )
        
        # Cargar datos
        print("Cargando datos...")
        strategy.load_data()
        
        # Calcular factores
        print("Calculando factores...")
        strategy.calculate_factors()
        
        # Calcular historial de regímenes
        print("Identificando regímenes de mercado...")
        strategy.calculate_regime_history()
        
        # Calcular rendimiento de factores
        print("Analizando rendimiento de factores...")
        strategy.calculate_factor_performance()
        
        # Ejecutar backtest
        print("Ejecutando backtest...")
        strategy.run_backtest()
        
        # Ejecutar análisis walkforward
        print("Ejecutando análisis walkforward...")
        strategy.run_walkforward_analysis(train_window=252*2, test_window=63, num_folds=5)
        
        # Ejecutar análisis de sensibilidad
        print("Ejecutando análisis de sensibilidad...")
        parameter_ranges = {
            'lookback_period': [126, 252, 378],
            'rebalance_freq': [10, 21, 42],
            'max_stock_weight': [0.03, 0.05, 0.07],
            'vix_high_threshold': [20, 25, 30],
            'max_factor_drawdown': [0.10, 0.15, 0.20]
        }
        strategy.run_sensitivity_analysis(parameter_ranges)
        
        print("Estrategia completada con éxito. Resultados guardados en ./artifacts/results/")
        
    except Exception as e:
        logging.error(f"Error en la ejecución principal: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        print(f"Error: {str(e)}. Ver detalles en ./artifacts/errors.txt")

if __name__ == "__main__":
    main()
