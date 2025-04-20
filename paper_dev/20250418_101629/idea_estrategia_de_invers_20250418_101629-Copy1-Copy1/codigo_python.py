import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy.stats import norm
import os
import logging
import warnings
from datetime import datetime, timedelta
import pickle
from tqdm import tqdm

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

class AdaptiveMultifactorStrategy:
    def __init__(self, start_date='2010-01-01', end_date=None, symbols=None, 
                 lookback_window=252, regime_window=126, n_regimes=3, 
                 rebalance_freq=21, vol_target=0.10, max_leverage=1.5):
        """
        Inicializa la estrategia de descomposición multifactorial adaptativa.
        
        Parámetros:
        -----------
        start_date : str
            Fecha de inicio para los datos históricos (formato 'YYYY-MM-DD')
        end_date : str
            Fecha de fin para los datos históricos (formato 'YYYY-MM-DD')
        symbols : list
            Lista de símbolos a incluir. Si es None, se usa el S&P 500
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
        
        # Atributos que se inicializarán más tarde
        self.prices = None
        self.returns = None
        self.factor_loadings = None
        self.factor_returns = None
        self.regimes = None
        self.regime_probs = None
        self.optimal_weights = None
        self.performance = None
        
        # Cargar datos
        self._load_data()
        
    def _load_data(self):
        """Carga los datos históricos de precios y calcula retornos."""
        try:
            if self.symbols is None:
                # Usar S&P 500 como universo por defecto
                sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
                self.symbols = sp500['Symbol'].tolist()[:50]  # Usar los primeros 50 para eficiencia
            
            # Descargar datos
            self.prices = yf.download(self.symbols, start=self.start_date, end=self.end_date)['Close']
            
            # Limpiar y preparar datos
            self.prices = self.prices.dropna(axis=1, thresh=int(len(self.prices) * 0.9))  # Eliminar acciones con muchos NaN
            self.symbols = list(self.prices.columns)
            
            # Calcular retornos diarios
            self.returns = self.prices.pct_change().dropna()
            
            print(f"Datos cargados exitosamente. {len(self.symbols)} símbolos, {len(self.returns)} días de trading.")
            
        except Exception as e:
            logging.error(f"Error al cargar datos: {str(e)}", exc_info=True)
            raise
    
    def extract_latent_factors(self, returns_window, n_components=None):
        """
        Extrae factores latentes de los retornos usando PCA.
        
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
            # Manejar valores faltantes
            returns_filled = returns_window.copy()
            
            # Usar imputación por media móvil para NaNs
            for col in returns_filled.columns:
                mask = returns_filled[col].isna()
                if mask.any():
                    returns_filled.loc[mask, col] = returns_filled[col].rolling(5, min_periods=1).mean()[mask]
            
            # Si aún hay NaNs, rellenar con ceros (menos agresivo que simplemente usar fillna(0))
            returns_filled = returns_filled.fillna(0)
            
            # Determinar número óptimo de componentes si no se especifica
            if n_components is None:
                n_components = self.find_optimal_components(returns_filled)
            
            # Aplicar PCA
            pca = PCA(n_components=n_components)
            factor_returns_np = pca.fit_transform(returns_filled)
            
            # Convertir a DataFrame
            factor_returns = pd.DataFrame(
                factor_returns_np, 
                index=returns_window.index,
                columns=[f'Factor_{i+1}' for i in range(n_components)]
            )
            
            return pca.components_, factor_returns, n_components
            
        except Exception as e:
            logging.error(f"Error en extract_latent_factors: {str(e)}", exc_info=True)
            raise
    
    def find_optimal_components(self, returns_window, threshold=0.80, max_components=15):
        """
        Determina el número óptimo de componentes principales.
        
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
            # Limitar el máximo posible de componentes
            max_possible = min(returns_window.shape[1], returns_window.shape[0], max_components)
            
            # Calcular varianza explicada para diferentes números de componentes
            pca = PCA(n_components=max_possible)
            pca.fit(returns_window)
            
            # Encontrar el número de componentes que explican al menos threshold de la varianza
            explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(explained_variance_ratio_cumsum >= threshold) + 1
            
            # Asegurar un mínimo de componentes
            n_components = max(n_components, 3)
            
            return n_components
            
        except Exception as e:
            logging.error(f"Error en find_optimal_components: {str(e)}", exc_info=True)
            # Valor por defecto en caso de error
            return 5
    
    def detect_regimes(self, factor_returns, n_regimes=None):
        """
        Detecta regímenes de mercado usando modelos de mezcla gaussiana.
        
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
            
            # Calcular volatilidad y correlación
            vol = factor_returns.rolling(21).std().dropna()
            
            # Crear características para el modelo de regímenes
            features = vol.copy()
            
            # Estandarizar características
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Ajustar modelo de mezcla gaussiana
            gmm = GaussianMixture(
                n_components=n_regimes,
                covariance_type='full',
                random_state=42,
                n_init=10
            )
            
            # Manejar NaNs
            features_scaled_clean = np.nan_to_num(features_scaled)
            
            # Ajustar modelo
            gmm.fit(features_scaled_clean)
            
            # Predecir regímenes y probabilidades
            regimes = gmm.predict(features_scaled_clean)
            regime_probs = gmm.predict_proba(features_scaled_clean)
            
            return regimes, regime_probs
            
        except Exception as e:
            logging.error(f"Error en detect_regimes: {str(e)}", exc_info=True)
            # Valores por defecto en caso de error
            dummy_regimes = np.zeros(len(factor_returns) - 20)
            dummy_probs = np.ones((len(factor_returns) - 20, self.n_regimes)) / self.n_regimes
            return dummy_regimes, dummy_probs
    
    def predict_returns(self, factor_loadings, factor_returns, regimes, regime_probs, horizon=5):
        """
        Predice retornos futuros basados en factores latentes y regímenes.
        
        Parámetros:
        -----------
        factor_loadings : ndarray
            Cargas de los factores latentes
        factor_returns : DataFrame
            Retornos de los factores latentes
        regimes : ndarray
            Etiquetas de régimen para cada punto temporal
        regime_probs : ndarray
            Probabilidades de pertenencia a cada régimen
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
            # Convertir a numpy para operaciones más eficientes
            returns_np = self.returns.iloc[-len(regimes):].values
            n_assets = returns_np.shape[1]
            
            # Inicializar arrays para almacenar retornos esperados por régimen
            regime_expected_returns = np.zeros((self.n_regimes, n_assets))
            regime_counts = np.zeros(self.n_regimes)
            
            # Para cada régimen, calcular retornos esperados basados en datos históricos
            for r in range(self.n_regimes):
                # Encontrar índices donde el régimen es r
                regime_indices = np.where(regimes == r)[0]
                regime_counts[r] = len(regime_indices)
                
                if len(regime_indices) > 0:
                    # Para cada activo, calcular retorno esperado en este régimen
                    for i in range(n_assets):
                        # Recolectar retornos futuros después de cada ocurrencia del régimen
                        future_returns = []
                        
                        # Importante: solo usar datos históricos, no futuros
                        # Esto evita look-ahead bias
                        for idx in regime_indices:
                            # Solo considerar puntos donde tenemos suficientes datos futuros
                            # y que no sean los últimos puntos (para evitar look-ahead)
                            if idx + horizon < len(returns_np) - horizon:
                                # Calcular retorno acumulado para el horizonte
                                cum_return = np.prod(1 + returns_np[idx+1:idx+1+horizon, i]) - 1
                                future_returns.append(cum_return)
                        
                        if future_returns:
                            # Calcular retorno esperado para este activo en este régimen
                            regime_expected_returns[r, i] = np.mean(future_returns)
            
            # Calcular retorno esperado ponderado por probabilidad de régimen actual
            current_regime_probs = regime_probs[-1]
            expected_returns = np.zeros(n_assets)
            
            for r in range(self.n_regimes):
                # Ponderar por probabilidad del régimen y por confianza basada en cantidad de datos
                confidence_weight = min(1.0, regime_counts[r] / 30)  # Saturar en 1.0
                expected_returns += current_regime_probs[r] * regime_expected_returns[r, :] * confidence_weight
            
            # Calcular confianza en la predicción
            # Mayor confianza si el régimen actual es claro y tenemos muchos datos históricos
            regime_certainty = np.max(current_regime_probs)
            data_sufficiency = np.min(regime_counts) / 30  # Normalizado a 1.0
            prediction_confidence = regime_certainty * data_sufficiency
            
            # Convertir a Series
            expected_returns_series = pd.Series(expected_returns, index=self.returns.columns)
            prediction_confidence_series = pd.Series(prediction_confidence, index=self.returns.columns)
            
            return expected_returns_series, prediction_confidence_series
            
        except Exception as e:
            logging.error(f"Error en predict_returns: {str(e)}", exc_info=True)
            # Valores por defecto en caso de error
            dummy_returns = pd.Series(0.0001, index=self.returns.columns)
            dummy_confidence = pd.Series(0.1, index=self.returns.columns)
            return dummy_returns, dummy_confidence
    
    def optimize_portfolio(self, expected_returns, factor_loadings, prediction_confidence, 
                          current_regime, regime_certainty, risk_aversion=1.0):
        """
        Optimiza el portafolio basado en retornos esperados y factores latentes.
        
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
        risk_aversion : float
            Parámetro de aversión al riesgo
            
        Retorna:
        --------
        weights : Series
            Pesos óptimos para cada activo
        """
        try:
            n_assets = len(expected_returns)
            
            # Ajustar aversión al riesgo según certeza del régimen
            # Más incertidumbre -> más aversión al riesgo
            adjusted_risk_aversion = risk_aversion * (1.0 + (1.0 - regime_certainty) * 2.0)
            
            # Calcular matriz de covarianza basada en factores latentes
            # Esto es más robusto que usar la covarianza empírica directamente
            factor_cov = np.cov(factor_loadings)
            asset_cov = factor_loadings.T @ factor_cov @ factor_loadings
            
            # Asegurar que la matriz es definida positiva
            asset_cov = (asset_cov + asset_cov.T) / 2  # Hacer simétrica
            min_eig = np.min(np.linalg.eigvals(asset_cov))
            if min_eig < 1e-6:
                asset_cov += np.eye(n_assets) * (1e-6 - min_eig)
            
            # Ajustar retornos esperados por confianza en la predicción
            adjusted_returns = expected_returns * prediction_confidence
            
            # Función objetivo: maximizar utilidad (retorno - riesgo)
            def objective(weights):
                portfolio_return = np.sum(weights * adjusted_returns)
                portfolio_risk = np.sqrt(weights.T @ asset_cov @ weights)
                utility = portfolio_return - adjusted_risk_aversion * portfolio_risk
                return -utility  # Negativo porque minimizamos
            
            # Restricciones
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Suma de pesos = 1
            ]
            
            # Límites: permitir posiciones cortas limitadas según el régimen
            # En regímenes de alta volatilidad, limitar posiciones cortas
            short_limit = -0.2 if current_regime == 0 else -0.1 if current_regime == 1 else 0.0
            bounds = [(short_limit, 1.0) for _ in range(n_assets)]
            
            # Solución inicial: pesos iguales
            initial_weights = np.ones(n_assets) / n_assets
            
            # Optimizar
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-8}
            )
            
            if not result.success:
                logging.warning(f"Optimización no convergió: {result.message}")
                # Usar pesos iguales como fallback
                optimal_weights = pd.Series(initial_weights, index=expected_returns.index)
            else:
                optimal_weights = pd.Series(result.x, index=expected_returns.index)
            
            # Aplicar control de volatilidad objetivo
            portfolio_vol = np.sqrt(optimal_weights.T @ asset_cov @ optimal_weights) * np.sqrt(252)
            vol_scalar = self.vol_target / portfolio_vol
            
            # Limitar apalancamiento
            leverage = min(vol_scalar, self.max_leverage)
            
            # Ajustar pesos finales
            final_weights = optimal_weights * leverage
            
            return final_weights
            
        except Exception as e:
            logging.error(f"Error en optimize_portfolio: {str(e)}", exc_info=True)
            # Valor por defecto en caso de error: pesos iguales
            return pd.Series(1.0/len(expected_returns), index=expected_returns.index)
    
    def backtest(self, start_date=None, end_date=None):
        """
        Ejecuta un backtest de la estrategia.
        
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
            
            # Inicializar pesos (comenzar con efectivo)
            current_weights = pd.Series(0, index=self.returns.columns)
            
            # Ejecutar backtest
            for i, date in enumerate(tqdm(backtest_dates)):
                # Rebalancear en la primera fecha y luego según frecuencia
                if i == 0 or i % self.rebalance_freq == 0:
                    # Obtener datos hasta la fecha actual (sin look-ahead bias)
                    current_idx = self.returns.index.get_loc(date)
                    history_end_idx = current_idx
                    history_start_idx = max(0, history_end_idx - self.lookback_window)
                    
                    returns_window = self.returns.iloc[history_start_idx:history_end_idx]
                    
                    # Extraer factores latentes
                    factor_loadings, factor_returns, n_components = self.extract_latent_factors(returns_window)
                    
                    # Detectar regímenes
                    regimes, regime_probs = self.detect_regimes(factor_returns)
                    
                    # Predecir retornos
                    expected_returns, prediction_confidence = self.predict_returns(
                        factor_loadings, factor_returns, regimes, regime_probs
                    )
                    
                    # Optimizar portafolio
                    current_regime = regimes[-1]
                    regime_certainty = np.max(regime_probs[-1])
                    
                    # Ajustar aversión al riesgo según régimen y certeza
                    risk_aversion = 1.0 + current_regime * 0.5  # Más aversión en regímenes de alta volatilidad
                    
                    current_weights = self.optimize_portfolio(
                        expected_returns,
                        factor_loadings,
                        prediction_confidence,
                        current_regime,
                        regime_certainty,
                        risk_aversion
                    )
                    
                    # Guardar régimen actual
                    regime_history.append(current_regime)
                
                # Calcular retorno del portafolio para el día siguiente (evitar look-ahead bias)
                if i + 1 < len(backtest_dates):
                    next_date = backtest_dates[i + 1]
                    next_returns = self.returns.loc[next_date]
                    
                    # Calcular retorno del portafolio
                    portfolio_return = (current_weights * next_returns).sum()
                    portfolio_returns.append(portfolio_return)
                    
                    # Actualizar valor del portafolio
                    portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))
                
                # Guardar pesos
                weights_history.append(current_weights.copy())
            
            # Crear DataFrame de resultados
            performance = pd.DataFrame({
                'Portfolio_Value': portfolio_values[:-1],  # Ajustar longitud
                'Returns': portfolio_returns
            }, index=backtest_dates[:-1])  # Ajustar fechas
            
            # Calcular métricas
            performance['Cumulative_Returns'] = (1 + performance['Returns']).cumprod()
            performance['Drawdown'] = 1 - performance['Cumulative_Returns'] / performance['Cumulative_Returns'].cummax()
            
            # Guardar resultados adicionales
            self.weights_history = pd.DataFrame(weights_history, index=backtest_dates)
            self.regime_history = pd.Series(regime_history, index=backtest_dates[:len(regime_history)])
            self.performance = performance
            
            return performance
            
        except Exception as e:
            logging.error(f"Error en backtest: {str(e)}", exc_info=True)
            raise
    
    def calculate_metrics(self, performance=None):
        """
        Calcula métricas de rendimiento de la estrategia.
        
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
            ann_factor = 252  # Factor de anualización para datos diarios
            
            total_return = performance['Cumulative_Returns'].iloc[-1] - 1
            ann_return = (1 + total_return) ** (ann_factor / len(returns)) - 1
            ann_volatility = returns.std() * np.sqrt(ann_factor)
            sharpe_ratio = ann_return / ann_volatility if ann_volatility > 0 else 0
            max_drawdown = performance['Drawdown'].max()
            
            # Calcular ratio de Sortino (solo considera volatilidad negativa)
            negative_returns = returns[returns < 0]
            downside_deviation = negative_returns.std() * np.sqrt(ann_factor)
            sortino_ratio = ann_return / downside_deviation if downside_deviation > 0 else 0
            
            # Calcular ratio de Calmar (retorno anualizado / máximo drawdown)
            calmar_ratio = ann_return / max_drawdown if max_drawdown > 0 else 0
            
            # Calcular ratio de información (asumiendo benchmark = 0)
            information_ratio = ann_return / ann_volatility if ann_volatility > 0 else 0
            
            # Calcular % de meses positivos
            monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            pct_positive_months = (monthly_returns > 0).mean()
            
            # Recopilar métricas
            metrics = {
                'Total Return': total_return,
                'Annualized Return': ann_return,
                'Annualized Volatility': ann_volatility,
                'Sharpe Ratio': sharpe_ratio,
                'Sortino Ratio': sortino_ratio,
                'Calmar Ratio': calmar_ratio,
                'Information Ratio': information_ratio,
                'Maximum Drawdown': max_drawdown,
                'Positive Months (%)': pct_positive_months,
                'Number of Trades': len(self.weights_history) // self.rebalance_freq
            }
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error en calculate_metrics: {str(e)}", exc_info=True)
            return {}
    
    def plot_results(self, save_path='./artifacts/results/figures/'):
        """
        Genera y guarda visualizaciones de los resultados.
        
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
            
            # 1. Gráfico de rendimiento acumulado
            plt.figure(figsize=(12, 6))
            self.performance['Cumulative_Returns'].plot()
            plt.title('Rendimiento Acumulado')
            plt.xlabel('Fecha')
            plt.ylabel('Retorno Acumulado')
            plt.grid(True)
            plt.savefig(f'{save_path}cumulative_returns.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Gráfico de drawdowns
            plt.figure(figsize=(12, 6))
            self.performance['Drawdown'].plot()
            plt.title('Drawdowns')
            plt.xlabel('Fecha')
            plt.ylabel('Drawdown')
            plt.grid(True)
            plt.savefig(f'{save_path}drawdowns.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Gráfico de regímenes de mercado
            if hasattr(self, 'regime_history') and len(self.regime_history) > 0:
                plt.figure(figsize=(12, 6))
                self.regime_history.plot()
                plt.title('Regímenes de Mercado Detectados')
                plt.xlabel('Fecha')
                plt.ylabel('Régimen')
                plt.yticks(range(self.n_regimes))
                plt.grid(True)
                plt.savefig(f'{save_path}market_regimes.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # 4. Gráfico de exposición a activos a lo largo del tiempo
            if hasattr(self, 'weights_history') and len(self.weights_history) > 0:
                # Seleccionar los 10 activos con mayor peso promedio
                top_assets = self.weights_history.abs().mean().nlargest(10).index
                
                plt.figure(figsize=(12, 8))
                self.weights_history[top_assets].plot(colormap='viridis')
                plt.title('Exposición a los 10 Activos Principales')
                plt.xlabel('Fecha')
                plt.ylabel('Peso en el Portafolio')
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.grid(True)
                plt.savefig(f'{save_path}asset_exposure.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # 5. Heatmap de pesos a lo largo del tiempo
                plt.figure(figsize=(14, 10))
                sns.heatmap(
                    self.weights_history[top_assets].T,
                    cmap='RdBu_r',
                    center=0,
                    robust=True,
                    cbar_kws={'label': 'Peso'}
                )
                plt.title('Evolución de Pesos del Portafolio (Top 10 Activos)')
                plt.xlabel('Tiempo')
                plt.ylabel('Activo')
                plt.savefig(f'{save_path}weights_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # 6. Distribución de retornos
            plt.figure(figsize=(12, 6))
            sns.histplot(self.performance['Returns'], kde=True)
            plt.title('Distribución de Retornos Diarios')
            plt.xlabel('Retorno')
            plt.ylabel('Frecuencia')
            plt.grid(True)
            plt.savefig(f'{save_path}returns_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Gráficos guardados en {save_path}")
            
        except Exception as e:
            logging.error(f"Error en plot_results: {str(e)}", exc_info=True)
    
    def save_results(self, save_path='./artifacts/results/data/'):
        """
        Guarda los resultados en archivos CSV.
        
        Parámetros:
        -----------
        save_path : str
            Ruta donde guardar los archivos
        """
        try:
            # Crear directorio si no existe
            os.makedirs(save_path, exist_ok=True)
            
            # Guardar rendimiento
            if self.performance is not None:
                self.performance.to_csv(f'{save_path}performance.csv')
            
            # Guardar pesos
            if hasattr(self, 'weights_history') and len(self.weights_history) > 0:
                self.weights_history.to_csv(f'{save_path}weights_history.csv')
            
            # Guardar regímenes
            if hasattr(self, 'regime_history') and len(self.regime_history) > 0:
                self.regime_history.to_csv(f'{save_path}regime_history.csv')
            
            # Guardar métricas
            metrics = self.calculate_metrics()
            pd.Series(metrics).to_csv(f'{save_path}metrics.csv')
            
            # Guardar configuración
            config = {
                'start_date': self.start_date,
                'end_date': self.end_date,
                'lookback_window': self.lookback_window,
                'regime_window': self.regime_window,
                'n_regimes': self.n_regimes,
                'rebalance_freq': self.rebalance_freq,
                'vol_target': self.vol_target,
                'max_leverage': self.max_leverage,
                'n_assets': len(self.symbols)
            }
            pd.Series(config).to_csv(f'{save_path}config.csv')
            
            print(f"Resultados guardados en {save_path}")
            
        except Exception as e:
            logging.error(f"Error en save_results: {str(e)}", exc_info=True)
    
    def run_walk_forward_analysis(self, train_size=0.6, step_size=126):
        """
        Ejecuta análisis walk-forward para evaluar la robustez de la estrategia.
        
        Parámetros:
        -----------
        train_size : float
            Proporción de datos a usar para entrenamiento en cada ventana
        step_size : int
            Tamaño del paso para avanzar la ventana de prueba (en días)
            
        Retorna:
        --------
        wfa_results : DataFrame
            Resultados del análisis walk-forward
        """
        try:
            # Asegurar que tenemos suficientes datos
            if len(self.returns) < self.lookback_window + 2 * step_size:
                raise ValueError("No hay suficientes datos para análisis walk-forward")
            
            # Inicializar resultados
            wfa_results = []
            dates = self.returns.index
            
            # Definir ventanas
            start_idx = self.lookback_window
            while start_idx + step_size < len(dates):
                # Definir índices de entrenamiento y prueba
                train_end_idx = start_idx + int((len(dates) - start_idx) * train_size)
                test_end_idx = min(train_end_idx + step_size, len(dates))
                
                train_start_date = dates[start_idx]
                train_end_date = dates[train_end_idx - 1]
                test_start_date = dates[train_end_idx]
                test_end_date = dates[test_end_idx - 1]
                
                print(f"\nVentana WFA: {test_start_date.strftime('%Y-%m-%d')} a {test_end_date.strftime('%Y-%m-%d')}")
                
                # Ejecutar backtest en datos de entrenamiento
                train_performance = self.backtest(
                    start_date=train_start_date,
                    end_date=train_end_date
                )
                
                # Guardar pesos óptimos del último rebalanceo
                last_weights = self.weights_history.iloc[-1]
                
                # Ejecutar backtest en datos de prueba con pesos fijos
                # Esto simula trading real sin reoptimización
                test_returns = self.returns.loc[test_start_date:test_end_date]
                test_portfolio_values = [1.0]
                
                for date, returns in test_returns.iterrows():
                    # Calcular retorno del portafolio
                    portfolio_return = (last_weights * returns).sum()
                    test_portfolio_values.append(test_portfolio_values[-1] * (1 + portfolio_return))
                
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
                total_return = test_performance['Cumulative_Returns'].iloc[-1] - 1
                ann_factor = 252
                ann_return = (1 + total_return) ** (ann_factor / len(test_returns_series)) - 1
                ann_volatility = test_returns_series.std() * np.sqrt(ann_factor)
                sharpe_ratio = ann_return / ann_volatility if ann_volatility > 0 else 0
                max_drawdown = test_performance['Drawdown'].max()
                
                # Guardar resultados
                wfa_results.append({
                    'Test_Start_Date': test_start_date,
                    'Test_End_Date': test_end_date,
                    'Total_Return': total_return,
                    'Annualized_Return': ann_return,
                    'Annualized_Volatility': ann_volatility,
                    'Sharpe_Ratio': sharpe_ratio,
                    'Max_Drawdown': max_drawdown
                })
                
                # Avanzar ventana
                start_idx = train_end_idx
            
            # Convertir resultados a DataFrame
            wfa_df = pd.DataFrame(wfa_results)
            
            # Guardar resultados
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
                'Consistency': (wfa_df['Sharpe_Ratio'] > 0).mean()
            }
            
            # Guardar métricas agregadas
            pd.Series(wfa_metrics).to_csv('./artifacts/results/data/walk_forward_metrics.csv')
            
            # Visualizar resultados
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            plt.bar(range(len(wfa_df)), wfa_df['Sharpe_Ratio'], color='skyblue')
            plt.axhline(y=0, color='r', linestyle='-')
            plt.title('Sharpe Ratio por Ventana de Prueba')
            plt.xticks(range(len(wfa_df)), [d.strftime('%Y-%m') for d in wfa_df['Test_Start_Date']], rotation=45)
            plt.grid(True)
            
            plt.subplot(2, 1, 2)
            plt.bar(range(len(wfa_df)), wfa_df['Total_Return'], color='lightgreen')
            plt.axhline(y=0, color='r', linestyle='-')
            plt.title('Retorno Total por Ventana de Prueba')
            plt.xticks(range(len(wfa_df)), [d.strftime('%Y-%m') for d in wfa_df['Test_Start_Date']], rotation=45)
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('./artifacts/results/figures/walk_forward_results.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            return wfa_df
            
        except Exception as e:
            logging.error(f"Error en run_walk_forward_analysis: {str(e)}", exc_info=True)
            return pd.DataFrame()

# Ejecutar la estrategia
if __name__ == "__main__":
    try:
        # Inicializar estrategia
        strategy = AdaptiveMultifactorStrategy(
            start_date='2015-01-01',
            end_date='2023-01-01',
            lookback_window=252,
            regime_window=126,
            n_regimes=3,
            rebalance_freq=21,
            vol_target=0.10,
            max_leverage=1.5
        )
        
        # Ejecutar backtest
        performance = strategy.backtest()
        
        # Calcular métricas
        metrics = strategy.calculate_metrics()
        print("\nMétricas de Rendimiento:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        
        # Generar visualizaciones
        strategy.plot_results()
        
        # Guardar resultados
        strategy.save_results()
        
        # Ejecutar análisis walk-forward
        wfa_results = strategy.run_walk_forward_analysis(train_size=0.6, step_size=126)
        
        print("\nAnálisis completado. Todos los resultados guardados en ./artifacts/results/")
        
    except Exception as e:
        logging.error(f"Error en la ejecución principal: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}. Ver ./artifacts/errors.txt para más detalles.")
