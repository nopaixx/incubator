
import os
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from hmmlearn import hmm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import time
from tqdm import tqdm
import traceback

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

# Configuraciones para visualización
plt.style.use('seaborn-darkgrid')
sns.set(style="darkgrid")
warnings.filterwarnings("ignore")

# Función para obtener tickers del S&P 500
def get_sp500_tickers():
    """Obtiene los tickers del S&P 500 y sus sectores GICS"""
    try:
        sp500_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        tickers = sp500_table['Symbol'].str.replace('.', '-').tolist()
        sector_map = {k.replace('.', '-'): v for k, v in 
                     sp500_table.set_index('Symbol')['GICS Sector'].to_dict().items()}
        subsector_map = {k.replace('.', '-'): v for k, v in 
                        sp500_table.set_index('Symbol')['GICS Sub-Industry'].to_dict().items()}
        return tickers, sector_map, subsector_map
    except Exception as e:
        logging.error(f"Error obteniendo tickers del S&P 500: {str(e)}")
        return [], {}, {}

# Función para descargar datos históricos
def get_historical_data(tickers, start_date, end_date, sleep_time=0.1):
    """Descarga datos históricos para los tickers especificados"""
    data = {}
    failed_tickers = []
    
    for ticker in tqdm(tickers, desc="Descargando datos"):
        try:
            ticker_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if len(ticker_data) > 60:  # Asegurar datos suficientes
                data[ticker] = ticker_data
            else:
                failed_tickers.append(ticker)
            time.sleep(sleep_time)  # Evitar sobrecargar la API
        except Exception as e:
            logging.error(f"Error descargando datos para {ticker}: {str(e)}")
            failed_tickers.append(ticker)
    
    print(f"Datos obtenidos para {len(data)} tickers. Fallaron {len(failed_tickers)} tickers.")
    return data

# Preprocesamiento de datos
def preprocess_data(data):
    """Preprocesa los datos históricos para la estrategia"""
    tickers = list(data.keys())
    
    # Obtener fechas comunes
    all_dates = set()
    for ticker_data in data.values():
        all_dates.update(ticker_data.index)
    all_dates = sorted(list(all_dates))
    
    # Crear DataFrames alineados
    prices = pd.DataFrame(index=all_dates, columns=tickers)
    volumes = pd.DataFrame(index=all_dates, columns=tickers)
    
    for ticker in tickers:
        prices.loc[data[ticker].index, ticker] = data[ticker]['Close']
        volumes.loc[data[ticker].index, ticker] = data[ticker]['Volume']
    
    # Calcular métricas derivadas
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan)
    realized_vol = returns.rolling(window=22).std() * np.sqrt(252)
    relative_volume = volumes / volumes.rolling(window=20).mean()
    
    # Manejar outliers en retornos
    mad = lambda x: np.abs(x - x.median()).median()
    returns_mad = returns.apply(mad)
    outlier_threshold = 3.5
    
    for col in returns.columns:
        threshold = outlier_threshold * returns_mad[col]
        outliers = np.abs(returns[col]) > threshold
        if outliers.any():
            returns.loc[outliers, col] = np.sign(returns.loc[outliers, col]) * threshold
    
    # Imputar datos faltantes (gaps < 3 días)
    prices_filled = prices.interpolate(method='linear', limit=3, axis=0)
    
    # Calcular ADV en dólares
    adv = (volumes * prices).rolling(window=20).mean()
    
    # Recalcular retornos con precios imputados
    returns_filled = prices_filled.pct_change().replace([np.inf, -np.inf], np.nan)
    
    processed_data = {
        'prices': prices_filled,
        'returns': returns_filled,
        'volumes': volumes,
        'realized_vol': realized_vol,
        'relative_volume': relative_volume,
        'adv': adv
    }
    
    return processed_data

# Sistema de Detección de Regímenes
class RegimeDetector:
    """Implementa el sistema de detección de regímenes usando HMM"""
    
    def __init__(self, min_train_years=5, persistence_days=3, prob_threshold=0.85):
        self.min_train_years = min_train_years
        self.persistence_days = persistence_days
        self.prob_threshold = prob_threshold
        self.hmm_models = {
            'hmm1': None,  # 2 estados, vol + correlaciones
            'hmm2': None,  # 3 estados, retornos + vol + volumen
            'hmm3': None   # 2 estados, dispersión + breadth
        }
        self.model_weights = {'hmm1': 1/3, 'hmm2': 1/3, 'hmm3': 1/3}
        self.last_calibration = None
        self.regime_history = None
        self.current_regime = None
        self.sector_map = {}
    
    def create_features(self, data, market_index='^GSPC'):
        """Crea features para los modelos HMM"""
        prices = data['prices']
        returns = data['returns']
        realized_vol = data['realized_vol']
        relative_volume = data['relative_volume']
        
        # Obtener datos de mercado si no están en los datos
        if market_index not in prices.columns:
            try:
                market_data = yf.download(market_index, 
                                      start=prices.index[0].strftime('%Y-%m-%d'),
                                      end=prices.index[-1].strftime('%Y-%m-%d'),
                                      progress=False)
                market_returns = market_data['Close'].pct_change().dropna()
                market_vol = market_returns.rolling(window=22).std() * np.sqrt(252)
            except Exception as e:
                logging.error(f"Error obteniendo datos de mercado: {str(e)}")
                # Usar promedio como proxy
                market_returns = returns.mean(axis=1)
                market_vol = market_returns.rolling(window=22).std() * np.sqrt(252)
        else:
            market_returns = returns[market_index]
            market_vol = realized_vol[market_index]
        
        # Features para HMM-1: volatilidad y correlaciones sectoriales
        vix_proxy = market_vol.dropna()
        
        # Calcular correlaciones entre sectores
        sector_returns = {}
        for sector in set(self.sector_map.values()):
            sector_tickers = [t for t in returns.columns if self.sector_map.get(t) == sector]
            if sector_tickers:
                sector_returns[sector] = returns[sector_tickers].mean(axis=1)
        
        sector_corr = pd.DataFrame(index=prices.index)
        if len(sector_returns) > 1:
            sectors = list(sector_returns.keys())
            for i in range(len(sectors)):
                for j in range(i+1, len(sectors)):
                    s1, s2 = sectors[i], sectors[j]
                    pair_df = pd.DataFrame({s1: sector_returns[s1], s2: sector_returns[s2]})
                    rolling_corr = pair_df.rolling(window=22).corr().iloc[1::2][s1].values
                    sector_corr[f'{s1}_{s2}'] = np.nan
                    sector_corr.loc[pair_df.index[21:], f'{s1}_{s2}'] = rolling_corr
            
            mean_sector_corr = sector_corr.mean(axis=1)
        else:
            mean_sector_corr = pd.Series(index=prices.index, data=0.5)
        
        # Features para HMM-2: retornos, volatilidad y volumen relativo
        market_rets = market_returns.dropna()
        avg_rel_vol = relative_volume.mean(axis=1)
        
        # Features para HMM-3: dispersión sectorial y breadth de mercado
        if len(sector_returns) > 1:
            sector_dispersion = pd.DataFrame(sector_returns).std(axis=1)
        else:
            sector_dispersion = pd.Series(index=prices.index, data=0.01)
        
        # Calcular market breadth (% de acciones sobre su MA50)
        above_ma50 = pd.DataFrame(index=prices.index, columns=prices.columns, data=False)
        ma50 = prices.rolling(window=50).mean()
        for col in prices.columns:
            above_ma50[col] = prices[col] > ma50[col]
        
        market_breadth = above_ma50.mean(axis=1)
        
        # Alinear y eliminar NaNs
        features_hmm1 = pd.DataFrame({
            'volatility': vix_proxy,
            'sector_correlation': mean_sector_corr
        }).dropna()
        
        features_hmm2 = pd.DataFrame({
            'returns': market_rets,
            'volatility': market_vol,
            'relative_volume': avg_rel_vol
        }).dropna()
        
        features_hmm3 = pd.DataFrame({
            'sector_dispersion': sector_dispersion,
            'market_breadth': market_breadth
        }).dropna()
        
        return {
            'hmm1': features_hmm1,
            'hmm2': features_hmm2,
            'hmm3': features_hmm3
        }
    
    def train_models(self, features, end_date=None):
        """Entrena los modelos HMM con las features proporcionadas"""
        self.hmm_models = {
            'hmm1': hmm.GaussianHMM(n_components=2, covariance_type="full", 
                                    n_iter=1000, random_state=42),
            'hmm2': hmm.GaussianHMM(n_components=3, covariance_type="full", 
                                    n_iter=1000, random_state=42),
            'hmm3': hmm.GaussianHMM(n_components=2, covariance_type="full", 
                                    n_iter=1000, random_state=42)
        }
        
        # Limitar datos a la fecha de fin si se proporciona
        if end_date:
            for key in features:
                features[key] = features[key][features[key].index <= end_date]
        
        # Entrenar modelos
        for key, model in self.hmm_models.items():
            if key in features and len(features[key]) > 0:
                X = StandardScaler().fit_transform(features[key])
                try:
                    self.hmm_models[key].fit(X)
                except Exception as e:
                    logging.error(f"Error entrenando modelo {key}: {str(e)}")
                    
        self.last_calibration = datetime.now()
    
    def predict_regimes(self, features, apply_persistence=True):
        """Predice regímenes con los modelos HMM"""
        predictions = {}
        
        # Predecir con cada modelo
        for key, model in self.hmm_models.items():
            if model is not None and key in features and len(features[key]) > 0:
                X = StandardScaler().fit_transform(features[key])
                try:
                    hidden_states = model.predict(X)
                    probs = model.predict_proba(X)
                    
                    pred_df = pd.DataFrame(index=features[key].index)
                    pred_df['state'] = hidden_states
                    
                    for i in range(model.n_components):
                        pred_df[f'prob_state_{i}'] = probs[:, i]
                    
                    predictions[key] = pred_df
                except Exception as e:
                    logging.error(f"Error prediciendo con modelo {key}: {str(e)}")
        
        if not predictions:
            return pd.Series(dtype='int')
        
        # Alinear índices
        all_indices = set()
        for pred_df in predictions.values():
            all_indices.update(pred_df.index)
        all_indices = sorted(list(all_indices))
        
        for key in predictions:
            predictions[key] = predictions[key].reindex(all_indices)
        
        # Combinar predicciones
        combined_df = pd.DataFrame(index=all_indices)
        
        if 'hmm1' in predictions:
            combined_df['hmm1_regime'] = predictions['hmm1']['state']
        if 'hmm2' in predictions:
            combined_df['hmm2_regime'] = predictions['hmm2']['state']
        if 'hmm3' in predictions:
            combined_df['hmm3_regime'] = predictions['hmm3']['state']
        
        # Mapear a régimen final (3 regímenes)
        def map_regime(row):
            hmm1 = row.get('hmm1_regime', 0)
            hmm2 = row.get('hmm2_regime', 1)
            hmm3 = row.get('hmm3_regime', 0)
            
            # Régimen 3 (Crisis): Alta volatilidad
            if hmm1 == 1 and hmm2 == 2:
                return 3
            # Régimen 1 (Favorable): Baja volatilidad y baja dispersión
            elif hmm1 == 0 and hmm3 == 1:
                return 1
            # Régimen 2 (Transición): Otros casos
            else:
                return 2
        
        combined_df['final_regime'] = combined_df.apply(map_regime, axis=1)
        
        # Aplicar filtro de persistencia
        if apply_persistence and len(combined_df) > self.persistence_days:
            final_regimes = combined_df['final_regime'].copy()
            
            for i in range(self.persistence_days, len(final_regimes)):
                window = combined_df['final_regime'].iloc[i-self.persistence_days:i]
                most_common = window.value_counts().idxmax()
                if (window == most_common).mean() >= self.prob_threshold:
                    final_regimes.iloc[i] = most_common
                else:
                    final_regimes.iloc[i] = final_regimes.iloc[i-1]
            
            combined_df['filtered_regime'] = final_regimes
            regime_series = combined_df['filtered_regime']
        else:
            regime_series = combined_df['final_regime']
        
        # Retrasar 2 días para evitar look-ahead bias
        if len(regime_series) > 2:
            regime_series = regime_series.shift(2).fillna(method='bfill')
        
        self.regime_history = regime_series
        self.current_regime = regime_series.iloc[-1] if not regime_series.empty else None
        
        return regime_series
    
    def fit_predict(self, data, sector_map, current_date=None, force_calibration=False):
        """Entrena y predice el régimen actual"""
        self.sector_map = sector_map
        
        if current_date is None:
            current_date = data['prices'].index[-1]
        
        # Verificar datos suficientes
        training_days_required = 252 * self.min_train_years
        if (data['prices'].index[-1] - data['prices'].index[0]).days < training_days_required:
            return 1  # Régimen por defecto con datos insuficientes
        
        # Crear features
        features = self.create_features(data)
        
        # Entrenar modelos si es necesario
        need_training = (force_calibration or 
                         self.hmm_models['hmm1'] is None or 
                         (self.last_calibration and 
                          (current_date - self.last_calibration.date()).days > 7))
        
        if need_training:
            self.train_models(features)
        
        # Predecir regímenes
        regimes = self.predict_regimes(features)
        
        # Devolver régimen actual
        if not regimes.empty:
            current_regime = regimes.iloc[-1]
        else:
            current_regime = 1  # Régimen por defecto
        
        return current_regime

# Componente de Selección de Pares
class PairSelector:
    """Sistema de identificación y selección de pares para arbitraje"""
    
    def __init__(self, min_liquidity=10e6, 
                 max_pairs_by_regime={1: 25, 2: 20, 3: 15}):
        self.min_liquidity = min_liquidity
        self.max_pairs_by_regime = max_pairs_by_regime
        self.candidate_pairs = None
        self.selected_pairs = None
    
    def prefilter_by_liquidity(self, data, min_adv=None):
        """Filtro de tickers por liquidez mínima"""
        if min_adv is None:
            min_adv = self.min_liquidity
            
        adv = data['adv'].iloc[-20:].mean()
        liquid_tickers = adv[adv > min_adv].index.tolist()
        
        return liquid_tickers
    
    def filter_by_events(self, tickers, days_ahead=7):
        """Filtro de tickers con eventos próximos (simulado)"""
        # En implementación real, usaríamos datos de calendarios de eventos
        np.random.seed(42)  # Para reproducibilidad
        filter_out = np.random.choice(tickers, size=int(len(tickers) * 0.1), replace=False)
        
        return [t for t in tickers if t not in filter_out]
    
    def calculate_similarity_score(self, data, tickers, lookback=252):
        """Calcula scores de similitud entre tickers"""
        prices = data['prices'].iloc[-lookback:][tickers]
        returns = data['returns'].iloc[-lookback:][tickers]
        vols = data['realized_vol'].iloc[-lookback:][tickers]
        
        similarity = pd.DataFrame(0, index=tickers, columns=tickers)
        
        # Beta a 1 año (proxy: correlación con mercado)
        market_returns = returns.mean(axis=1)
        betas = {}
        
        for ticker in tickers:
            X = market_returns.values.reshape(-1, 1)
            y = returns[ticker].values
            mask = ~np.isnan(X.flatten()) & ~np.isnan(y)
            if mask.sum() > 30:
                X, y = X[mask], y[mask]
                beta = np.cov(y, X.flatten())[0, 1] / np.var(X.flatten())
                betas[ticker] = beta
            else:
                betas[ticker] = 1.0
        
        # Capitalización (proxy: precio)
        mcap = prices.iloc[-1]
        mcap = (mcap - mcap.min()) / (mcap.max() - mcap.min())
        
        # Volatilidad realizada
        vol = vols.iloc[-60:].mean()
        vol = (vol - vol.min()) / (vol.max() - vol.min())
        
        # Ratios de valoración (simulados)
        np.random.seed(42)
        pe_ratios = pd.Series(np.random.uniform(10, 30, size=len(tickers)), index=tickers)
        pb_ratios = pd.Series(np.random.uniform(1, 5, size=len(tickers)), index=tickers)
        
        # Calcular similitud
        for i, ticker1 in enumerate(tickers):
            for j, ticker2 in enumerate(tickers):
                if i < j:
                    # Beta similarity (30%)
                    beta_sim = 1 - min(abs(betas[ticker1] - betas[ticker2]) / 2, 1)
                    
                    # Market cap similarity (25%)
                    mcap_sim = 1 - abs(mcap[ticker1] - mcap[ticker2])
                    
                    # Volatility similarity (25%)
                    vol_sim = 1 - abs(vol[ticker1] - vol[ticker2])
                    
                    # Valuation similarity (20%)
                    pe_sim = 1 - min(abs(pe_ratios[ticker1] - pe_ratios[ticker2]) / 20, 1)
                    pb_sim = 1 - min(abs(pb_ratios[ticker1] - pb_ratios[ticker2]) / 4, 1)
                    val_sim = (pe_sim + pb_sim) / 2
                    
                    # Weighted score
                    sim_score = 0.3*beta_sim + 0.25*mcap_sim + 0.25*vol_sim + 0.2*val_sim
                    similarity.loc[ticker1, ticker2] = sim_score
                    similarity.loc[ticker2, ticker1] = sim_score
        
        return similarity
    
    def generate_candidate_pairs(self, data, sector_map, subsector_map, similarity_threshold=0.7):
        """Genera pares candidatos basados en sector/subsector y similitud"""
        # Filtrar por liquidez y eventos
        liquid_tickers = self.prefilter_by_liquidity(data)
        filtered_tickers = self.filter_by_events(liquid_tickers)
        
        if len(filtered_tickers) < 2:
            return []
        
        # Calcular scores de similitud
        similarity = self.calculate_similarity_score(data, filtered_tickers)
        
        # Generar pares candidatos
        candidate_pairs = []
        
        # Agrupar por sector y subsector
        sector_groups = {}
        subsector_groups = {}
        
        for ticker in filtered_tickers:
            sector = sector_map.get(ticker)
            subsector = subsector_map.get(ticker)
            
            if sector:
                if sector not in sector_groups:
                    sector_groups[sector] = []
                sector_groups[sector].append(ticker)
            
            if subsector:
                if subsector not in subsector_groups:
                    subsector_groups[subsector] = []
                subsector_groups[subsector].append(ticker)
        
        # Buscar pares en mismo subsector
        for subsector, tickers in subsector_groups.items():
            if len(tickers) < 2:
                continue
                
            for i, ticker1 in enumerate(tickers):
                for ticker2 in tickers[i+1:]:
                    if similarity.loc[ticker1, ticker2] >= similarity_threshold:
                        candidate_pairs.append((ticker1, ticker2))
        
        # Buscar en mismo sector si necesitamos más pares
        if len(candidate_pairs) < 50:
            for sector, tickers in sector_groups.items():
                if len(tickers) < 2:
                    continue
                    
                for i, ticker1 in enumerate(tickers):
                    for ticker2 in tickers[i+1:]:
                        if (ticker1, ticker2) not in candidate_pairs and (ticker2, ticker1) not in candidate_pairs:
                            if similarity.loc[ticker1, ticker2] >= similarity_threshold * 0.9:
                                candidate_pairs.append((ticker1, ticker2))
        
        self.candidate_pairs = candidate_pairs
        return candidate_pairs
    
    def test_cointegration(self, data, pair, window, significance=0.05):
        """Prueba de cointegración de Johansen para un par"""
        ticker1, ticker2 = pair
        prices = data['prices'].iloc[-window:][list(pair)]
        
        if len(prices) < 60 or prices[ticker1].isna().any() or prices[ticker2].isna().any():
            return {'coint': False, 'pvalue': 1.0, 'half_life': np.inf, 'hedge_ratio': 1.0}
        
        try:
            # Test de Johansen
            result = coint_johansen(prices, det_order=0, k_ar_diff=1)
            trace_stat = result.lr1[0]  # Estadístico de traza para r=0
            crit_value = result.cvt[0, 1]  # Valor crítico al 5%
            
            coint = trace_stat > crit_value
            
            if coint:
                # Vector de cointegración normalizado
                coef = result.evec[:, 0]
                hedge_ratio = -coef[1] / coef[0]
                
                # Calcular spread
                spread = prices[ticker1] + hedge_ratio * prices[ticker2]
                
                # Estimar half-life
                lagged_spread = spread.shift(1).dropna()
                delta_spread = spread.diff().dropna()
                
                if len(lagged_spread) > 30:
                    # Regresión para modelo AR(1)
                    X = lagged_spread.values.reshape(-1, 1)
                    y = delta_spread.values
                    X = np.hstack([np.ones_like(X), X])
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    
                    half_life = -np.log(2) / beta[1] if beta[1] < 0 else np.inf
                    
                    if half_life <= 0 or half_life > 252:
                        coint = False
                        half_life = np.inf
                else:
                    half_life = np.inf
                    coint = False
            else:
                hedge_ratio = 1.0
                half_life = np.inf
            
            # P-valor aproximado
            p_value = 0.01 if trace_stat > result.cvt[0, 0] else 0.05 if trace_stat > result.cvt[0, 1] else 0.1 if trace_stat > result.cvt[0, 2] else 1.0
            
            return {
                'coint': coint,
                'pvalue': p_value,
                'half_life': half_life,
                'hedge_ratio': hedge_ratio
            }
        
        except Exception as e:
            logging.error(f"Error en test de cointegración para {pair}: {str(e)}")
            return {'coint': False, 'pvalue': 1.0, 'half_life': np.inf, 'hedge_ratio': 1.0}
    
    def test_structural_stability(self, data, pair, hedge_ratio, lookback=252):
        """Evalúa la estabilidad estructural de la relación"""
        ticker1, ticker2 = pair
        prices = data['prices'].iloc[-lookback:][list(pair)]
        
        if prices[ticker1].isna().any() or prices[ticker2].isna().any() or len(prices) < 60:
            return 1.0  # Baja estabilidad por defecto
        
        # Calcular spread histórico
        spread = prices[ticker1] + hedge_ratio * prices[ticker2]
        
        # Evaluar estabilidad con ventanas móviles
        stability_scores = []
        
        # Dividir en 4 sub-periodos
        subperiod_length = len(spread) // 4
        
        for i in range(4):
            start_idx = i * subperiod_length
            end_idx = (i+1) * subperiod_length if i < 3 else len(spread)
            subspread = spread.iloc[start_idx:end_idx]
                
            if len(subspread) < 30:
                continue
                
            # Test ADF en el sub-periodo
            try:
                adf_result = adfuller(subspread, maxlag=10)
                is_stationary = adf_result[1] < 0.05
                stability_scores.append(1 if is_stationary else 0)
            except:
                stability_scores.append(0)
        
        # Verificar consistencia de estacionariedad
        stationarity_consistency = sum(stability_scores) / len(stability_scores) if stability_scores else 0
        
        # Calcular volatilidad del spread en cada sub-periodo
        subperiod_vols = []
        
        for i in range(4):
            start_idx = i * subperiod_length
            end_idx = (i+1) * subperiod_length if i < 3 else len(spread)
            subspread = spread.iloc[start_idx:end_idx]
                
            if len(subspread) < 10:
                continue
                
            subperiod_vols.append(subspread.std())
        
        # Consistencia de volatilidad
        vol_consistency = 1.0
        if len(subperiod_vols) > 1:
            vol_ratio = max(subperiod_vols) / min(subperiod_vols)
            vol_consistency = 1.0 / min(vol_ratio, 5.0)
        
        # Puntuación final (1-10)
        stability_score = 1.0 + 9.0 * (0.7 * stationarity_consistency + 0.3 * vol_consistency)
        
        return stability_score
    
    def select_pairs(self, data, regime, candidate_pairs=None):
        """Selecciona los mejores pares para operar según el régimen actual"""
        if candidate_pairs is None:
            candidate_pairs = self.candidate_pairs
            
        if not candidate_pairs:
            return []
        
        # Parámetros según régimen
        if regime == 1:  # Baja volatilidad / Alta predictibilidad
            window = 252
            half_life_range = (10, 25)
            max_pvalue = 0.01
            max_pairs = self.max_pairs_by_regime.get(1, 25)
        elif regime == 2:  # Transición / Volatilidad moderada
            window = 180
            half_life_range = (7, 20)
            max_pvalue = 0.03
            max_pairs = self.max_pairs_by_regime.get(2, 20)
        else:  # Crisis / Alta volatilidad
            window = 126
            half_life_range = (5, 15)
            max_pvalue = 0.05
            max_pairs = self.max_pairs_by_regime.get(3, 15)
        
        # Evaluar pares candidatos
        pair_results = []
        
        for pair in tqdm(candidate_pairs, desc="Evaluando cointegración"):
            ticker1, ticker2 = pair
            
            # Test de cointegración
            coint_result = self.test_cointegration(data, pair, window, max_pvalue)
            
            if not coint_result['coint']:
                continue
                
            if not half_life_range[0] <= coint_result['half_life'] <= half_life_range[1]:
                continue
                
            # Evaluar estabilidad estructural
            stability = self.test_structural_stability(data, pair, coint_result['hedge_ratio'])
            
            # Calcular liquidez combinada
            ticker1_adv = data['adv'].iloc[-20:][ticker1].mean()
            ticker2_adv = data['adv'].iloc[-20:][ticker2].mean()
            combined_liquidity = min(ticker1_adv, ticker2_adv)
            
            # Rendimiento histórico (simulado)
            np.random.seed(hash(f"{ticker1}_{ticker2}") % 2**32)
            historical_performance = np.random.uniform(0.5, 1.5)
            
            # Score compuesto según régimen
            if regime == 1:
                composite_score = (0.35 * (1/coint_result['pvalue']) + 
                                   0.35 * stability + 
                                   0.2 * (combined_liquidity/1e7) + 
                                   0.1 * historical_performance)
            elif regime == 2:
                composite_score = (0.4 * (1/coint_result['pvalue']) + 
                                   0.3 * stability + 
                                   0.2 * (combined_liquidity/1e7) + 
                                   0.1 * historical_performance)
            else:
                composite_score = (0.4 * (1/coint_result['pvalue']) + 
                                   0.3 * stability + 
                                   0.2 * (combined_liquidity/1e7) + 
                                   0.1 * historical_performance)
            
            # Guardar resultados
            pair_results.append({
                'ticker1': ticker1,
                'ticker2': ticker2,
                'hedge_ratio': coint_result['hedge_ratio'],
                'half_life': coint_result['half_life'],
                'pvalue': coint_result['pvalue'],
                'stability': stability,
                'liquidity': combined_liquidity,
                'composite_score': composite_score
            })
        
        # Ordenar por score compuesto
        sorted_pairs = sorted(pair_results, key=lambda x: x['composite_score'], reverse=True)
        
        # Seleccionar mejores pares
        selected_pairs = sorted_pairs[:max_pairs]
        
        self.selected_pairs = selected_pairs
        return selected_pairs

# Modelo Predictivo de Convergencia
class ConvergencePredictor:
    """Modelo predictivo para la probabilidad de convergencia"""
    
    def __init__(self, learning_rate=0.01, max_depth=4, n_estimators=200):
        self.models = {
            1: GradientBoostingClassifier(learning_rate=learning_rate, max_depth=max_depth, 
                                          n_estimators=n_estimators, subsample=0.8,
                                          random_state=42),
            2: GradientBoostingClassifier(learning_rate=learning_rate, max_depth=max_depth, 
                                          n_estimators=n_estimators, subsample=0.8,
                                          random_state=42),
            3: GradientBoostingClassifier(learning_rate=learning_rate, max_depth=max_depth, 
                                          n_estimators=n_estimators, subsample=0.8,
                                          random_state=42)
        }
        self.scalers = {1: StandardScaler(), 2: StandardScaler(), 3: StandardScaler()}
        self.is_trained = {1: False, 2: False, 3: False}
    
    def create_features(self, data, pair, hedge_ratio, regime, lookback=252):
        """Crea features para el modelo de predicción"""
        ticker1, ticker2 = pair
        prices = data['prices'].iloc[-lookback:][list(pair)]
        
        if len(prices) < 60 or prices[ticker1].isna().any() or prices[ticker2].isna().any():
            return None
        
        # Calcular spread y z-score
        spread = prices[ticker1] + hedge_ratio * prices[ticker2]
        spread_mean = spread.rolling(window=60).mean()
        spread_std = spread.rolling(window=60).std()
        z_score = (spread - spread_mean) / spread_std
        
        # Crear DataFrame de features
        features = pd.DataFrame(index=z_score.index)
        
        # Z-score y cambios
        features['z_score'] = z_score
        features['z_score_change_3d'] = z_score - z_score.shift(3)
        features['z_score_change_5d'] = z_score - z_score.shift(5)
        features['z_score_change_10d'] = z_score - z_score.shift(10)
        
        # Volatilidad relativa
        spread_vol = spread_std / spread_mean.abs()
        spread_vol_60d_avg = spread_vol.rolling(window=60).mean()
        features['rel_vol'] = spread_vol / spread_vol_60d_avg
        
        # Ratio de volumen anormal
        vol1 = data['relative_volume'][ticker1].iloc[-lookback:] if ticker1 in data['relative_volume'] else pd.Series(1, index=prices.index)
        vol2 = data['relative_volume'][ticker2].iloc[-lookback:] if ticker2 in data['relative_volume'] else pd.Series(1, index=prices.index)
        features['abnormal_volume'] = (vol1 + vol2) / 2
        
        # Variables dummy de régimen
        features['regime_1'] = 1 if regime == 1 else 0
        features['regime_2'] = 1 if regime == 2 else 0
        features['regime_3'] = 1 if regime == 3 else 0
        
        # Eliminar NaNs
        features = features.dropna()
        
        return features
    
    def create_target(self, data, pair, hedge_ratio, forward_period=10, threshold=0.5):
        """Crea variable objetivo para entrenamiento"""
        ticker1, ticker2 = pair
        prices = data['prices'][list(pair)]
        
        # Calcular spread y z-score
        spread = prices[ticker1] + hedge_ratio * prices[ticker2]
        spread_mean = spread.rolling(window=60).mean()
        spread_std = spread.rolling(window=60).std()
        z_score = (spread - spread_mean) / spread_std
        
        # Crear variable objetivo
        target = pd.Series(index=z_score.index, data=0)
        
        # Determinar convergencia futura
        for i in range(len(z_score) - forward_period):
            current_z = z_score.iloc[i]
            
            if abs(current_z) > 1.0:
                future_z = z_score.iloc[i+1:i+forward_period+1]
                min_distance = future_z.abs().min()
                
                if min_distance < threshold * abs(current_z):
                    target.iloc[i] = 1
        
        return target
    
    def train(self, data, selected_pairs, regime, max_history=1260):
        """Entrena el modelo para un régimen específico"""
        prices = data['prices'].iloc[-max_history:]
        
        all_features = []
        all_targets = []
        
        for pair_info in selected_pairs:
            ticker1 = pair_info['ticker1']
            ticker2 = pair_info['ticker2']
            hedge_ratio = pair_info['hedge_ratio']
            half_life = pair_info['half_life']
            
            # Crear features
            pair = (ticker1, ticker2)
            features = self.create_features(data, pair, hedge_ratio, regime)
            
            if features is None or len(features) < 60:
                continue
                
            # Añadir half-life como feature
            features['half_life'] = half_life
            
            # Crear target
            target = self.create_target(data, pair, hedge_ratio)
            
            # Alinear features y target
            common_index = features.index.intersection(target.index)
            if len(common_index) < 30:
                continue
                
            features = features.loc[common_index]
            target = target.loc[common_index]
            
            all_features.append(features)
            all_targets.append(target)
        
        if not all_features:
            return False
            
        # Concatenar datos
        X = pd.concat(all_features)
        y = pd.concat(all_targets)
        
        # Validación cruzada temporal
        tscv = TimeSeriesSplit(n_splits=5)
        best_score = 0
        
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Escalar features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Entrenar modelo
            model = self.models[regime]
            model.fit(X_train_scaled, y_train)
            
            # Evaluar
            score = model.score(X_test_scaled, y_test)
            if score > best_score:
                best_score = score
                self.scalers[regime] = scaler
        
        # Modelo final con todos los datos
        X_scaled = self.scalers[regime].fit_transform(X)
        self.models[regime].fit(X_scaled, y)
        self.is_trained[regime] = True
        
        return True
    
    def predict_convergence(self, data, pair_info, regime):
        """Predice probabilidad de convergencia para un par"""
        if not self.is_trained[regime]:
            return 0.5  # Valor neutro si no está entrenado
            
        ticker1 = pair_info['ticker1']
        ticker2 = pair_info['ticker2']
        hedge_ratio = pair_info['hedge_ratio']
        half_life = pair_info['half_life']
        
        # Crear features
        pair = (ticker1, ticker2)
        features = self.create_features(data, pair, hedge_ratio, regime, lookback=60)
        
        if features is None or len(features) < 5:
            return 0.5
            
        # Añadir half-life
        features['half_life'] = half_life
        
        # Usar solo la última fila
        latest_features = features.iloc[-1:].copy()
        
        # Escalar features
        latest_features_scaled = self.scalers[regime].transform(latest_features)
        
        # Predecir probabilidad
        probability = self.models[regime].predict_proba(latest_features_scaled)[0, 1]
        
        return probability

# Generador de Señales
class SignalGenerator:
    """Genera señales de trading basadas en z-scores y regímenes"""
    
    def __init__(self):
        # Umbrales de entrada por régimen
        self.entry_thresholds = {
            1: {'long': -2.0, 'short': 2.0},
            2: {'long': -2.2, 'short': 2.2},
            3: {'long': -2.5, 'short': 2.5}
        }
        
        # Umbrales de salida por régimen
        self.exit_thresholds = {
            1: {'long': -0.5, 'short': 0.5},
            2: {'long': -0.7, 'short': 0.7},
            3: {'long': -1.0, 'short': 1.0}
        }
        
        # Bandas de no-transacción
        self.no_trade_base_width = 0.2
    
    def calculate_z_score(self, data, pair_info, lookback=60):
        """Calcula z-score actual para un par"""
        ticker1 = pair_info['ticker1']
        ticker2 = pair_info['ticker2']
        hedge_ratio = pair_info['hedge_ratio']
        
        # Obtener precios
        prices = data['prices'].iloc[-lookback:][[ticker1, ticker2]]
        
        if len(prices) < lookback/2 or prices[ticker1].isna().any() or prices[ticker2].isna().any():
            return None
        
        # Calcular spread
        spread = prices[ticker1] + hedge_ratio * prices[ticker2]
        
        # Calcular media y desviación con EWMA
        spread_mean = spread.ewm(halflife=60).mean().iloc[-1]
        spread_std = spread.ewm(halflife=21).std().iloc[-1]
        
        # Z-score
        z_score = (spread.iloc[-1] - spread_mean) / spread_std
        
        return z_score
    
    def adjust_no_trade_band(self, pair_info, vol_increase=0.0, cost_bps=1.0):
        """Ajusta banda de no-transacción según volatilidad y costos"""
        vol_adj = self.no_trade_base_width + 0.1 * (vol_increase / 25.0)
        cost_adj = vol_adj + 0.05 * cost_bps
        return cost_adj
    
    def generate_signal(self, data, pair_info, regime, current_position=0, conv_probability=0.5):
        """Genera señal de trading para un par"""
        # Calcular z-score
        z_score = self.calculate_z_score(data, pair_info)
        
        if z_score is None:
            return {'signal': 0, 'z_score': None, 'strength': 0}
        
        # Ajustar umbrales basados en probabilidad de convergencia
        prob_adj = (conv_probability - 0.5) * 0.3
        
        # Obtener umbrales para el régimen
        entry_long = self.entry_thresholds[regime]['long'] + prob_adj
        entry_short = self.entry_thresholds[regime]['short'] - prob_adj
        exit_long = self.exit_thresholds[regime]['long'] - prob_adj
        exit_short = self.exit_thresholds[regime]['short'] + prob_adj
        
        # Ajustar bandas de no-transacción
        vol_increase = 0.0  # En implementación real, se calcularía
        cost_bps = 1.0
        no_trade_band = self.adjust_no_trade_band(pair_info, vol_increase, cost_bps)
        exit_no_trade_band = no_trade_band * 1.2
        
        # Determinar señal
        signal = 0
        strength = 0
        
        if current_position == 0:  # Sin posición
            if z_score < entry_long:
                signal = 1  # Comprar
                strength = min(1.0, (entry_long - z_score) / abs(entry_long * 0.5))
            elif z_score > entry_short:
                signal = -1  # Vender
                strength = min(1.0, (z_score - entry_short) / abs(entry_short * 0.5))
        elif current_position == 1:  # Posición larga
            if z_score > exit_long:
                signal = -1  # Cerrar
                strength = min(1.0, (z_score - exit_long) / abs(exit_long * 0.5))
        elif current_position == -1:  # Posición corta
            if z_score < exit_short:
                signal = 1  # Cerrar
                strength = min(1.0, (exit_short - z_score) / abs(exit_short * 0.5))
        
        # Aplicar bandas de no-transacción
        if abs(signal) > 0 and abs(z_score) < no_trade_band and current_position == 0:
            signal = 0
            strength = 0
        elif abs(signal) > 0 and abs(z_score) < exit_no_trade_band and current_position != 0:
            signal = 0
            strength = 0
        
        return {
            'signal': signal,
            'z_score': z_score,
            'strength': strength
        }

# Gestión de Posiciones y Riesgo
class PositionManager:
    """Gestiona posiciones y riesgo para la estrategia"""
    
    def __init__(self):
        # Límites de posición por régimen
        self.position_limits = {
            1: 0.03,  # 3% máximo por par en Régimen 1
            2: 0.025, # 2.5% en Régimen 2
            3: 0.02   # 2% en Régimen 3
        }
        
        # Factores ATR para stop-loss
        self.atr_factors = {
            1: 3.0,
            2: 2.5,
            3: 2.0
        }
        
        # Límites de concentración por sector
        self.sector_limits = {
            1: 0.20,  # 20% por sector en Régimen 1
            2: 0.175, # 17.5% en Régimen 2
            3: 0.15   # 15% en Régimen 3
        }
        
        # Volatilidad objetivo por régimen
        self.vol_targets = {
            1: 0.09,  # 9% anualizada
            2: 0.08,  # 8% anualizada
            3: 0.06   # 6% anualizada
        }
        
        # Circuit breakers
        self.circuit_breakers = {
            'level1': {'reduction': 0.25, 'vix_percentile': 80, 'vix_increase': 0.10},
            'level2': {'reduction': 0.50, 'vix_percentile': 90, 'correlation': 0.70},
            'level3': {'reduction': 0.75, 'vix_percentile': 95, 'correlation': 0.80}
        }
        
        self.current_positions = {}
    
    def calculate_position_size(self, pair_info, signal_strength, regime, 
                               conv_probability, current_volatility):
        """Calcula tamaño óptimo de posición"""
        if current_volatility <= 0:
            return 0
            
        # Volatilidad inversa normalizada
        vol_sizing = min(1.0 / current_volatility, 5.0) / 5.0
        
        # Ajuste por convicción
        conv_multiplier = 0.5 + conv_probability
        
        # Ajuste por liquidez
        liquidity = pair_info['liquidity']
        liquidity_factor = min(1.0, liquidity / 50e6)
        
        # Tamaño base
        base_size = vol_sizing * conv_multiplier * liquidity_factor
        
        # Aplicar límite por régimen
        max_size = self.position_limits[regime]
        position_size = min(base_size * max_size, max_size)
        
        # Aplicar fuerza de señal
        position_size *= signal_strength
        
        return position_size
    
    def calculate_stop_loss(self, data, pair_info, regime, entry_price, position_type):
        """Calcula nivel de stop-loss"""
        ticker1 = pair_info['ticker1']
        ticker2 = pair_info['ticker2']
        hedge_ratio = pair_info['hedge_ratio']
        
        # Calcular spread
        prices = data['prices'].iloc[-20:][[ticker1, ticker2]]
        spread = prices[ticker1] + hedge_ratio * prices[ticker2]
        
        # Calcular ATR del spread
        spread_high = spread.rolling(window=2).max()
        spread_low = spread.rolling(window=2).min()
        tr = spread_high - spread_low
        atr = tr.rolling(window=14).mean().iloc[-1]
        
        if pd.isna(atr) or atr == 0:
            atr = spread.std()
            
        # Aplicar factor según régimen
        atr_factor = self.atr_factors[regime]
        stop_distance = atr * atr_factor
        
        # Calcular stop-loss
        if position_type == 1:  # Long
            stop_loss = entry_price - stop_distance
        else:  # Short
            stop_loss = entry_price + stop_distance
            
        return stop_loss
    
    def calculate_time_stop(self, pair_info):
        """Calcula stop temporal basado en half-life"""
        half_life = pair_info['half_life']
        time_stop = int(2.5 * half_life)
        return max(5, time_stop)
    
    def check_circuit_breakers(self, data, sector_map, vix_data=None):
        """Comprueba activación de circuit breakers"""
        # Crear proxy de VIX si no se proporciona
        if vix_data is None:
            returns = data['returns'].mean(axis=1)
            vix_proxy = returns.rolling(window=22).std() * np.sqrt(252) * 100
            vix_data = vix_proxy
        
        # Percentiles históricos
        vix_80 = vix_data.quantile(0.80)
        vix_90 = vix_data.quantile(0.90)
        vix_95 = vix_data.quantile(0.95)
        
        current_vix = vix_data.iloc[-1]
        vix_1d_change = current_vix / vix_data.iloc[-2] - 1 if len(vix_data) > 1 else 0
        
        # Correlación promedio
        returns = data['returns'].iloc[-20:]
        corr_matrix = returns.corr()
        avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        
        # Verificar niveles
        if current_vix > vix_95 and avg_correlation > self.circuit_breakers['level3']['correlation']:
            return 1.0 - self.circuit_breakers['level3']['reduction']
        elif current_vix > vix_90 or avg_correlation > self.circuit_breakers['level2']['correlation']:
            return 1.0 - self.circuit_breakers['level2']['reduction']
        elif current_vix > vix_80 and vix_1d_change > self.circuit_breakers['level1']['vix_increase']:
            return 1.0 - self.circuit_breakers['level1']['reduction']
        else:
            return 1.0
    
    def check_sector_concentration(self, new_positions, sector_map, regime):
        """Verifica límites de concentración por sector"""
        # Calcular exposición por sector
        sector_exposure = {}
        
        for pair_id, position in new_positions.items():
            ticker1, ticker2 = pair_id.split('_')
            sector1 = sector_map.get(ticker1)
            sector2 = sector_map.get(ticker2)
            
            size = abs(position['size'])
            
            if sector1:
                if sector1 not in sector_exposure:
                    sector_exposure[sector1] = 0
                sector_exposure[sector1] += size / 2
                
            if sector2:
                if sector2 not in sector_exposure:
                    sector_exposure[sector2] = 0
                sector_exposure[sector2] += size / 2
        
        # Verificar límites
        sector_limit = self.sector_limits[regime]
        adjustment_needed = False
        
        for sector, exposure in sector_exposure.items():
            if exposure > sector_limit:
                adjustment_needed = True
                break
        
        if not adjustment_needed:
            return new_positions
        
        # Calcular factores de escala
        sector_scales = {}
        for sector, exposure in sector_exposure.items():
            if exposure > sector_limit:
                sector_scales[sector] = sector_limit / exposure
            else:
                sector_scales[sector] = 1.0
        
        # Ajustar posiciones
        adjusted_positions = {}
        for pair_id, position in new_positions.items():
            ticker1, ticker2 = pair_id.split('_')
            sector1 = sector_map.get(ticker1)
            sector2 = sector_map.get(ticker2)
            
            scale1 = sector_scales.get(sector1, 1.0) if sector1 else 1.0
            scale2 = sector_scales.get(sector2, 1.0) if sector2 else 1.0
            scale = min(scale1, scale2)
            
            adjusted_position = position.copy()
            adjusted_position['size'] *= scale
            adjusted_positions[pair_id] = adjusted_position
        
        return adjusted_positions
    
    def optimize_portfolio(self, data, pairs_with_signals, regime, sector_map):
        """Optimiza el portfolio basado en señales y restricciones"""
        new_positions = {}
        
        # Calcular volatilidad de cada par
        pair_volatilities = {}
        for pair_info in pairs_with_signals:
            ticker1 = pair_info['ticker1']
            ticker2 = pair_info['ticker2']
            hedge_ratio = pair_info['hedge_ratio']
            
            prices = data['prices'].iloc[-60:][[ticker1, ticker2]]
            if len(prices) < 30 or prices[ticker1].isna().any() or prices[ticker2].isna().any():
                continue
                
            spread = prices[ticker1] + hedge_ratio * prices[ticker2]
            volatility = spread.pct_change().std() * np.sqrt(252)
            
            pair_id = f"{ticker1}_{ticker2}"
            pair_volatilities[pair_id] = volatility
        
        # Ordenar pares por fuerza de señal * probabilidad
        sorted_pairs = sorted(
            pairs_with_signals, 
            key=lambda x: abs(x['signal']['signal']) * x['signal']['strength'] * x['conv_probability'],
            reverse=True
        )
        
        # Asignar posiciones
        total_risk = 0
        
        for pair_info in sorted_pairs:
            ticker1 = pair_info['ticker1']
            ticker2 = pair_info['ticker2']
            hedge_ratio = pair_info['hedge_ratio']
            signal = pair_info['signal']['signal']
            strength = pair_info['signal']['strength']
            z_score = pair_info['signal']['z_score']
            conv_probability = pair_info['conv_probability']
            
            pair_id = f"{ticker1}_{ticker2}"
            
            if signal == 0 or pair_id not in pair_volatilities:
                continue
            
            # Calcular tamaño
            position_size = self.calculate_position_size(
                pair_info, strength, regime, conv_probability, pair_volatilities[pair_id]
            )
            
            # Ajustar signo
            signed_position = position_size * np.sign(signal)
            
            # Simular precio de entrada
            entry_price = 1.0  # Normalizado
            
            # Calcular stops
            stop_loss = self.calculate_stop_loss(
                data, pair_info, regime, entry_price, np.sign(signal)
            )
            
            time_stop = self.calculate_time_stop(pair_info)
            
            # Guardar posición
            new_positions[pair_id] = {
                'ticker1': ticker1,
                'ticker2': ticker2,
                'hedge_ratio': hedge_ratio,
                'size': signed_position,
                'signal': signal,
                'strength': strength,
                'z_score': z_score,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'time_stop': time_stop,
                'entry_date': data['prices'].index[-1],
                'days_held': 0
            }
            
            # Acumular riesgo
            total_risk += abs(signed_position) * pair_volatilities[pair_id]
        
        # Verificar concentración sectorial
        adjusted_positions = self.check_sector_concentration(new_positions, sector_map, regime)
        
        # Escalar a volatilidad objetivo
        target_vol = self.vol_targets[regime]
        
        if total_risk > 0:
            vol_scale = target_vol / total_risk
            for pair_id in adjusted_positions:
                adjusted_positions[pair_id]['size'] *= vol_scale
        
        # Aplicar circuit breakers
        circuit_breaker_scale = self.check_circuit_breakers(data, sector_map)
        
        for pair_id in adjusted_positions:
            adjusted_positions[pair_id]['size'] *= circuit_breaker_scale
        
        self.current_positions = adjusted_positions
        return adjusted_positions

# Estrategia completa
class StatArbStrategy:
    """Implementación completa de la estrategia de arbitraje estadístico"""
    
    def __init__(self):
        self.regime_detector = RegimeDetector()
        self.pair_selector = PairSelector()
        self.convergence_predictor = ConvergencePredictor()
        self.signal_generator = SignalGenerator()
        self.position_manager = PositionManager()
        
        self.current_regime = None
        self.selected_pairs = []
        self.current_positions = {}
        self.positions_history = []
        self.regime_history = []
        self.equity_curve = None
        
        # Métricas de rendimiento
        self.metrics = {
            'sharpe_ratio': None,
            'sortino_ratio': None,
            'max_drawdown': None,
            'annual_return': None,
            'volatility': None,
            'win_rate': None,
            'avg_trade_duration': None
        }
    
    def initialize(self, data, sector_map, subsector_map):
        """Inicializa la estrategia con datos históricos"""
        print("Detectando régimen inicial...")
        self.current_regime = self.regime_detector.fit_predict(data, sector_map)
        print(f"Régimen inicial: {self.current_regime}")
        
        print("Generando pares candidatos...")
        candidate_pairs = self.pair_selector.generate_candidate_pairs(
            data, sector_map, subsector_map)
        print(f"Pares candidatos: {len(candidate_pairs)}")
        
        print("Seleccionando pares óptimos...")
        self.selected_pairs = self.pair_selector.select_pairs(
            data, self.current_regime)
        print(f"Pares seleccionados: {len(self.selected_pairs)}")
        
        print("Entrenando modelos predictivos...")
        self.convergence_predictor.train(
            data, self.selected_pairs, self.current_regime)
    
    def update(self, data, sector_map, subsector_map, current_date=None):
        """Actualiza la estrategia con nuevos datos"""
        if current_date is None:
            current_date = data['prices'].index[-1]
        
        # Verificar recalibración (lunes o cambio de régimen)
        is_monday = current_date.weekday() == 0
        prev_regime = self.current_regime
        
        # Actualizar régimen
        self.current_regime = self.regime_detector.fit_predict(
            data, sector_map, current_date)
        
        # Registrar historia
        self.regime_history.append({
            'date': current_date,
            'regime': self.current_regime
        })
        
        # Recalibrar si es necesario
        needs_recalibration = is_monday or (prev_regime != self.current_regime)
        
        if needs_recalibration:
            candidate_pairs = self.pair_selector.generate_candidate_pairs(
                data, sector_map, subsector_map)
            
            self.selected_pairs = self.pair_selector.select_pairs(
                data, self.current_regime, candidate_pairs)
            
            self.convergence_predictor.train(
                data, self.selected_pairs, self.current_regime)
        
        # Generar señales
        pairs_with_signals = []
        
        for pair_info in self.selected_pairs:
            ticker1 = pair_info['ticker1']
            ticker2 = pair_info['ticker2']
            pair_id = f"{ticker1}_{ticker2}"
            
            # Determinar posición actual
            current_position = 0
            if pair_id in self.current_positions:
                current_position = np.sign(self.current_positions[pair_id]['size'])
            
            # Predecir convergencia
            conv_probability = self.convergence_predictor.predict_convergence(
                data, pair_info, self.current_regime)
            
            # Generar señal
            signal = self.signal_generator.generate_signal(
                data, pair_info, self.current_regime, current_position, conv_probability)
            
            if signal['z_score'] is not None:
                pair_info['signal'] = signal
                pair_info['conv_probability'] = conv_probability
                pairs_with_signals.append(pair_info)
        
        # Optimizar portfolio
        updated_positions = self.position_manager.optimize_portfolio(
            data, pairs_with_signals, self.current_regime, sector_map)
        
        # Actualizar posiciones
        self.current_positions = updated_positions
        
        # Registrar historia
        position_snapshot = {
            'date': current_date,
            'regime': self.current_regime,
            'positions': {k: v.copy() for k, v in updated_positions.items()}
        }
        self.positions_history.append(position_snapshot)
        
        return updated_positions
    
    def backtest(self, data, sector_map, subsector_map, start_date=None, end_date=None):
        """Realiza backtest de la estrategia"""
        print("Inicializando backtest...")
        prices = data['prices']
        
        if start_date is None:
            # Usar primeros 5 años para entrenamiento inicial
            training_days = 252 * 5
            if len(prices) > training_days:
                start_date = prices.index[training_days]
            else:
                start_date = prices.index[len(prices) // 2]
        
        if end_date is None:
            end_date = prices.index[-1]
        
        # Filtrar datos para backtest
        backtest_dates = prices.loc[start_date:end_date].index
        
        # Inicializar equity curve
        equity_curve = pd.DataFrame(index=backtest_dates, columns=['equity', 'returns', 'drawdown'])
        equity_curve['equity'] = 1.0
        
        # Inicializar con datos hasta start_date
        training_data = {
            'prices': prices.loc[:start_date],
            'returns': data['returns'].loc[:start_date],
            'volumes': data['volumes'].loc[:start_date],
            'realized_vol': data['realized_vol'].loc[:start_date],
            'relative_volume': data['relative_volume'].loc[:start_date],
            'adv': data['adv'].loc[:start_date]
        }
        
        self.initialize(training_data, sector_map, subsector_map)
        
        # Simular trading
        print(f"Ejecutando backtest desde {start_date} hasta {end_date}...")
        
        previous_positions = {}
        trades_log = []
        
        for i, current_date in enumerate(tqdm(backtest_dates)):
            # Datos hasta fecha actual (sin look-ahead bias)
            current_data = {
                'prices': prices.loc[:current_date],
                'returns': data['returns'].loc[:current_date],
                'volumes': data['volumes'].loc[:current_date],
                'realized_vol': data['realized_vol'].loc[:current_date],
                'relative_volume': data['relative_volume'].loc[:current_date],
                'adv': data['adv'].loc[:current_date]
            }
            
            # Actualizar estrategia
            current_positions = self.update(current_data, sector_map, subsector_map, current_date)
            
            # Calcular P&L diario
            daily_pnl = 0
            
            # P&L de posiciones cerradas
            closed_positions = set(previous_positions.keys()) - set(current_positions.keys())
            
            for pair_id in closed_positions:
                old_position = previous_positions[pair_id]
                
                # Simular precio de cierre
                ticker1 = old_position['ticker1']
                ticker2 = old_position['ticker2']
                hedge_ratio = old_position['hedge_ratio']
                
                # Calcular retorno del spread
                if ticker1 in prices.columns and ticker2 in prices.columns:
                    old_spread = prices.loc[old_position['entry_date'], ticker1] + \
                               hedge_ratio * prices.loc[old_position['entry_date'], ticker2]
                    new_spread = prices.loc[current_date, ticker1] + \
                               hedge_ratio * prices.loc[current_date, ticker2]
                    
                    # Retorno según dirección
                    if old_position['signal'] > 0:  # Long
                        trade_return = (new_spread - old_spread) / abs(old_spread)
                    else:  # Short
                        trade_return = (old_spread - new_spread) / abs(old_spread)
                    
                    position_pnl = old_position['size'] * trade_return
                    daily_pnl += position_pnl
                    
                    # Registrar trade
                    trades_log.append({
                        'pair_id': pair_id,
                        'entry_date': old_position['entry_date'],
                        'exit_date': current_date,
                        'days_held': (current_date - old_position['entry_date']).days,
                        'entry_signal': old_position['signal'],
                        'entry_z_score': old_position['z_score'],
                        'position_size': old_position['size'],
                        'pnl': position_pnl,
                        'return': trade_return
                    })
            
            # P&L de posiciones actualizadas
            common_positions = set(previous_positions.keys()) & set(current_positions.keys())
            
            for pair_id in common_positions:
                old_position = previous_positions[pair_id]
                new_position = current_positions[pair_id]
                
                # Si el tamaño cambió, calcular P&L para la parte cerrada
                if abs(old_position['size']) != abs(new_position['size']):
                    size_diff = old_position['size'] - new_position['size']
                    
                    ticker1 = old_position['ticker1']
                    ticker2 = old_position['ticker2']
                    hedge_ratio = old_position['hedge_ratio']
                    
                    if ticker1 in prices.columns and ticker2 in prices.columns:
                        old_spread = prices.loc[old_position['entry_date'], ticker1] + \
                                   hedge_ratio * prices.loc[old_position['entry_date'], ticker2]
                        new_spread = prices.loc[current_date, ticker1] + \
                                   hedge_ratio * prices.loc[current_date, ticker2]
                        
                        if old_position['signal'] > 0:
                            trade_return = (new_spread - old_spread) / abs(old_spread)
                        else:
                            trade_return = (old_spread - new_spread) / abs(old_spread)
                        
                        position_pnl = size_diff * trade_return
                        daily_pnl += position_pnl
                        
                        # Registrar trade parcial
                        trades_log.append({
                            'pair_id': pair_id,
                            'entry_date': old_position['entry_date'],
                            'exit_date': current_date,
                            'days_held': (current_date - old_position['entry_date']).days,
                            'entry_signal': old_position['signal'],
                            'entry_z_score': old_position['z_score'],
                            'position_size': size_diff,
                            'pnl': position_pnl,
                            'return': trade_return,
                            'partial': True
                        })
            
            # Actualizar equity curve
            if i > 0:
                equity_curve.loc[current_date, 'returns'] = daily_pnl
                equity_curve.loc[current_date, 'equity'] = equity_curve.iloc[i-1]['equity'] * (1 + daily_pnl)
            else:
                equity_curve.loc[current_date, 'returns'] = 0
            
            # Actualizar posiciones anteriores
            previous_positions = {k: v.copy() for k, v in current_positions.items()}
        
        # Calcular drawdown
        equity = equity_curve['equity']
        drawdown = 1 - equity / equity.cummax()
        equity_curve['drawdown'] = drawdown
        
        # Calcular métricas
        self.calculate_performance_metrics(equity_curve, trades_log)
        
        # Guardar resultados
        self.equity_curve = equity_curve
        self.trades_log = pd.DataFrame(trades_log)
        
        return equity_curve
    
    def calculate_performance_metrics(self, equity_curve, trades_log):
        """Calcula métricas de rendimiento"""
        # Convertir trades_log a DataFrame si es lista
        if isinstance(trades_log, list):
            trades_df = pd.DataFrame(trades_log)
        else:
            trades_df = trades_log
        
        # Retornos diarios
        daily_returns = equity_curve['returns'].dropna()
        
        # Sharpe Ratio (anualizado)
        sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
        
        # Sortino Ratio (anualizado)
        negative_returns = daily_returns[daily_returns < 0]
        sortino = np.sqrt(252) * daily_returns.mean() / negative_returns.std() if len(negative_returns) > 0 and negative_returns.std() > 0 else 0
        
        # Maximum Drawdown
        max_drawdown = equity_curve['drawdown'].max()
        
        # Annualized Return
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        annual_return = (equity_curve['equity'].iloc[-1] / equity_curve['equity'].iloc[0]) ** (365 / max(days, 1)) - 1
        
        # Annualized Volatility
        annual_vol = daily_returns.std() * np.sqrt(252)
        
        # Win Rate
        if len(trades_df) > 0:
            win_rate = (trades_df['pnl'] > 0).mean()
            avg_trade_duration = trades_df['days_held'].mean()
        else:
            win_rate = 0
            avg_trade_duration = 0
        
        # Guardar métricas
        self.metrics = {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'annual_return': annual_return,
            'volatility': annual_vol,
            'win_rate': win_rate,
            'avg_trade_duration': avg_trade_duration
        }
        
        return self.metrics
    
    def walk_forward_test(self, data, sector_map, subsector_map, 
                         training_years=5, validation_months=12, test_months=6):
        """Realiza validación walk-forward"""
        prices = data['prices']
        dates = prices.index
        
        # Convertir a días
        training_days = training_years * 252
        validation_days = validation_months * 21
        test_days = test_months * 21
        
        if len(dates) < training_days + validation_days + test_days:
            print("Datos insuficientes para walk-forward testing.")
            return None
        
        # Inicializar resultados
        results = {
            'windows': [],
            'equity_curves': [],
            'metrics': []
        }
        
        # Definir ventanas
        start_idx = training_days
        while start_idx + validation_days + test_days <= len(dates):
            # Definir fechas
            train_start = dates[max(0, start_idx - training_days)]
            train_end = dates[start_idx]
            validation_end = dates[start_idx + validation_days]
            test_end = dates[min(len(dates) - 1, start_idx + validation_days + test_days)]
            
            print(f"\nWalk-forward ventana {len(results['windows'])+1}:")
            print(f"Entrenamiento: {train_start} a {train_end}")
            print(f"Validación: {train_end} a {validation_end}")
            print(f"Prueba: {validation_end} a {test_end}")
            
            # Nueva instancia de estrategia
            strategy = StatArbStrategy()
            
            # Datos de entrenamiento
            train_data = {
                'prices': prices.loc[:train_end],
                'returns': data['returns'].loc[:train_end],
                'volumes': data['volumes'].loc[:train_end],
                'realized_vol': data['realized_vol'].loc[:train_end],
                'relative_volume': data['relative_volume'].loc[:train_end],
                'adv': data['adv'].loc[:train_end]
            }
            
            strategy.initialize(train_data, sector_map, subsector_map)
            
            # Ejecutar backtest
            test_data = {
                'prices': prices.loc[:test_end],
                'returns': data['returns'].loc[:test_end],
                'volumes': data['volumes'].loc[:test_end],
                'realized_vol': data['realized_vol'].loc[:test_end],
                'relative_volume': data['relative_volume'].loc[:test_end],
                'adv': data['adv'].loc[:test_end]
            }
            
            equity_curve = strategy.backtest(test_data, sector_map, subsector_map, 
                                           validation_end, test_end)
            
            # Guardar resultados
            results['windows'].append({
                'train_start': train_start,
                'train_end': train_end,
                'validation_end': validation_end,
                'test_end': test_end
            })
            results['equity_curves'].append(equity_curve)
            results['metrics'].append(strategy.metrics)
            
            # Avanzar
            start_idx += test_days
        
        # Métricas agregadas
        if results['metrics']:
            avg_metrics = {}
            for key in results['metrics'][0].keys():
                avg_metrics[key] = np.mean([m[key] for m in results['metrics']])
            
            results['avg_metrics'] = avg_metrics
        
        return results
    
    def plot_equity_curve(self, filename='equity_curve.png'):
        """Genera gráfico de curva de equity"""
        if self.equity_curve is None:
            print("No hay curva de equity disponible.")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Equity
        plt.subplot(2, 1, 1)
        plt.plot(self.equity_curve.index, self.equity_curve['equity'], label='Equity')
        plt.title('Curva de Equity')
        plt.ylabel('Equity')
        plt.grid(True)
        plt.legend()
        
        # Drawdown
        plt.subplot(2, 1, 2)
        plt.fill_between(self.equity_curve.index, self.equity_curve['drawdown'], 
                       alpha=0.3, color='red')
        plt.plot(self.equity_curve.index, self.equity_curve['drawdown'], 
               color='red', label='Drawdown')
        plt.title('Drawdown')
        plt.ylabel('Drawdown')
        plt.ylim(0, max(self.equity_curve['drawdown']) * 1.1)
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'./artifacts/results/figures/{filename}')
        plt.close()
    
    def plot_regime_distribution(self, filename='regime_distribution.png'):
        """Grafica distribución de regímenes"""
        if not self.regime_history:
            print("No hay historia de regímenes disponible.")
            return
        
        # Crear DataFrame
        regime_df = pd.DataFrame(self.regime_history)
        regime_df.set_index('date', inplace=True)
        
        plt.figure(figsize=(12, 6))
        
        # Distribución
        plt.subplot(1, 2, 1)
        regime_counts = regime_df['regime'].value_counts()
        labels = [f'Régimen {i}' for i in regime_counts.index]
        plt.pie(regime_counts, labels=labels, autopct='%1.1f%%', 
              startangle=90, shadow=True)
        plt.title('Distribución de Regímenes')
        
        # Evolución
        plt.subplot(1, 2, 2)
        plt.plot(regime_df.index, regime_df['regime'], marker='o', markersize=3)
        plt.title('Evolución de Regímenes')
        plt.ylabel('Régimen')
        plt.yticks([1, 2, 3])
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'./artifacts/results/figures/{filename}')
        plt.close()
    
    def plot_pair_zscore(self, pair_id, data, filename=None):
        """Grafica z-score de un par específico"""
        if pair_id not in self.current_positions:
            print(f"El par {pair_id} no está en posiciones actuales.")
            return
        
        position = self.current_positions[pair_id]
        ticker1 = position['ticker1']
        ticker2 = position['ticker2']
        hedge_ratio = position['hedge_ratio']
        
        # Calcular spread y z-score
        prices = data['prices'].iloc[-252:][[ticker1, ticker2]]
        spread = prices[ticker1] + hedge_ratio * prices[ticker2]
        
        spread_mean = spread.rolling(window=60).mean()
        spread_std = spread.rolling(window=60).std()
        z_score = (spread - spread_mean) / spread_std
        
        # Crear gráfico
        plt.figure(figsize=(12, 8))
        
        # Spread
        plt.subplot(2, 1, 1)
        plt.plot(spread.index, spread, label='Spread')
        plt.plot(spread_mean.index, spread_mean, label='Media Móvil', linestyle='--')
        plt.fill_between(spread.index, 
                       spread_mean + 2*spread_std, 
                       spread_mean - 2*spread_std, 
                       alpha=0.2, 
                       label='±2 Desv. Est.')
        plt.title(f'Spread: {ticker1} - {ticker2} (HR: {hedge_ratio:.4f})')
        plt.ylabel('Spread')
        plt.grid(True)
        plt.legend()
        
        # Z-score
        plt.subplot(2, 1, 2)
        plt.plot(z_score.index, z_score, label='Z-Score')
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        plt.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='±2')
        plt.axhline(y=-2, color='red', linestyle='--', alpha=0.5)
        plt.axhline(y=1, color='orange', linestyle='--', alpha=0.5, label='±1')
        plt.axhline(y=-1, color='orange', linestyle='--', alpha=0.5)
        plt.title('Z-Score del Spread')
        plt.ylabel('Z-Score')
        plt.ylim(-4, 4)
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(f'./artifacts/results/figures/{filename}')
        else:
            plt.savefig(f'./artifacts/results/figures/pair_{ticker1}_{ticker2}_zscore.png')
        
        plt.close()
    
    def save_performance_summary(self, filename='performance_summary.csv'):
        """Guarda métricas en CSV"""
        metrics_df = pd.DataFrame([self.metrics])
        metrics_df.to_csv(f'./artifacts/results/data/{filename}', index=False)
    
    def save_trade_log(self, filename='trade_log.csv'):
        """Guarda registro de operaciones en CSV"""
        if hasattr(self, 'trades_log'):
            self.trades_log.to_csv(f'./artifacts/results/data/{filename}', index=False)

# Función principal
def main():
    """Función principal para ejecutar la estrategia"""
    print("Iniciando implementación de estrategia de arbitraje estadístico...")
    
    # Período para datos
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=252*6)).strftime('%Y-%m-%d')
    
    # Obtener tickers del S&P 500
    print("Obteniendo tickers del S&P 500...")
    tickers, sector_map, subsector_map = get_sp500_tickers()
    
    # Para pruebas, usar subconjunto de tickers
    tickers = tickers[:100]  # Usar primeros 100 para demostración
    
    # Descargar datos
    print(f"Descargando datos desde {start_date} hasta {end_date}...")
    data_dict = get_historical_data(tickers, start_date, end_date)
    
    # Preprocesar
    print("Procesando datos...")
    processed_data = preprocess_data(data_dict)
    
    # Crear estrategia
    strategy = StatArbStrategy()
    
    try:
        # Inicializar
        strategy.initialize(processed_data, sector_map, subsector_map)
        
        # Ejecutar backtest
        print("Ejecutando backtest...")
        backtest_start = (datetime.now() - timedelta(days=252*2)).strftime('%Y-%m-%d')
        equity_curve = strategy.backtest(processed_data, sector_map, subsector_map, backtest_start)
        
        # Validación walk-forward
        print("Ejecutando validación walk-forward...")
        wf_results = strategy.walk_forward_test(processed_data, sector_map, subsector_map)
        
        # Generar gráficos
        print("Generando visualizaciones...")
        strategy.plot_equity_curve()
        strategy.plot_regime_distribution()
        
        # Graficar z-scores de pares activos
        if strategy.current_positions:
            print("Graficando z-scores de pares seleccionados...")
            for i, pair_id in enumerate(list(strategy.current_positions.keys())[:5]):
                strategy.plot_pair_zscore(pair_id, processed_data)
        
        # Guardar métricas
        print("Guardando métricas de rendimiento...")
        strategy.save_performance_summary()
        strategy.save_trade_log()
        
        # Guardar resultados de walk-forward
        if wf_results:
            print("Guardando resultados de walk-forward...")
            wf_metrics = pd.DataFrame(wf_results['metrics'])
            wf_metrics.to_csv('./artifacts/results/data/walkforward_metrics.csv', index=False)
            
            # Equity curve combinada
            if wf_results['equity_curves']:
                wf_equity = pd.concat([curve[['equity']] for curve in wf_results['equity_curves']], axis=1)
                wf_equity.columns = [f'window_{i}' for i in range(len(wf_results['equity_curves']))]
                wf_equity.to_csv('./artifacts/results/data/walkforward_equity.csv')
        
        print("Implementación completada con éxito.")
        return strategy
        
    except Exception as e:
        logging.error(f"Error en ejecución: {str(e)}")
        with open('./artifacts/errors.txt', 'a') as f:
            f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR EN EJECUCIÓN:\n")
            traceback.print_exc(file=f)
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    try:
        strategy = main()
    except Exception as e:
        logging.error(f"Error en ejecución principal: {str(e)}")
        with open('./artifacts/errors.txt', 'a') as f:
            f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR EN EJECUCIÓN PRINCIPAL:\n")
            traceback.print_exc(file=f)
        print(f"Error: {str(e)}")
