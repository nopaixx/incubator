import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
import pytz
import schedule
from adaptive_multifactor_strategy import AdaptiveMultifactorStrategy

# Configurar logging
logging.basicConfig(
    filename='./artifacts/alpaca_trading.log',
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class AlpacaStrategyAdapter:
    def __init__(self, lookback_years=3, rebalance_freq=21, use_all_assets=True):
        """
        Inicializa el adaptador para Alpaca Paper Trading.
        
        Parámetros:
        -----------
        lookback_years : int
            Años de datos históricos a utilizar
        rebalance_freq : int
            Frecuencia de rebalanceo en días de trading
        use_all_assets : bool
            Si se deben usar todos los activos disponibles
        """
        # Crear directorio para datos persistentes
        os.makedirs('./artifacts/data', exist_ok=True)
        # Cargar variables de entorno
        load_dotenv()
        
        # Configurar credenciales de Alpaca
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.api_secret = os.getenv('ALPACA_API_SECRET')
        self.base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("No se encontraron credenciales de Alpaca en el archivo .env")
        
        # Inicializar API de Alpaca
        self.api = tradeapi.REST(
            self.api_key,
            self.api_secret,
            self.base_url,
            api_version='v2'
        )
        
        # Guardar parámetros
        self.lookback_years = lookback_years
        self.rebalance_freq = rebalance_freq
        self.use_all_assets = use_all_assets
        
        # Zona horaria de mercado de EE.UU.
        self.eastern_tz = pytz.timezone('US/Eastern')
        
        # Verificar conexión con Alpaca
        try:
            account = self.api.get_account()
            logging.info(f"Conexión exitosa con Alpaca. Saldo: ${float(account.equity):.2f}")
            print(f"Conexión exitosa con Alpaca. Saldo: ${float(account.equity):.2f}")
        except Exception as e:
            logging.error(f"Error al conectar con Alpaca: {str(e)}")
            raise
        
        # Inicializar estrategia base (se configurará al obtener datos)
        self.strategy = None
        self.last_rebalance_date = None
        self.symbols = None
    
    def get_tradable_assets(self, min_price=5, min_market_cap=500000000):
        """
        Obtiene lista de activos negociables en Alpaca que cumplan criterios mínimos.
        """
        try:
            logging.info("Obteniendo lista de activos negociables en Alpaca...")
            assets = self.api.list_assets(status='active', asset_class='us_equity')
            # Filtrar activos negociables
            tradable_assets = [
                asset for asset in assets 
                if asset.tradable and asset.marginable and asset.shortable and asset.easy_to_borrow
            ]
            

            # Obtener precios actuales
            symbols = [asset.symbol for asset in tradable_assets]
            
            # Limitar a 100 símbolos por solicitud (limitación de Alpaca)
            # Limitar a 100 símbolos por solicitud (limitación de Alpaca)
            prices = {}
            for i in range(0, len(symbols), 100):
                batch = symbols[i:i+100]                
                
                # Verificar que todos los símbolos sean válidos antes de hacer la solicitud
                valid_batch = [symbol for symbol in batch if self.api.get_asset(symbol).status == 'active']
                
                try:
                    if valid_batch:
                        current_prices = self.api.get_latest_trade(valid_batch)
                        for symbol, trade in current_prices.items():
                            prices[symbol] = trade.price
                    else:
                        logging.warning("No symbols in the batch are valid.")
                except Exception as e:
                    print("error", e)
                    logging.warning(f"Error al obtener precios para lote de símbolos: {str(e)}")
                    
            # Filtrar por precio mínimo

            filtered_symbols = [
                symbol for symbol in symbols 
                if symbol in prices and prices[symbol] >= min_price
            ]
            
            logging.info(f"Se encontraron {len(filtered_symbols)} activos negociables que cumplen los criterios")
            return filtered_symbols
            
        except Exception as e:
            logging.error(f"Error al obtener activos negociables: {str(e)}")
            # Fallback a S&P 500 como universo
            logging.info("Usando símbolos de S&P 500 como fallback")
            try:
                import yfinance as yf
                sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
                return sp500['Symbol'].tolist()
            except:
                # Si todo falla, usar una lista de símbolos grandes conocidos
                return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'PG']
    
    def get_historical_data(self, symbols, start_date, end_date=None):
        """
        Obtiene datos históricos de Alpaca para los símbolos especificados.
        Optimizado para grandes cantidades de símbolos.
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        logging.info(f"Obteniendo datos históricos desde {start_date} hasta {end_date}")
        
        # Crear DataFrames vacíos para precios y volúmenes
        all_prices = pd.DataFrame()
        all_volumes = pd.DataFrame()
        
        # Procesar en lotes de 100 símbolos (limitación de Alpaca)
        for i in range(0, len(symbols), 100):
            batch_symbols = symbols[i:i+100]
            logging.info(f"Obteniendo datos para lote {i//100 + 1}/{(len(symbols)-1)//100 + 1} ({len(batch_symbols)} símbolos)")
            
            try:
                # Obtener barras diarias
                barset = self.api.get_bars(
                    batch_symbols,
                    '1D',
                    start=start_date,
                    end=end_date,
                    adjustment='all'  # Ajustar por splits y dividendos
                ).df
                
                if barset.empty:
                    continue
                
                # Procesar datos para este lote
                for symbol in batch_symbols:
                    symbol_data = barset[barset.index.get_level_values('symbol') == symbol]
                    
                    if not symbol_data.empty:
                        # Extraer precios de cierre y volumen
                        close_prices = symbol_data['close']
                        close_prices.index = close_prices.index.get_level_values('timestamp')
                        
                        volume = symbol_data['volume']
                        volume.index = volume.index.get_level_values('timestamp')
                        
                        # Añadir a los DataFrames principales
                        all_prices[symbol] = close_prices
                        all_volumes[symbol] = volume
            
            except Exception as e:
                logging.error(f"Error al obtener datos para lote de símbolos: {str(e)}")
        
        # Reindexar para asegurar que todas las fechas estén presentes
        all_dates = pd.date_range(start=start_date, end=end_date, freq='B')
        all_prices = all_prices.reindex(all_dates)
        all_volumes = all_volumes.reindex(all_dates)
        
        # Eliminar columnas con demasiados valores faltantes
        min_data_points = int(len(all_prices) * 0.9)  # Requerir al menos 90% de datos
        all_prices = all_prices.dropna(axis=1, thresh=min_data_points)
        
        # Filtrar volúmenes para coincidir con precios disponibles
        all_volumes = all_volumes[all_prices.columns]
        
        logging.info(f"Datos históricos obtenidos exitosamente para {all_prices.shape[1]} símbolos")
        
        return all_prices, all_volumes
    
    def initialize_strategy(self, start_date=None, end_date=None):
        """
        Inicializa la estrategia base con datos históricos.
        """
        try:
            
            # Calcular fecha de inicio si no se proporciona
            if start_date is None:
                years_back = self.lookback_years
                start_date = (datetime.now() - timedelta(days=365 * years_back)).strftime('%Y-%m-%d')
            
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Obtener símbolos negociables
            self.symbols = self.get_tradable_assets()
            print(self.symbols)
            logging.info(f"Inicializando estrategia con {len(self.symbols)} símbolos")
            
            # Inicializar estrategia base con los parámetros originales que funcionaron bien
            self.strategy = AdaptiveMultifactorStrategy(
                start_date=start_date,
                end_date=end_date,
                symbols=self.symbols,
                lookback_window=189,                # Más corto que 252 para adaptación más rápida
                regime_window=63,                   # Más corto para detectar cambios de régimen
                n_regimes=5,                        # Más regímenes para mejor granularidad
                rebalance_freq=self.rebalance_freq, # Usar la frecuencia especificada (original: 21 días)
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
                use_all_assets=self.use_all_assets  # Usar todos los activos sin filtrar
            )
            
            # Ejecutar backtest en datos históricos para preparar la estrategia
            self.strategy.backtest()
            logging.info("Estrategia inicializada exitosamente con backtest completo")
            
            return True
            
        except Exception as e:
            logging.error(f"Error al inicializar estrategia: {str(e)}", exc_info=True)
            return False
    
    def get_current_portfolio(self):
        """
        Obtiene el portafolio actual en Alpaca.
        """
        try:
            # Obtener todas las posiciones
            positions = self.api.list_positions()
            
            # Convertir a diccionario con pesos basados en el equity total
            account = self.api.get_account()
            total_equity = float(account.equity)
            
            portfolio = {}
            for position in positions:
                symbol = position.symbol
                market_value = float(position.market_value)
                weight = market_value / total_equity
                portfolio[symbol] = weight
            
            # Añadir efectivo disponible como posición
            cash = float(account.cash)
            cash_weight = cash / total_equity
            portfolio['CASH'] = cash_weight
            
            return portfolio, total_equity
            
        except Exception as e:
            logging.error(f"Error al obtener portafolio actual: {str(e)}")
            return {}, 0
    
    def calculate_trades(self, target_weights, current_portfolio, total_equity):
        """
        Calcula las órdenes necesarias para rebalancear el portafolio.
        """
        try:
            trades = {}
            
            # Para cada activo en los pesos objetivo
            for symbol, target_weight in target_weights.items():
                if symbol in current_portfolio:
                    # Si ya tenemos posición, calcular ajuste
                    current_weight = current_portfolio[symbol]
                    weight_diff = target_weight - current_weight
                    
                    # Si la diferencia es significativa, crear orden
                    if abs(weight_diff) > 0.01:  # Umbral mínimo para operar (1%)
                        target_value = target_weight * total_equity
                        current_value = current_weight * total_equity
                        value_diff = target_value - current_value
                        
                        # Calcular cantidad aproximada a operar
                        # Obtenemos el precio actual
                        try:
                            current_price = float(self.api.get_latest_trade(symbol).price)
                            shares_to_trade = int(value_diff / current_price)
                            
                            if shares_to_trade != 0:
                                trades[symbol] = shares_to_trade
                        except:
                            logging.warning(f"No se pudo obtener precio para {symbol}")
                            
                else:
                    # Si es una nueva posición
                    if target_weight > 0.01:  # Solo abrir si es significativa
                        target_value = target_weight * total_equity
                        
                        # Calcular cantidad aproximada a comprar
                        try:
                            current_price = float(self.api.get_latest_trade(symbol).price)
                            shares_to_buy = int(target_value / current_price)
                            
                            if shares_to_buy > 0:
                                trades[symbol] = shares_to_buy
                        except:
                            logging.warning(f"No se pudo obtener precio para {symbol}")
            
            # Verificar posiciones actuales que deben cerrarse
            for symbol, current_weight in current_portfolio.items():
                if symbol != 'CASH' and symbol not in target_weights and current_weight > 0:
                    # Cerrar posición completamente
                    position = self.api.get_position(symbol)
                    shares_to_sell = -int(position.qty)
                    trades[symbol] = shares_to_sell
            
            return trades
            
        except Exception as e:
            logging.error(f"Error al calcular operaciones: {str(e)}")
            return {}
    
    def execute_trades(self, trades):
        """
        Ejecuta las operaciones en Alpaca.
        """
        if not trades:
            logging.info("No hay operaciones para ejecutar")
            return True
        
        try:
            results = []
            
            # Primero cancelamos todas las órdenes pendientes
            self.api.cancel_all_orders()
            
            # Esperar a que se procesen las cancelaciones
            time.sleep(2)
            
            # Ahora ejecutamos las nuevas órdenes
            for symbol, quantity in trades.items():
                try:
                    if quantity > 0:
                        # Compra
                        order = self.api.submit_order(
                            symbol=symbol,
                            qty=abs(quantity),
                            side='buy',
                            type='market',
                            time_in_force='day'
                        )
                        logging.info(f"✅ Orden de compra creada: {symbol}, {quantity} acciones")
                        results.append((symbol, 'compra', quantity, 'creada'))
                    else:
                        # Venta
                        order = self.api.submit_order(
                            symbol=symbol,
                            qty=abs(quantity),
                            side='sell',
                            type='market',
                            time_in_force='day'
                        )
                        logging.info(f"✅ Orden de venta creada: {symbol}, {abs(quantity)} acciones")
                        results.append((symbol, 'venta', abs(quantity), 'creada'))
                
                except Exception as e:
                    logging.error(f"❌ Error al crear orden para {symbol}: {str(e)}")
                    results.append((symbol, 'error', quantity, str(e)))
            
            # Resumir resultados
            return results
            
        except Exception as e:
            logging.error(f"Error al ejecutar operaciones: {str(e)}")
            return []
    
    def run_strategy_update(self):
        """
        Actualiza la estrategia con datos recientes y genera señales.
        """
        try:
            logging.info("Iniciando actualización de estrategia...")
            
            # Si la estrategia no está inicializada, hacerlo ahora
            if self.strategy is None:
                success = self.initialize_strategy()
                if not success:
                    logging.error("No se pudo inicializar la estrategia")
                    return False
            
            # Verificar si el mercado está abierto
            clock = self.api.get_clock()
            if not clock.is_open:
                next_open = clock.next_open.astimezone(self.eastern_tz)
                logging.info(f"Mercado cerrado. Próxima apertura: {next_open.strftime('%Y-%m-%d %H:%M:%S')}")
                return False
            
            # Obtener fecha actual en zona horaria del mercado
            now = datetime.now(self.eastern_tz)
            current_date = now.strftime('%Y-%m-%d')
            
            # Verificar la fecha del último rebalanceo desde el archivo de persistencia
            last_rebalance_date = self._load_last_rebalance_date()
            if last_rebalance_date:
                # Calcular días de trading transcurridos desde el último rebalanceo
                days_since_rebalance = self._calculate_trading_days_since(last_rebalance_date)
                
                logging.info(f"Último rebalanceo: {last_rebalance_date}, días de trading transcurridos: {days_since_rebalance}")
                
                # Solo rebalancear si han pasado los días configurados
                if days_since_rebalance < self.rebalance_freq:
                    logging.info(f"No han pasado suficientes días de trading para rebalanceo ({days_since_rebalance}/{self.rebalance_freq}). Próximo rebalanceo en {self.rebalance_freq - days_since_rebalance} días.")
                    return False
            
            # También verificar si ya hicimos un rebalanceo hoy
            if self.last_rebalance_date == current_date:
                logging.info(f"Ya se realizó un rebalanceo hoy ({current_date}). Esperando a mañana.")
                return False
            
            logging.info("Obteniendo pesos óptimos de portafolio...")
            
            # Usar los pesos más recientes calculados por la estrategia
            if hasattr(self.strategy, 'weights_history') and len(self.strategy.weights_history) > 0:
                target_weights = self.strategy.weights_history.iloc[-1].copy()
                
                # Filtrar pesos pequeños
                target_weights = target_weights[abs(target_weights) > 0.01]
                
                # Normalizar pesos para asegurar que suman 1
                if target_weights.sum() != 0:
                    target_weights = target_weights / target_weights.sum()
            else:
                logging.error("No hay pesos calculados en la estrategia")
                return False
            
            # Obtener portafolio actual
            current_portfolio, total_equity = self.get_current_portfolio()
            
            if total_equity <= 0:
                logging.error("No se pudo obtener el valor total del portafolio")
                return False
            
            logging.info(f"Portafolio actual: {len(current_portfolio)} posiciones, ${total_equity:.2f} total")
            
            # Calcular operaciones necesarias
            trades = self.calculate_trades(target_weights, current_portfolio, total_equity)
            
            if not trades:
                logging.info("No se necesitan realizar operaciones de rebalanceo")
                return True
            
            logging.info(f"Se requieren {len(trades)} operaciones para rebalancear")
            
            # Ejecutar operaciones
            trade_results = self.execute_trades(trades)
            
            # Actualizar fecha de último rebalanceo y guardarla
            self.last_rebalance_date = current_date
            self._save_last_rebalance_date(current_date)
            
            # Generar informe de operaciones
            if trade_results:
                logging.info("Resumen de operaciones:")
                for symbol, action, quantity, status in trade_results:
                    logging.info(f"  {symbol}: {action} {quantity} acciones - {status}")
            
            logging.info("Rebalanceo completado exitosamente")
            return True
            
        except Exception as e:
            logging.error(f"Error en run_strategy_update: {str(e)}", exc_info=True)
            return False
    
    def _save_last_rebalance_date(self, date_str):
        """
        Guarda la fecha del último rebalanceo en un archivo para persistencia.
        """
        try:
            with open('./artifacts/data/last_rebalance.txt', 'w') as f:
                f.write(date_str)
            logging.info(f"Fecha de último rebalanceo guardada: {date_str}")
        except Exception as e:
            logging.error(f"Error al guardar fecha de rebalanceo: {str(e)}")
    
    def _load_last_rebalance_date(self):
        """
        Carga la fecha del último rebalanceo desde el archivo de persistencia.
        """
        try:
            if os.path.exists('./artifacts/data/last_rebalance.txt'):
                with open('./artifacts/data/last_rebalance.txt', 'r') as f:
                    date_str = f.read().strip()
                logging.info(f"Fecha de último rebalanceo cargada: {date_str}")
                return date_str
            else:
                logging.info("No se encontró archivo de fecha de último rebalanceo")
                return None
        except Exception as e:
            logging.error(f"Error al cargar fecha de rebalanceo: {str(e)}")
            return None
    
    def _calculate_trading_days_since(self, date_str):
        """
        Calcula el número de días de trading transcurridos desde la fecha proporcionada.
        """
        try:
            last_date = datetime.strptime(date_str, '%Y-%m-%d')
            now = datetime.now()
            
            # Para simplificar, podemos usar la API de Alpaca para obtener días de trading
            calendar = self.api.get_calendar(
                start=last_date.strftime('%Y-%m-%d'),
                end=now.strftime('%Y-%m-%d')
            )
            
            # El número de días de trading es la longitud del calendario menos 1 (excluir el día inicial)
            trading_days = len(calendar) - 1
            return max(0, trading_days)
            
        except Exception as e:
            logging.error(f"Error al calcular días de trading: {str(e)}")
            # En caso de error, estimar días hábiles (excluyendo fines de semana)
            try:
                days_diff = (now - last_date).days
                weekends = 0
                for i in range(days_diff):
                    day = (last_date + timedelta(days=i)).weekday()
                    if day >= 5:  # 5 y 6 son sábado y domingo
                        weekends += 1
                return days_diff - weekends
            except:
                return 0
    
    def setup_scheduled_runs(self, time_str="09:35"):
        """
        Configura ejecuciones programadas diarias.
        Solo ejecuta en días de trading (lunes a viernes).
        
        Parámetros:
        -----------
        time_str : str
            Hora de ejecución en formato HH:MM (hora del Este de EE.UU.)
        """
        def job():
            # Verificar si es día de semana (lunes a viernes)
            if datetime.now().weekday() < 5:  # 0-4 son lunes a viernes
                self.run_strategy_update()
            else:
                logging.info("Hoy es fin de semana. No se ejecuta la estrategia.")
        
        # Programar ejecución diaria
        schedule.every().day.at(time_str).do(job)
        
        logging.info(f"Ejecución programada para las {time_str} (hora del Este) en días de trading")
        
        print(f"Estrategia programada para ejecutarse a las {time_str} (ET) cada día de trading")
        print("Dejando el programa en ejecución. Presiona Ctrl+C para detener.")
        
        # Bucle principal
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Verificar cada minuto
        except KeyboardInterrupt:
            logging.info("Programa detenido por el usuario")
            print("Programa detenido.")

# Función principal para ejecutar el adaptador
def main():
    try:
        print("Iniciando adaptador de estrategia para Alpaca Paper Trading...")
        
        # Crear adaptador con frecuencia de rebalanceo de 21 días (como la estrategia original)
        adapter = AlpacaStrategyAdapter(
            lookback_years=3,
            rebalance_freq=21,  # Rebalancear cada 21 días de trading
            use_all_assets=True  # Usar todos los activos sin filtrar
        )
        
        # Inicializar estrategia con datos históricos
        adapter.initialize_strategy()
        
        # Ejecutar inmediatamente un rebalanceo para comenzar
        print("Ejecutando primer rebalanceo de la estrategia...")
        adapter.run_strategy_update()
        
        # Configurar ejecuciones programadas (9:35 AM ET, poco después de la apertura)
        adapter.setup_scheduled_runs(time_str="09:35")
        
    except Exception as e:
        logging.error(f"Error en la función principal: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}. Ver ./artifacts/alpaca_trading.log para más detalles.")

if __name__ == "__main__":
    main()