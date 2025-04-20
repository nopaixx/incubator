#!/usr/bin/env python3
"""
Script sencillo para ejecutar la estrategia de trading en Alpaca.
Incluye capacidad de recuperación ante caídas o interrupciones.
"""
import os
import logging
import signal
import sys

from adapter_alpaca import AlpacaStrategyAdapter


# Crear directorio para logs si no existe
os.makedirs('./artifacts', exist_ok=True)

# Configurar logging básico para este script
logging.basicConfig(
    filename='./artifacts/strategy_run.log',
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def handle_exit(signum, frame):
    """Manejador para salida limpia al recibir señales de terminación"""
    print("\n⚠️ Recibida señal de terminación. Cerrando la estrategia de forma segura...")
    logging.info("Programa terminado por señal del sistema")
    sys.exit(0)

def main():
    """Función principal para ejecutar la estrategia"""
    try:
        # Configurar manejadores de señales para salida limpia
        signal.signal(signal.SIGINT, handle_exit)  # Ctrl+C
        signal.signal(signal.SIGTERM, handle_exit) # kill
        
        print("🚀 Iniciando estrategia multifactorial con Alpaca Paper Trading")
        print("--------------------------------------------------------------")
        
        # Crear e inicializar el adaptador
        adapter = AlpacaStrategyAdapter(
            lookback_years=3,       # Usar 3 años de datos históricos
            rebalance_freq=21,      # Rebalancear cada 21 días (como en el original)
            use_all_assets=True     # Usar todos los activos disponibles
        )
        
        # Inicializar estrategia con datos históricos
        print("⏳ Inicializando estrategia con datos históricos...")
        adapter.initialize_strategy()
        
        # Verificar si es necesario realizar rebalanceo o si es muy pronto
        # El adaptador ahora verifica la fecha del último rebalanceo
        print("⚖️ Verificando si es necesario rebalancear...")
        last_rebalance = adapter._load_last_rebalance_date()
        
        if last_rebalance:
            days_since = adapter._calculate_trading_days_since(last_rebalance)
            print(f"ℹ️ Último rebalanceo: {last_rebalance} ({days_since} días de trading)")
            
            if days_since >= adapter.rebalance_freq:
                print(f"✅ Han pasado {days_since} días de trading. Ejecutando rebalanceo...")
                adapter.run_strategy_update()
            else:
                print(f"ℹ️ Solo han pasado {days_since}/{adapter.rebalance_freq} días. Esperando...")
                print(f"   El próximo rebalanceo será en {adapter.rebalance_freq - days_since} días de trading")
        else:
            print("🔄 No hay registro de rebalanceos previos. Ejecutando primer rebalanceo...")
            adapter.run_strategy_update()
        
        # Configurar ejecuciones programadas (9:35 AM ET, poco después de la apertura)
        print("🕒 Configurando ejecuciones programadas...")
        print(f"   La estrategia verificará diariamente a las 9:35 AM ET si deben hacerse rebalanceos")
        print(f"   La frecuencia de rebalanceo está configurada a cada {adapter.rebalance_freq} días de trading")
        print("\n💡 IMPORTANTE: Este programa debe mantenerse en ejecución para funcionar automáticamente")
        print("   Si se cierra, puedes reiniciarlo en cualquier momento y continuará desde donde lo dejó")
        
        adapter.setup_scheduled_runs(time_str="09:35")
        
    except Exception as e:
        logging.error(f"Error al ejecutar la estrategia: {str(e)}", exc_info=True)
        print(f"❌ Error: {str(e)}. Revisa el archivo de log para más detalles.")
        print("   ./artifacts/strategy_run.log")
        print("   ./artifacts/alpaca_trading.log")

if __name__ == "__main__":
    main()