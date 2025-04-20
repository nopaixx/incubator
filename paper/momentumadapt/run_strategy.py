#!/usr/bin/env python3
"""
Script principal para ejecutar la estrategia de trading automatizada en Alpaca
con carga desde .env y opción de forzar rebalanceo.
"""
import os
import argparse
from datetime import datetime
import logging
import signal
import sys
from dotenv import load_dotenv
from alpaca_strategy_resilient import AlpacaTradingStrategy

# Variable global para la estrategia
strategy = None

def signal_handler(sig, frame):
    """Maneja la señal de interrupción (Ctrl+C) para guardar estado antes de salir."""
    global strategy
    if strategy:
        print("\nInterrumpiendo estrategia, guardando estado...")
        strategy.logger.info("Interrupción detectada, guardando estado")
        strategy.save_state()
        print("Estado guardado. Saliendo...")
    sys.exit(0)

def main():
    """Función principal que inicializa y ejecuta la estrategia."""
    global strategy
    
    # Cargar variables de entorno desde .env
    load_dotenv()
    
    # Leer las variables específicas de la estrategia desde .env
    default_api_key = os.environ.get("MOMENTUM_ADAPT_API_KEY", "")
    default_api_secret = os.environ.get("MOMENTUM_ADAPT_API_SECRET", "")
    default_base_url = os.environ.get("MOMENTUM_ADAPT_BASE_URL", "https://paper-api.alpaca.markets")
    
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description="Estrategia de Trading Automatizada en Alpaca (Resiliente)")
    parser.add_argument(
        "--api_key", 
        type=str, 
        default=default_api_key,
        help="API Key de Alpaca (por defecto: valor de MOMENTUM_ADAPT_API_KEY en .env)"
    )
    parser.add_argument(
        "--api_secret", 
        type=str, 
        default=default_api_secret,
        help="API Secret de Alpaca (por defecto: valor de MOMENTUM_ADAPT_API_SECRET en .env)"
    )
    parser.add_argument(
        "--base_url", 
        type=str, 
        default=default_base_url,
        help="URL base de Alpaca (por defecto: valor de MOMENTUM_ADAPT_BASE_URL en .env)"
    )
    parser.add_argument(
        "--initial_cash", 
        type=float, 
        default=None,
        help="Efectivo inicial a invertir (por defecto: todo el efectivo disponible)"
    )
    parser.add_argument(
        "--rebalance_freq", 
        type=str, 
        choices=["D", "W", "M"], 
        default="W",
        help="Frecuencia de rebalanceo: D=diario, W=semanal, M=mensual (por defecto: M)"
    )
    parser.add_argument(
        "--check_interval", 
        type=int, 
        default=60,
        help="Intervalo en minutos para verificar si es momento de rebalancear (por defecto: 60)"
    )
    parser.add_argument(
        "--run_once", 
        action="store_true",
        help="Ejecutar la estrategia una sola vez en lugar de programada"
    )
    parser.add_argument(
        "--force_rebalance", 
        action="store_true",
        help="Forzar rebalanceo inmediato independientemente de la fecha del último rebalanceo"
    )
    
    args = parser.parse_args()
    
    # Verificar credenciales
    if not args.api_key or not args.api_secret:
        print("ERROR: Se requieren API Key y Secret de Alpaca.")
        print("Puedes proporcionarlas como argumentos, como variables de entorno, o en un archivo .env:")
        print("  En .env:")
        print("    MOMENTUM_ADAPT_API_KEY='tu_api_key'")
        print("    MOMENTUM_ADAPT_API_SECRET='tu_api_secret'")
        print("    MOMENTUM_ADAPT_BASE_URL='https://paper-api.alpaca.markets'")
        sys.exit(1)
    
    # Crear directorios necesarios si no existen
    for directory in ['logs', 'data']:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    print("\n=== Estrategia de Trading Automatizada en Alpaca (Resiliente) ===")
    print(f"Base URL: {args.base_url}")
    print(f"Frecuencia de rebalanceo: {args.rebalance_freq}")
    print(f"Intervalo de verificación: {args.check_interval} minutos")
    if args.force_rebalance:
        print("MODO REBALANCEO FORZADO: Se realizará un rebalanceo inmediato")
    
    # Configurar manejador de señales para interrupciones
    signal.signal(signal.SIGINT, signal_handler)
    
    # Inicializar estrategia
    strategy = AlpacaTradingStrategy(args.api_key, args.api_secret, args.base_url)
    
    # Si se solicita rebalanceo forzado, borrar fecha del último rebalanceo
    if args.force_rebalance:
        strategy.last_rebalance_date = None
        strategy.logger.info("Rebalanceo forzado solicitado - se ignorará la fecha del último rebalanceo")
    
    # Ejecutar estrategia
    if args.run_once:
        print("Ejecutando estrategia una sola vez...")
        success = strategy.run_strategy(
            initial_cash=args.initial_cash,
            rebalance_freq=args.rebalance_freq
        )
        if success:
            print("Estrategia ejecutada exitosamente.")
        else:
            print("Error ejecutando estrategia. Revisa los logs para más detalles.")
    else:
        print(f"Iniciando estrategia programada (verificando cada {args.check_interval} minutos)...")
        try:
            strategy.run_scheduled_strategy(
                interval_minutes=args.check_interval,
                rebalance_freq=args.rebalance_freq
            )
        except KeyboardInterrupt:
            # Este bloque no debería ejecutarse ya que el manejador de señales se encarga,
            # pero se incluye como respaldo
            print("\nEstrategia detenida manualmente.")
            strategy.save_state()

if __name__ == "__main__":
    main()