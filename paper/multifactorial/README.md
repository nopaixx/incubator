# Estrategia Multifactorial Adaptativa con Alpaca Paper Trading

Este proyecto adapta la Estrategia Multifactorial Adaptativa para ser ejecutada automáticamente con Alpaca Paper Trading. Mantiene la lógica original de negociación que ha demostrado buenos rendimientos, pero la integra con la API de Alpaca para operar automáticamente.

## 🚀 Configuración rápida (para principiantes)

Sigue estos pasos sencillos para poner en marcha la estrategia:

### 1. Requisitos previos

- Python 3.8 o superior
- Cuenta de Alpaca Paper Trading (gratuita)

### 2. Obtener credenciales de Alpaca

1. Regístrate en [Alpaca](https://app.alpaca.markets/signup) si aún no tienes cuenta
2. Ve a tu Dashboard y haz clic en "View API Keys"
3. Obtén tu API Key y API Secret para Paper Trading

### 3. Configurar el entorno

1. Clona o descarga este repositorio
2. Instala las dependencias:

```bash
pip install -r requirements.txt
```

3. Crea un archivo `.env` en la carpeta raíz con tus credenciales:

```
ALPACA_API_KEY=TU_API_KEY_AQUI
ALPACA_API_SECRET=TU_API_SECRET_AQUI
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### 4. Ejecutar la estrategia

Ejecuta el script principal:

```bash
python run_strategy.py
```

El script hará lo siguiente:
- Inicializará la estrategia con datos históricos
- Ejecutará un primer rebalanceo de la cartera
- Configurará la ejecución programada para futuros rebalanceos

## ⏰ Frecuencia de ejecución

La estrategia está configurada para ejecutarse **cada 21 días de trading** (igual que la estrategia original) a las 9:35 AM (hora del Este de EE.UU.), poco después de la apertura del mercado.

Una vez iniciado el script, se ejecutará automáticamente:
- Solo en días de trading (lunes a viernes, excluyendo feriados de mercado)
- A las 9:35 AM ET cada día
- Verificará si han pasado 21 días desde el último rebalanceo antes de realizar operaciones

## 🔄 Ejecución continua

Para mantener la estrategia funcionando continuamente (incluso si tu computadora se reinicia), tienes varias opciones:

### Opción 1: Configurar como servicio del sistema (recomendado)

En Linux/Mac con systemd:

1. Crea un archivo de servicio:
```bash
sudo nano /etc/systemd/system/trading-strategy.service
```

2. Añade el siguiente contenido (ajusta las rutas):
```
[Unit]
Description=Trading Strategy Service
After=network.target

[Service]
User=tu_usuario
WorkingDirectory=/ruta/a/tu/proyecto
ExecStart=/usr/bin/python3 /ruta/a/tu/proyecto/run_strategy.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

3. Habilita e inicia el servicio:
```bash
sudo systemctl enable trading-strategy
sudo systemctl start trading-strategy
```

### Opción 2: Usar un administrador de procesos como PM2

Si estás familiarizado con Node.js, PM2 es una excelente opción:

```bash
npm install -g pm2
pm2 start run_strategy.py --name "trading-strategy" --interpreter python3
pm2 save
pm2 startup
```

### Opción 3: Usar Docker (para usuarios avanzados)

Se proporciona un Dockerfile para ejecutar la estrategia en un contenedor.

## 📊 Monitoreo y logs

La estrategia guarda información detallada en los siguientes archivos:

- `artifacts/alpaca_trading.log`: Log principal con todas las operaciones y decisiones
- `artifacts/strategy_run.log`: Log de ejecución del script principal
- `artifacts/errors.txt`: Registro de errores (de la estrategia original)

Para ver el estado actual:

```bash
# Ver los últimos 50 logs de operaciones
tail -n 50 artifacts/alpaca_trading.log
```

## ⚠️ Importante

- Esta es una estrategia automatizada - revisar regularmente que esté funcionando correctamente
- Comprobar periódicamente el balance y rendimiento en el dashboard de Alpaca
- Los eventos de mercado extremos pueden requerir intervención manual

## 🔧 Personalización

Si deseas modificar algunos parámetros básicos, edita el archivo `run_strategy.py`:

- `lookback_years`: Años de datos históricos a utilizar (predeterminado: 3)
- `rebalance_freq`: Frecuencia de rebalanceo en días (predeterminado: 21)
- `time_str`: Hora de ejecución diaria (predeterminado: "09:35")

## 📞 Soporte y ayuda

Si encuentras problemas:

1. Verifica los archivos de log para entender el error
2. Consulta la [documentación de Alpaca](https://alpaca.markets/docs/api-documentation/)
3. Busca ayuda en comunidades de trading algorítmico como r/algotrading

¡Buena suerte con tu trading! 📈