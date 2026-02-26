# 🌾 Alpha Harvester — Implementation Guide (God-Tier)

> **Documento Vivo de Implementación**: Este documento es la guía técnica exhaustiva para construir el bot de Cash & Carry "Alpha Harvester". Está diseñado para que cualquier ingeniero pueda seguirlo paso a paso. Úsalo como checklist maestro.

---

## 🏗️ Fase 1: Infraestructura y Connector Base

### Dependencias y Configuración Global
- [ ] Modificar `pyproject.toml` para agregar `ccxt>=4.0.0` en `[project.dependencies]`.
- [ ] Agregar variables de entorno a `.env.example`:
  - `BINANCE_API_KEY`
  - `BINANCE_API_SECRET`
  - `PAPER_TRADING` (default: `true`)
  - `MAX_POSITION_USDT` (default: `1000`)
  - `MIN_FUNDING_RATE` (default: `0.00035`)
  - `FUTURES_LEVERAGE` (default: `2`)
  - `SCAN_SYMBOLS` (default: `BTC/USDT,ETH/USDT`)
  - `USE_BNB_FEE_DISCOUNT` (default: `true`)

### El Conector Universal de Binance (`autopilot/connectors/binance_connector.py`)
- [ ] Crear clase `BinanceConnector` inicializando dos instancias de `ccxt.binance`:
  - Una para `spot` (defaultOptions: `{'defaultType': 'spot'}`)
  - Otra para `futures` (defaultOptions: `{'defaultType': 'future'}`)
- [ ] Implementar `setup_account(symbol)`:
  - Cambiar margin type a `CROSSED` (`fapiPrivate_post_margintype`).
  - Cambiar position mode a `One-Way` (`fapiPrivate_post_positionside_dual` con `dualSidePosition=false`).
  - Setear leverage a `2x` (`fapiPrivate_post_leverage`).
  - *Manejo de errores*: Ignorar error si ya está configurado.
- [ ] Implementar `get_funding_rates(symbols)`:
  - Consumir `fapiPublic_get_premiumindex`.
  - Extraer `lastFundingRate`, `nextFundingTime` y el clave `fundingIntervalHours` (intervalo dinámico).
- [ ] Implementar `get_funding_rate_history(symbol, limit)`:
  - Consumir `/fapi/v1/fundingRate`.
- [ ] Implementar Market Data getters:
  - `get_mark_price(symbol)` (futures mark price).
  - `get_spot_price(symbol)` (spot last price).
  - `get_basis(symbol)`: Calcular la diferencia porcentual `(futures_mark - spot_last) / spot_last`.
- [ ] Implementar **Predictive Signals (God-Tier)**:
  - `get_open_interest(symbol)` (`/fapi/v1/openInterest`).
  - `get_long_short_ratio(symbol, period="5m")` (`/futures/data/globalLongShortAccountRatio`).
  - `get_top_trader_positions(symbol, period="5m")` (`/futures/data/topLongShortPositionRatio`).
  - `get_recent_liquidations(symbol, limit=10)` (`/fapi/v1/allForceOrders`).
- [ ] Implementar Wallet & Position getters:
  - `get_account_balance()`: Sumar USDT libre en spot y USDT libre en futures margin.
  - `get_position(symbol)`: Retornar tamaño de posición, unrealized PnL y el **ADL (Auto-Deleveraging) quantile**.
- [ ] Implementar `transfer_between_wallets(asset, amount, from_type, to_type)`:
  - Consumir endpoint de transferencia universal (`sapi_post_asset_transfer`). Tipos: `MAIN` (spot) ↔ `UMFUTURE` (futures).
- [ ] Implementar Order Execution (Limit Orders mandatorios):
  - `place_spot_limit_order(symbol, side, qty, price)`.
  - `place_futures_limit_order(symbol, side, qty, price)`.
  - `place_market_fallback_order(...)` (Solo para unwinds de emergencia).
- [ ] (Tests) Crear `tests/autopilot/connectors/test_binance_connector.py` mockeando `ccxt`.

---

## 🧬 Fase 2: Modelos y Arquitectura del Workflow

### Estructura Base (`workflows/alpha_harvester/`)
- [ ] Crear directorio y `__init__.py`.
- [ ] Crear `manifest.yaml` declarando el cron trigger (cada 5m para dar tiempo a los predictores) o webhook para Telegram, y config validation.

### Modelos de Datos Meticulosos (`models.py`)
- [ ] `PredictiveSignals`: Pydantic model con `oi_trend` (float), `ls_ratio` (float), `top_trader_skew` (float), `liquidation_volume` (float), `entry_confidence` (0.0 a 1.0).
- [ ] `MarketSnapshot`: Precios, spread, `predicted_rate`, `next_funding_time`, `interval_hours`, e instanciar `PredictiveSignals`.
- [ ] `SymbolOpportunity`: `pure_premium` (rate - interest 0.01%), `fee_adjusted_apy`, y `discounted_score` (multiplicado por confidence).
- [ ] `ActionDecision`: Enum (`ENTER`, `EXIT`, `HOLD`, `SKIP`, `DODGE`).
- [ ] `FundingOpportunity`: Envuelve `SymbolOpportunity`, `ActionDecision` y `recommended_size_usdt` (Quarter-Kelly escalar).
- [ ] `TradeExecution`: Estado de piernas (FILLED, PARTIAL), fill prices reales, e invoice de slippage estimado vs real.
- [ ] `PortfolioState`: Posiciones abiertas, PnL realized + unrealized (por basis), bankroll total (auto-compounding).

---

## ⚙️ Fase 3: Motores Analíticos (Core Logic sin LLM)

### Motor de Riesgo y Sizing (`risk.py`)
- [ ] Implementar función **Auto-Compounding**: Leer `PortfolioState` guardado y sumar profit a `MAX_POSITION_USDT`.
- [ ] Implementar **Quarter-Kelly Sizing**: 
  - Fórmula: `f = (bp - q) / b`, mitigado a `f/4`. 
  - Input: Win rate histórico (del backtest/paper), APY esperado.
  - Escalar el resultado final por `entry_confidence` (si signal=1.0 usa todo el quarter-kelly, si 0.5 usa la mitad).
- [ ] Función calcular capital-eficiente: `max_position_per_leg = total_capital / (1 + (1 / leverage))`.
- [ ] Implementar filtro de convergencia/divergencia temporal: Basis > 0.5% **solo emite EXIT_SIGNAL_BASIS si se mantiene por más de 2 intervalos consecutivos (10 min)**. Buffer de estado transitorio requerido.

### El Shadow Engine (`paper_engine.py`)
- [ ] Implementar ledger local (`.alpha_harvester_paper_state.json`).
- [ ] Simular fills: Tomar mark/spot prices, aplicar **0.05% slippage penalizado**.
- [ ] Simular Funding Collection: Función cron interna que lea si se cruzó la marca de hora (`next_funding_time`) y abone el `predicted_rate` de la ventana anterior al PnL ficticio, actualizando el bankroll per-step.

### Backtest Engine Histórico (`backtest.py` - Tool de CLI)
- [ ] Script standalone que descarga 30 días de `get_funding_rate_history`.
- [ ] Ejecuta todas las reglas definidas sobre el histórico.
- [ ] Retorna: `# Trades`, `Total APY`, `Max Drawdown`, `Sharpe Ratio`. Criterio de rechazo duro codificado en assert: `Sharpe < 1.5 -> abort`.

---

## 🚀 Fase 4: Pipeline DAG (El Orquestador)

Implementar en `steps.py` como `FunctionalAgent` (cero LLMs).

- [ ] **Step 1: `fetch_market_data`**: Llamadas a APIs del connector. Retorna `MarketSnapshot` parcial (precios + rates).
- [ ] **Step 2: `fetch_predictive_signals`**: (Asíncrono paralelo a 1) Obtiene la capa predictiva. Une data y retorna `MarketSnapshot` completo.
- [ ] **Step 3: `evaluate_funding_opportunity`**:
  - Fórmula real: Restar 0.01% (interest rate default). Comparar el premium resultante vs el threshold (0.035%).
  - Detectar **Double-Profit**: Si premium > 0.1%, inflar tamaño un 20% (agresivo por mean reversion).
- [ ] **Step 4: `risk_gate`**:
  - Bloqueos hard: Drawdown acumulado > 3%? Abortar día. ADL quantile en rojo (>3)? Abortar símbolo.
- [ ] **Step 5: `funding_dodge`** (El Santo Grial del timing):
  - Verifica la hora local vs `next_funding_time`.
  - Si faltan < 90 segundos Y rate es negativo > 0.16% (costo de huir): Retorna `DODGE_SIGNAL`.
- [ ] **Step 6: `prepare_capital`**:
  - Revisa estado actual vs `FundingOpportunity`.
  - Si es Entry: Ejecuta transferencia interna `transfer_between_wallets` exacto a 67/33 (spot/futures). Llama `setup_account`.
- [ ] **Step 7: `execute_trade`**:
  - Branching: Si `PAPER_TRADING=true`, llamar `paper_engine`.
  - Si LIVE: Enviar limit orders **concurrentes** usando `asyncio.gather` para minimizar riesgo de leg asimétrica. 
  - Manejo de fallos: Si spot fillea pero futures no, cancelar orden y market-sell el spot (unwind agresivo de protección).
- [ ] **Step 8: `notify_and_persist`**:
  - Formatear payload de EventBus para ser capturado por notificaciones push/telegram. 

### Orquestación en `pipeline.yaml`
- [ ] Definir topología `DAG`. Mostrar explícitamente paralelismo entre `fetch_market_data` y `fetch_predictive_signals`.

---

## 🎮 Fase 5: Centro de Comando (Telegram)

### Handlers de Comandos (`telegram_commands.py`)
- [ ] Ligar webhook handler del workflow al bus del platform.
- [ ] Comando `/stats`: Calcular Sharpe, Drawdown, y APY anualizado iterando la historia de la DB (o local ledger).
- [ ] Comando `/positions`: Iterar el state y mostrar exposición por símbolo + basis live.
- [ ] Comando `/kill`: Invoca unwind market orden duro a todas las bolsas, limpia capital de margin a spot, y apaga variables de entorno operables localmente (kill switch state).
- [ ] Comando `/mode`: Alterna el estado en memoria de `PAPER_TRADING`, reiniciando los adaptadores del connector en caliente.

---

## �� Criterios de Aceptación (Verificación Final)

- [ ] `pytest workflows/alpha_harvester/tests/` corre al 100%. Las pruebas unitarias cubren la matemática del dodger rate y los fees divididos maker/taker (0.156% round-trip verificado en assertions).
- [ ] El bot es completamente determinístico bajo el framework ADK pero está blindado con loggers `.info` sobre la capa de Functional Agents en cada nodo del DAG.
- [ ] El backtesting de 1 mes arroja un Sharpe > 1.5 y Drawdown dócil < 5%. 
