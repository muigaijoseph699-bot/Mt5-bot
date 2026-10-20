"""
EMA-crossover / ATR-stop trend bot that trades an MT5 account through the MetaApi cloud
service, so it does not need a PC or a desktop MT5 terminal.

    python mt5_cloud_bot.py --once    # one pass (use with a scheduler, e.g. GitHub Actions)
    python mt5_cloud_bot.py           # loop forever, waking after each candle closes

Settings come from environment variables (see CLOUD_README.md). It is DRY RUN by default:
it only logs what it would do until you set LIVE_TRADING=1.
"""
import argparse
import asyncio
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mt5bot")

TF_SECONDS = {"5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400}


def env(name, default):
    v = os.getenv(name)
    return v if v not in (None, "") else default


@dataclass
class BotConfig:
    symbols: list = field(default_factory=lambda: ["EURUSD", "XAUUSD", "BTCUSD"])
    timeframe: str = "1h"
    fast_ema: int = 20
    slow_ema: int = 50
    atr_period: int = 14
    atr_mult: float = 2.5
    direction: str = "both"          # both | long | short
    risk_percent: float = 1.0        # % of equity risked per trade
    daily_loss_pct: float = 3.0      # no new entries after this loss
    max_spread_atr: float = 0.20     # skip entry if spread > this x ATR (0 = off)
    max_lots: float = 1.0            # hard cap per trade
    max_open_positions: int = 4
    min_equity: float = 0.0          # stop opening trades below this equity (0 = off)
    entry_window_min: int = 15       # only enter on a signal this fresh
    delay_seconds: int = 20          # loop mode: wait this long after the candle closes
    candles: int = 300
    comment: str = "EMA_ATR"
    live: bool = False               # False = dry run (log only)
    allow_real: bool = False         # must be True to trade a REAL account

    @classmethod
    def from_env(cls):
        d = cls()
        return cls(
            symbols=[s.strip() for s in env("SYMBOLS", ",".join(d.symbols)).split(",") if s.strip()],
            timeframe=env("TIMEFRAME", d.timeframe),
            fast_ema=int(env("FAST_EMA", d.fast_ema)),
            slow_ema=int(env("SLOW_EMA", d.slow_ema)),
            atr_period=int(env("ATR_PERIOD", d.atr_period)),
            atr_mult=float(env("ATR_MULT", d.atr_mult)),
            direction=env("DIRECTION", d.direction).lower(),
            risk_percent=float(env("RISK_PERCENT", d.risk_percent)),
            daily_loss_pct=float(env("DAILY_LOSS_PCT", d.daily_loss_pct)),
            max_spread_atr=float(env("MAX_SPREAD_ATR", d.max_spread_atr)),
            max_lots=float(env("MAX_LOTS", d.max_lots)),
            max_open_positions=int(env("MAX_OPEN_POSITIONS", d.max_open_positions)),
            min_equity=float(env("MIN_EQUITY", d.min_equity)),
            entry_window_min=int(env("ENTRY_WINDOW_MIN", d.entry_window_min)),
            live=env("LIVE_TRADING", "0") == "1",
            allow_real=env("ALLOW_REAL", "0") == "1",
        )

    def validate(self):
        assert self.timeframe in TF_SECONDS, f"TIMEFRAME must be one of {list(TF_SECONDS)}"
        assert self.direction in ("both", "long", "short"), "DIRECTION must be both/long/short"
        assert 0 < self.fast_ema < self.slow_ema, "FAST_EMA must be smaller than SLOW_EMA"
        assert 0 < self.risk_percent <= 5, "RISK_PERCENT must be between 0 and 5"
        assert self.symbols, "SYMBOLS is empty"


# ----------------------------------------------------------------------------------------
# Indicators (same logic as the Python backtester and the MQL5 EA)
# ----------------------------------------------------------------------------------------
def warmup(cfg):
    return max(cfg.slow_ema, cfg.atr_period) + 2


def add_indicators(df, cfg):
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=cfg.fast_ema, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=cfg.slow_ema, adjust=False).mean()
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / cfg.atr_period, adjust=False).mean()
    above = df["ema_fast"] > df["ema_slow"]
    was_above = above.shift(1, fill_value=False)
    df["cross_up"] = above & ~was_above
    df["cross_down"] = ~above & was_above
    df.iloc[: warmup(cfg), df.columns.get_indexer(["cross_up", "cross_down"])] = False
    return df


# ----------------------------------------------------------------------------------------
# Data and sizing helpers
# ----------------------------------------------------------------------------------------
async def get_closed_candles(account, symbol, cfg, now):
    """Latest candles for `symbol`, sorted oldest-first, with the still-forming one removed."""
    try:
        raw = await account.get_historical_candles(symbol, cfg.timeframe, limit=cfg.candles)
    except TypeError:
        raw = await account.get_historical_candles(symbol, cfg.timeframe)
    if not raw:
        return None
    df = pd.DataFrame(raw)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").drop_duplicates("time").set_index("time")
    df = df[["open", "high", "low", "close"]].astype(float)
    tf = pd.Timedelta(seconds=TF_SECONDS[cfg.timeframe])
    return df[df.index + tf <= pd.Timestamp(now)]  # closed candles only


def calc_lots(equity, sl_distance, spec, price, cfg):
    """Lots so that hitting the stop loses risk_percent of equity.
    Uses MetaApi's lossTickValue (value of one tick for 1 lot, in account currency)."""
    tick = float(spec["tickSize"])
    tick_value = float(price.get("lossTickValue") or 0)
    if tick <= 0 or tick_value <= 0 or sl_distance <= 0:
        log.warning("Cannot size trade (tickSize=%s lossTickValue=%s)", tick, tick_value)
        return 0.0
    loss_per_lot = sl_distance / tick * tick_value
    lots = equity * cfg.risk_percent / 100.0 / loss_per_lot

    step = float(spec.get("volumeStep") or 0.01)
    vmin = float(spec.get("minVolume") or step)
    vmax = float(spec.get("maxVolume") or cfg.max_lots)
    lots = math.floor(lots / step + 1e-9) * step
    lots = min(lots, vmax, cfg.max_lots)
    if lots < vmin:
        log.warning("Skipped: risk allows %.4f lots, below broker minimum %.4f", lots, vmin)
        return 0.0
    decimals = max(0, int(math.ceil(-math.log10(step) - 1e-9)))
    return round(lots, decimals)


# ----------------------------------------------------------------------------------------
# Per-symbol logic
# ----------------------------------------------------------------------------------------
async def process_symbol(conn, account, symbol, cfg, equity, halted, now, last_seen):
    actions = []
    tf = timedelta(seconds=TF_SECONDS[cfg.timeframe])

    df = await get_closed_candles(account, symbol, cfg, now)
    if df is None or len(df) < warmup(cfg) + 5:
        log.warning("%s: not enough candle data", symbol)
        return actions
    df = add_indicators(df, cfg)
    sig, ts = df.iloc[-1], df.index[-1]
    if last_seen.get(symbol) == ts:
        return actions
    last_seen[symbol] = ts

    bullish = bool(sig["ema_fast"] > sig["ema_slow"])
    all_pos = await conn.get_positions()
    mine = [p for p in all_pos if p["symbol"] == symbol]
    log.info("%s: candle %s close=%.5f trend=%s positions=%d", symbol, ts, sig["close"],
             "up" if bullish else "down", len(mine))

    # 1) Exit when the trend has flipped against an open position
    for p in mine:
        is_long = p["type"] == "POSITION_TYPE_BUY"
        if (is_long and not bullish) or (not is_long and bullish):
            if cfg.live:
                await conn.close_position(position_id=p["id"])
                log.info("%s: closed %s position %s", symbol, "long" if is_long else "short", p["id"])
            else:
                log.info("%s: DRY RUN would close %s position %s", symbol, "long" if is_long else "short", p["id"])
            actions.append(("close", symbol, p["id"]))
    if actions:
        if cfg.live:
            all_pos = await conn.get_positions()
        else:
            all_pos = [p for p in all_pos if p["symbol"] != symbol]
        mine = [p for p in all_pos if p["symbol"] == symbol]

    # 2) Enter on a fresh crossover only
    if mine:
        return actions
    fresh = (pd.Timestamp(now) - (ts + pd.Timedelta(tf))) <= pd.Timedelta(minutes=cfg.entry_window_min)
    side = None
    if sig["cross_up"] and cfg.direction in ("both", "long"):
        side = "buy"
    elif sig["cross_down"] and cfg.direction in ("both", "short"):
        side = "sell"
    if side is None:
        return actions
    if not fresh:
        log.info("%s: %s signal is stale, ignoring", symbol, side)
        return actions
    if halted:
        log.warning("%s: %s signal skipped, loss limit reached", symbol, side)
        return actions
    if len(all_pos) >= cfg.max_open_positions:
        log.warning("%s: %s signal skipped, max open positions reached", symbol, side)
        return actions

    price = await conn.get_symbol_price(symbol)
    spec = await conn.get_symbol_specification(symbol)
    bid, ask = float(price["bid"]), float(price["ask"])
    atr = float(sig["atr"])
    if cfg.max_spread_atr > 0 and (ask - bid) > atr * cfg.max_spread_atr:
        log.warning("%s: spread %.5f too wide vs ATR %.5f, skipped", symbol, ask - bid, atr)
        return actions

    entry = ask if side == "buy" else bid
    dist = atr * cfg.atr_mult
    sl = round(entry - dist if side == "buy" else entry + dist, int(spec.get("digits", 5)))
    lots = calc_lots(equity, abs(entry - sl), spec, price, cfg)
    if lots <= 0:
        return actions

    if not cfg.live:
        log.info("%s: DRY RUN would %s %.2f lots @ ~%.5f, stop %.5f", symbol, side.upper(), lots, entry, sl)
    else:
        opts = {"comment": cfg.comment}
        if side == "buy":
            res = await conn.create_market_buy_order(symbol=symbol, volume=lots, stop_loss=sl, options=opts)
        else:
            res = await conn.create_market_sell_order(symbol=symbol, volume=lots, stop_loss=sl, options=opts)
        log.info("%s: %s %.2f lots, stop %.5f -> %s", symbol, side.upper(), lots, sl,
                 res.get("stringCode") if isinstance(res, dict) else res)
    actions.append((side, symbol, lots, sl))
    return actions


# ----------------------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------------------
async def run_pass(conn, account, cfg, state, now=None):
    now = now or datetime.now(timezone.utc)
    info = await conn.get_account_information()
    if info.get("type") == "real" and not cfg.allow_real:
        raise SystemExit("REAL account detected. Set ALLOW_REAL=1 to trade it (demo first!).")
    if not info.get("tradeAllowed", True):
        log.warning("Trading is not allowed on this account right now")

    equity, balance = float(info["equity"]), float(info["balance"])
    today = now.date().isoformat()
    if state.get("day") != today:
        state["day"], state["day_start_equity"] = today, equity
    lim = 1 - cfg.daily_loss_pct / 100.0
    halted = cfg.daily_loss_pct > 0 and (equity < state["day_start_equity"] * lim or equity < balance * lim)
    if cfg.min_equity and equity < cfg.min_equity:
        halted = True
    log.info("Account %s equity=%.2f balance=%.2f halted=%s live=%s", info.get("type"), equity, balance,
             halted, cfg.live)

    actions = []
    for symbol in cfg.symbols:
        try:
            actions += await process_symbol(conn, account, symbol, cfg, equity, halted, now,
                                            state.setdefault("last_seen", {}))
        except Exception as e:  # one bad symbol must not stop the others
            log.exception("%s: error: %s", symbol, e)
    return actions


async def run(cfg, once):
    from metaapi_cloud_sdk import MetaApi  # imported here so the logic can be tested offline

    token, account_id = os.environ["METAAPI_TOKEN"], os.environ["METAAPI_ACCOUNT_ID"]
    api = MetaApi(token)
    state = {}
    while True:
        try:
            account = await api.metatrader_account_api.get_account(account_id)
            if account.state != "DEPLOYED":
                await account.deploy()
            if account.connection_status != "CONNECTED":
                log.info("Waiting for MetaApi to connect to the broker (can take a few minutes)...")
                await account.wait_connected()
            conn = account.get_rpc_connection()
            await conn.connect()
            await conn.wait_synchronized()
            try:
                await run_pass(conn, account, cfg, state)
            finally:
                try:
                    await conn.close()
                except Exception:
                    pass
        except SystemExit:
            raise
        except Exception as e:
            log.exception("Pass failed: %s", e)
        if once:
            return
        poll = min(TF_SECONDS[cfg.timeframe], 3600)
        nxt = (math.floor(time.time() / poll) + 1) * poll + cfg.delay_seconds
        log.info("Sleeping %.0f s until next check", nxt - time.time())
        await asyncio.sleep(max(1, nxt - time.time()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="run a single pass and exit")
    args = ap.parse_args()
    cfg = BotConfig.from_env()
    cfg.validate()
    log.info("Starting: %s %s live=%s direction=%s risk=%.2f%%", cfg.symbols, cfg.timeframe, cfg.live,
             cfg.direction, cfg.risk_percent)
    asyncio.run(run(cfg, args.once))
