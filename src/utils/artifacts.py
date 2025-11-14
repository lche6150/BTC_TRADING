from pathlib import Path

def _normalize_symbol(symbol: str) -> str:
    return symbol.lower()


def get_artifact_root(symbol: str) -> Path:
    return Path("models") / _normalize_symbol(symbol)


def get_gru_backtest_dir(symbol: str) -> Path:
    return get_artifact_root(symbol) / "gru_backtest"


def get_gru_signal_dir(symbol: str) -> Path:
    return get_artifact_root(symbol) / "gru_signal"
