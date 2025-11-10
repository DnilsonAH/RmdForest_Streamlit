import os
import matplotlib.pyplot as plt
import pandas as pd


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_basic_stats(df: pd.DataFrame, plot_dir: str):
    _ensure_dir(plot_dir)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Date"], df["cu_close"], label="Cobre Close", color="tab:blue")
    ax.set_ylabel("Cobre Close")
    ax.set_title("Precio de Cierre del Cobre")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "basic_stats.png"))
    plt.close(fig)
    return fig


def plot_indicators_sma(df: pd.DataFrame, plot_dir: str):
    _ensure_dir(plot_dir)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Date"], df["cu_close"], label="Close", color="black")
    ax.plot(df["Date"], df["cu_sma_5"], label="SMA 5", color="tab:green")
    ax.plot(df["Date"], df["cu_sma_10"], label="SMA 10", color="tab:red")
    ax.set_title("Cobre SMA 5/10")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "indicators_sma.png"))
    plt.close(fig)
    return fig


def plot_indicators_rsi(df: pd.DataFrame, plot_dir: str):
    _ensure_dir(plot_dir)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Date"], df["cu_rsi_14"], label="RSI 14", color="tab:purple")
    ax.axhline(30, color="gray", linestyle="--", linewidth=1)
    ax.axhline(70, color="gray", linestyle="--", linewidth=1)
    ax.set_title("RSI Cobre (14)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "indicators_rsi.png"))
    plt.close(fig)
    return fig