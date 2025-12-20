import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 0) Load JSON
# =========================
JSON_PATH = Path("tabu_only_runs.json")  # 같은 폴더에 두는 걸 추천
# JSON_PATH = Path(r"C:\...\tabu_only_runs.json")  # 필요하면 절대경로로

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

results = data.get("results_by_seed", [])
if not results:
    raise ValueError("results_by_seed가 비어있습니다. JSON 구조를 확인하세요.")

# =========================
# 1) Seed-level KPI DataFrame
# =========================
rows = []
for r in results:
    seed = r["seed"]
    s = r["tabu_summary"]
    rows.append({
        "seed": seed,
        "total_items": s["total_items"],
        "delivered_items": s["delivered_items"],
        "undelivered_items": s["undelivered_items"],
        "on_time_rate_pct": s["on_time_rate_pct"],
        "avg_delay_min": s["avg_delay_min"],
        "max_delay_min": s["max_delay_min"],
        "p95_lead_time_min": s["p95_lead_time_min"],
        "max_lead_time_min": s["max_lead_time_min"],
        "battery_swaps": s["battery_swaps"],
        "robot_throughput_std": s["robot_throughput_std"],
        "avg_event_compute_ms": s["avg_event_compute_ms"],
        "p95_event_compute_ms": s["p95_event_compute_ms"],
    })

df = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)

print("\n=== Seed-level summary (Tabu only) ===")
print(df[[
    "seed", "on_time_rate_pct", "p95_lead_time_min", "max_delay_min",
    "avg_event_compute_ms", "battery_swaps", "robot_throughput_std"
]])
print("\n=== KPI aggregate stats ===")
print(df[[
    "on_time_rate_pct", "p95_lead_time_min", "max_delay_min",
    "avg_event_compute_ms", "battery_swaps", "robot_throughput_std"
]].describe().T)


def hist_plot(series, title, xlabel, bins=10):
    x = series.dropna().values
    plt.figure(figsize=(8, 4.8))
    plt.hist(x, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


def box_plot(series, title, ylabel):
    x = series.dropna().values
    plt.figure(figsize=(6.5, 4.8))
    plt.boxplot(x, vert=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


# =========================
# 2) Seed-level Histograms (핵심 KPI 분포)
# =========================
# (a) On-time rate
hist_plot(
    df["on_time_rate_pct"],
    title="Histogram: On-time rate across seeds (Tabu)",
    xlabel="on_time_rate (%)",
    bins=8
)
box_plot(df["on_time_rate_pct"], "Boxplot: On-time rate across seeds (Tabu)", "on_time_rate (%)")

# (b) p95 lead time
hist_plot(
    df["p95_lead_time_min"],
    title="Histogram: p95 lead time across seeds (Tabu)",
    xlabel="p95 lead time (min)",
    bins=8
)
box_plot(df["p95_lead_time_min"], "Boxplot: p95 lead time across seeds (Tabu)", "p95 lead time (min)")

# (c) max delay
hist_plot(
    df["max_delay_min"],
    title="Histogram: max delay across seeds (Tabu)",
    xlabel="max delay (min)",
    bins=8
)
box_plot(df["max_delay_min"], "Boxplot: max delay across seeds (Tabu)", "max delay (min)")

# (d) avg event compute time
hist_plot(
    df["avg_event_compute_ms"],
    title="Histogram: avg event compute time across seeds (Tabu)",
    xlabel="avg event compute (ms)",
    bins=8
)
box_plot(df["avg_event_compute_ms"], "Boxplot: avg event compute time across seeds (Tabu)", "avg event compute (ms)")


# =========================
# 3) Seed=42 item-level Histograms (애니메이션 로그 기반)
# =========================
anim = data.get("animation", None)
if anim is None:
    print("\n[WARN] animation 섹션이 없어서 seed=42 아이템 로그 히스토그램은 생략합니다.")
else:
    anim_seed = anim.get("seed", None)
    items = anim.get("items", [])
    if not items:
        print("\n[WARN] animation.items가 비어있어서 seed=42 아이템 로그 히스토그램은 생략합니다.")
    else:
        df_items = pd.DataFrame(items)

        # delivered only (안전)
        df_deliv = df_items[df_items["status"] == "DELIVERED"].copy()
        df_deliv["lead_time_min"] = pd.to_numeric(df_deliv["lead_time_min"], errors="coerce")
        df_deliv["delay_min"] = pd.to_numeric(df_deliv["delay_min"], errors="coerce")
        df_deliv["create_time_min"] = pd.to_numeric(df_deliv["create_time_min"], errors="coerce")
        df_deliv["arrival_time_min"] = pd.to_numeric(df_deliv["arrival_time_min"], errors="coerce")

        print(f"\n=== Item-level (animation seed={anim_seed}) delivered count: {len(df_deliv)} ===")
        print(df_deliv[["item_type", "lead_time_min", "delay_min"]].describe(include="all"))

        # (1) Lead time histogram
        plt.figure(figsize=(8, 4.8))
        plt.hist(df_deliv["lead_time_min"].dropna().values, bins=25)
        plt.title(f"Histogram: Lead time (min) | seed={anim_seed} (Tabu)")
        plt.xlabel("lead time (min)")
        plt.ylabel("count")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.show()

        # (2) Delay histogram (전체: 0 포함)
        plt.figure(figsize=(8, 4.8))
        plt.hist(df_deliv["delay_min"].fillna(0).values, bins=20)
        plt.title(f"Histogram: Delay (min, includes 0) | seed={anim_seed} (Tabu)")
        plt.xlabel("delay (min)")
        plt.ylabel("count")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.show()

        # (3) Delay histogram (양수만: ‘지연이 실제로 발생한 케이스’ 분포)
        pos = df_deliv[df_deliv["delay_min"] > 0]["delay_min"].dropna().values
        if len(pos) > 0:
            plt.figure(figsize=(8, 4.8))
            plt.hist(pos, bins=15)
            plt.title(f"Histogram: Positive delays only | seed={anim_seed} (Tabu)")
            plt.xlabel("delay (min) where delay>0")
            plt.ylabel("count")
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.tight_layout()
            plt.show()
        else:
            print("[INFO] seed=42에서 delay>0 케이스가 없습니다.")

        # (4) Create time distribution (언제 주문이 몰리는지)
        plt.figure(figsize=(8, 4.8))
        plt.hist(df_deliv["create_time_min"].dropna().values, bins=30)
        plt.title(f"Histogram: Order creation time (min from 07:00) | seed={anim_seed}")
        plt.xlabel("create_time (min from 07:00)")
        plt.ylabel("count")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.show()

        # (5) Arrival time distribution (언제 배송완료가 몰리는지)
        plt.figure(figsize=(8, 4.8))
        plt.hist(df_deliv["arrival_time_min"].dropna().values, bins=30)
        plt.title(f"Histogram: Delivery completion time (min from 07:00) | seed={anim_seed}")
        plt.xlabel("arrival_time (min from 07:00)")
        plt.ylabel("count")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.show()

        # (6) Item type별 lead time 분포 (겹쳐서 보기)
        #     Coffee/Book/Flower의 “난이도 차이(부피/픽업 위치/수요)”가 있으면 분포가 갈림
        types = sorted(df_deliv["item_type"].dropna().unique().tolist())
        plt.figure(figsize=(8, 4.8))
        for t in types:
            lt = df_deliv[df_deliv["item_type"] == t]["lead_time_min"].dropna().values
            if len(lt) > 0:
                plt.hist(lt, bins=20, alpha=0.5, label=t)
        plt.title(f"Histogram: Lead time by item type | seed={anim_seed}")
        plt.xlabel("lead time (min)")
        plt.ylabel("count")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.show()
