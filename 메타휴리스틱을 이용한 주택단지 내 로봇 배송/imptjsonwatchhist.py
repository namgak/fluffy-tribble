import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1) JSON 로드
# ============================================================
path = "experiment_order_logs.json"   # 같은 폴더에 있을 때
# path = "/mnt/data/experiment_order_logs.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

runs = data.get("runs", [])
items_rows = []
orders_rows = []

for r in runs:
    run_id = r["run_id"]
    for o in r.get("orders", []):
        oo = dict(o)
        oo["run_id"] = run_id
        orders_rows.append(oo)
    for it in r.get("items", []):
        ii = dict(it)
        ii["run_id"] = run_id
        items_rows.append(ii)

df_items = pd.DataFrame(items_rows)
df_orders = pd.DataFrame(orders_rows)

# ============================================================
# 2) 타입 정리
# ============================================================
for c in ["create_time_min", "due_time_min", "arrival_time_min",
          "lead_time_min", "delay_min"]:
    if c in df_items.columns:
        df_items[c] = pd.to_numeric(df_items[c], errors="coerce")

# 배송 완료 아이템
df_deliv = df_items[
    (df_items["status"] == "DELIVERED") &
    (df_items["arrival_time_min"].notna())
].copy()

def pctl(s, q):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    return float(np.percentile(s, q))

# ============================================================
# 3) run별 지표 계산
# ============================================================
metrics = []
for run_id, g in df_items.groupby("run_id"):
    g_del = g[
        (g["status"] == "DELIVERED") &
        (g["arrival_time_min"].notna())
    ]

    ontime_rate = (g_del["delay_min"] <= 0).mean() if len(g_del) else np.nan

    metrics.append({
        "run_id": run_id,
        "items_total": len(g),
        "items_delivered": len(g_del),
        "on_time_rate": ontime_rate,
        "lead_p95": pctl(g_del["lead_time_min"], 95),
        "lead_max": g_del["lead_time_min"].max(),
        "delay_p95": pctl(g_del["delay_min"], 95),
        "delay_max": g_del["delay_min"].max(),
    })

df_metrics = pd.DataFrame(metrics).sort_values("run_id")

print("\n=== RUN METRICS ===")
print(df_metrics.to_string(index=False))

# ============================================================
# 4) 시각화 ① run별 on-time rate(%)
# ============================================================
plt.figure()
plt.bar(
    df_metrics["run_id"].astype(str),
    df_metrics["on_time_rate"] * 100
)
plt.title("On-time delivery rate by run")
plt.xlabel("run")
plt.ylabel("on-time rate (%)")
plt.ylim(0, 100)
plt.show()

# ============================================================
# 5) 시각화 ② lead time p95 vs max (run별)
# ============================================================
plt.figure()
plt.plot(
    df_metrics["run_id"],
    df_metrics["lead_p95"],
    marker="o",
    label="lead time p95"
)
plt.plot(
    df_metrics["run_id"],
    df_metrics["lead_max"],
    marker="s",
    label="lead time max"
)
plt.title("Lead time p95 vs max by run")
plt.xlabel("run")
plt.ylabel("lead time (min)")
plt.legend()
plt.show()

# ============================================================
# 6) 시각화 ③ 로봇별 처리량 편차
#    (배송 완료 아이템 수 기준)
# ============================================================
robot_load = (
    df_deliv
    .groupby("assigned_robot_id")
    .size()
    .reset_index(name="delivered_items")
    .sort_values("assigned_robot_id")
)

plt.figure()
plt.bar(
    robot_load["assigned_robot_id"].astype(str),
    robot_load["delivered_items"]
)
plt.title("Delivered items per robot")
plt.xlabel("robot id")
plt.ylabel("number of delivered items")
plt.show()

# ============================================================
# 7) CSV 저장 (발표/보고서용)
# ============================================================
df_metrics.to_csv("analysis_run_metrics.csv", index=False)
robot_load.to_csv("analysis_robot_load.csv", index=False)

print("\nSaved:")
print(" - analysis_run_metrics.csv")
print(" - analysis_robot_load.csv")
