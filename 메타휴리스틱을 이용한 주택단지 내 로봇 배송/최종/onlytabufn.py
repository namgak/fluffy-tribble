import math
import json
import time
from collections import deque, defaultdict

import matplotlib.pyplot as plt
import matplotlib.animation as animation


# =============================================================================
# 1) Config
# =============================================================================
class Config:
    RUN_DURATION_MIN = 900
    ORDER_CUTOFF_MIN = 870
    ORDER_PROB_PER_MIN = 0.10

    ROBOT_COUNT = 15
    ROBOT_SPEED_KMH = 5.0
    ROBOT_SPEED_MPM = (ROBOT_SPEED_KMH * 1000) / 60.0

    ROBOT_CAPACITY = 10
    BATTERY_MAX_MINUTES = 6 * 60
    BATTERY_THRESH = 0.1
    TIME_SWAP = 3
    TIME_PICKUP = 1
    TIME_DELIVERY = 1

    VOL_COFFEE = 1
    VOL_FLOWER = 3
    VOL_BOOK = 2

    C_THRESH = 2
    TW = 15

    # ----------------------------
    # Objective weights
    # ----------------------------
    OBJ_W_DELAY = 120.0
    OBJ_W_TRAVEL = 1.0
    OBJ_W_ROUTE_LEN = 0.4
    OBJ_PENALTY_BATT = 1e6
    OBJ_PENALTY_CAP = 1e6

    # ----------------------------
    # Tabu parameters
    # ----------------------------
    N_CANDIDATE = 5
    I_LOCAL = 70
    TABU_SIZE = 20
    NEIGHBOR_SAMPLES = 18

    # ----------------------------
    # Insertion optimization
    # ----------------------------
    INSERTION_SAMPLES = 15  # (pickup_idx, delivery_idx) 후보 샘플링 개수

    # ----------------------------
    # Animation
    # ----------------------------
    TRAIL_MAX_POINTS = 80
    LOG_LINES_RECENT = 16
    SUMMARY_MAX_LINES = 10

    # Seeds (multi-seed headless, animate only seed=42)
    SEEDS = [42, 43, 44, 45, 46]
    ANIMATE_SEED = 42

    OUT_JSON = "tabu_only_runs.json"


# =============================================================================
# 2) Nodes
# =============================================================================
NODES = {
    "N_0": (80, 250),
    "S_1": (40, 240),
    "S_2": (350, 180),
    "S_3": (580, 80),

    "A_201": (20, 100), "A_202": (60, 90), "A_203": (90, 40), "A_209": (60, 110),
    "A_601": (320, 260), "A_602": (300, 260), "A_603": (320, 290),
    "A_604": (320, 320), "A_605": (320, 350),
    "A_701": (420, 270), "A_702": (450, 260), "A_704": (550, 260),

    "A_307": (250, 180), "A_308": (200, 200), "A_309": (200, 230),
    "A_310": (250, 230), "A_311": (300, 200), "A_306": (320, 180),
    "A_305": (400, 180), "A_301": (450, 210), "A_302": (500, 210),
    "A_303": (550, 210), "A_304": (550, 180),

    "A_101": (220, 130), "A_102": (300, 120), "A_103": (400, 100),
    "A_104": (500, 100), "A_105": (600, 120),
    "A_204": (150, 50), "A_205": (300, 20), "A_206": (400, 60),
    "A_207": (250, 80), "A_210": (650, 60), "A_211": (750, 80),
    "A_212": (720, 120), "A_213": (680, 100)
}

APT_NODES = [k for k in NODES.keys() if k.startswith("A_")]


def minute_to_clock_str(minute_from_0700: int):
    total = 7 * 60 + minute_from_0700
    hh = (total // 60) % 24
    mm = total % 60
    return f"{hh:02d}:{mm:02d}"


# =============================================================================
# 3) RNG wrapper (seed-fixed)
# =============================================================================
class RNG:
    def __init__(self, seed: int):
        import random
        self.r = random.Random(seed)

    def random(self):
        return self.r.random()

    def randint(self, a, b):
        return self.r.randint(a, b)

    def choice(self, seq):
        return self.r.choice(seq)

    def sample(self, seq, k):
        return self.r.sample(seq, k)


# =============================================================================
# 4) Core classes
# =============================================================================
class Item:
    def __init__(self, item_id, order_id, item_type, volume, pickup_node, delivery_node, create_time):
        self.id = item_id
        self.order_id = order_id
        self.type = item_type
        self.volume = volume
        self.pickup_node = pickup_node
        self.delivery_node = delivery_node
        self.create_time = create_time
        self.due_time = create_time + Config.TW
        self.status = "WAITING"
        self.assigned_robot_id = None
        self.arrival_time = None

    def lead_time(self):
        if self.arrival_time is None:
            return None
        return float(max(0.0, self.arrival_time - self.create_time))

    def delay(self):
        if self.arrival_time is None:
            return None
        return float(max(0.0, self.arrival_time - self.due_time))


class Robot:
    def __init__(self, r_id, color=None):
        self.id = r_id
        self.current_node = "N_0"
        self.pos = list(NODES["N_0"])
        self.battery = Config.BATTERY_MAX_MINUTES
        self.current_load = 0

        self.route = []
        self.items_on_board = []
        self.assigned_items = []

        self.state = "IDLE"
        self.dist_to_next = 0.0
        self.swap_count = 0

        self.color = color
        self.trail = [tuple(self.pos)]


# =============================================================================
# 5) Basic helpers
# =============================================================================
def check_feasibility(robot, new_item):
    if robot.battery < Config.BATTERY_MAX_MINUTES * Config.BATTERY_THRESH:
        return False
    if robot.current_load + new_item.volume > Config.ROBOT_CAPACITY:
        return False
    return True

def is_pure_return_route(route):
    if not route:
        return False
    for t in route:
        if t.get("type") not in ("RETURN", "WAIT"):
            return False
    return True

def cancel_return_if_needed(robot, log_stack, current_time, reason=""):
    if is_pure_return_route(robot.route):
        robot.route.clear()
        if reason:
            log_stack.append(f"[RETURN_CANCEL] t={int(current_time):>3} | R{robot.id} ({reason})")
        else:
            log_stack.append(f"[RETURN_CANCEL] t={int(current_time):>3} | R{robot.id}")


# =============================================================================
# 6) Objective (projected)
# =============================================================================
def simulate_route_projection(robot: Robot, current_time: float, route_override=None):
    route = robot.route if route_override is None else route_override

    curr_pos = robot.pos[:]
    t = float(current_time)
    total_travel = 0.0
    total_service = 0.0
    delivered_pred = {}

    for task in route:
        nxt = task["node"]
        nx, ny = NODES[nxt]
        dist = math.sqrt((nx - curr_pos[0])**2 + (ny - curr_pos[1])**2) * 1.2
        travel_min = dist / Config.ROBOT_SPEED_MPM
        t += travel_min
        total_travel += travel_min
        curr_pos = [nx, ny]

        st = float(task.get("service_time", 0))
        if st > 0:
            t += st
            total_service += st

        if task["type"] == "DELIVERY" and task.get("item") is not None:
            it = task["item"]
            delivered_pred[it.id] = t

    projected_need = total_travel + total_service
    batt_ok = (projected_need <= robot.battery + 1e-9)
    return total_travel, delivered_pred, batt_ok

def objective_total_cost(robots, current_time: float):
    total_delay = 0.0
    total_travel = 0.0
    total_route_len = 0.0
    penalty = 0.0

    for r in robots:
        travel, delivered_pred, batt_ok = simulate_route_projection(r, current_time)
        total_travel += travel
        total_route_len += len(r.route)

        if not batt_ok:
            penalty += Config.OBJ_PENALTY_BATT
        if r.current_load > Config.ROBOT_CAPACITY:
            penalty += Config.OBJ_PENALTY_CAP

        for task in r.route:
            if task["type"] == "DELIVERY" and task.get("item") is not None:
                it = task["item"]
                arr = delivered_pred.get(it.id, None)
                if arr is None:
                    continue
                total_delay += max(0.0, arr - it.due_time)

    return (
        Config.OBJ_W_DELAY * total_delay
        + Config.OBJ_W_TRAVEL * total_travel
        + Config.OBJ_W_ROUTE_LEN * total_route_len
        + penalty
    )


# =============================================================================
# 7) Insertion optimization (sampled best insertion)
# =============================================================================
def make_pickup_task(item: Item):
    return {"node": item.pickup_node, "type": "PICKUP", "item": item, "service_time": Config.TIME_PICKUP}

def make_delivery_task(item: Item):
    return {"node": item.delivery_node, "type": "DELIVERY", "item": item, "service_time": Config.TIME_DELIVERY}

def insert_pickup_delivery(route, item: Item, pickup_idx: int, delivery_idx_after_pickup: int):
    L = len(route)
    if pickup_idx < 0 or pickup_idx > L:
        raise ValueError("pickup_idx out of range")

    r1 = route[:pickup_idx] + [make_pickup_task(item)] + route[pickup_idx:]
    L2 = len(r1)

    if delivery_idx_after_pickup < 0 or delivery_idx_after_pickup > L2:
        raise ValueError("delivery_idx out of range")
    if delivery_idx_after_pickup <= pickup_idx:
        raise ValueError("delivery must be after pickup")

    r2 = r1[:delivery_idx_after_pickup] + [make_delivery_task(item)] + r1[delivery_idx_after_pickup:]
    return r2

def sampled_best_insertion_route(item: Item, target_robot: Robot, robots, current_time: float, rng: RNG,
                                samples: int = None):
    if samples is None:
        samples = Config.INSERTION_SAMPLES

    old_route = target_robot.route
    L = len(old_route)

    cand_pairs = set()
    cand_pairs.add((L, L + 1))  # always include append

    max_trials = samples * 8
    trials = 0
    while len(cand_pairs) < (samples + 1) and trials < max_trials:
        trials += 1
        p = rng.randint(0, L)
        d = rng.randint(p + 1, L + 1)
        cand_pairs.add((p, d))

    best_cost = None
    best_route = None

    for (p, d) in cand_pairs:
        try:
            cand_route = insert_pickup_delivery(old_route, item, p, d)
        except ValueError:
            continue

        target_robot.route = cand_route
        cost = objective_total_cost(robots, current_time)

        if (best_cost is None) or (cost < best_cost):
            best_cost = cost
            best_route = cand_route

    target_robot.route = old_route

    if best_route is None:
        best_route = old_route + [make_pickup_task(item), make_delivery_task(item)]
        target_robot.route = best_route
        best_cost = objective_total_cost(robots, current_time)
        target_robot.route = old_route

    return best_route, float(best_cost)


# =============================================================================
# 8) Tabu Search (localized reassign + best insertion)
# =============================================================================
def get_candidate_robots_near_item(trigger_item, robots):
    pickup_loc = NODES[trigger_item.pickup_node]
    dists = []
    for r in robots:
        d = math.sqrt((r.pos[0] - pickup_loc[0])**2 + (r.pos[1] - pickup_loc[1])**2)
        dists.append((r, d))
    dists.sort(key=lambda x: x[1])
    return [x[0] for x in dists[:Config.N_CANDIDATE]]

def extract_unloaded_items_in_route(robot: Robot):
    items = []
    seen = set()
    for t in robot.route:
        if t["type"] == "PICKUP" and t.get("item") is not None:
            it = t["item"]
            if it.status != "LOADED" and it.id not in seen:
                seen.add(it.id)
                items.append(it)
    return items

def remove_item_tasks_from_route(route, item: Item):
    return [t for t in route if t.get("item") != item]

def run_localized_tabu_search(trigger_item, robots, log_stack, current_time, tabu_list: deque, rng: RNG):
    candidates = get_candidate_robots_near_item(trigger_item, robots)
    if len(candidates) < 2:
        return

    global_best_cost = objective_total_cost(robots, current_time)
    current_cost = global_best_cost

    for _ in range(Config.I_LOCAL):
        best_move = None
        best_cost = current_cost

        for _k in range(Config.NEIGHBOR_SAMPLES):
            r1, r2 = rng.sample(candidates, 2)
            movable_items = extract_unloaded_items_in_route(r1)
            if not movable_items:
                continue

            item_to_move = rng.choice(movable_items)

            move_key = (item_to_move.id, r1.id, r2.id)
            is_tabu = move_key in tabu_list

            old_r1 = r1.route
            old_r2 = r2.route

            r1.route = remove_item_tasks_from_route(old_r1, item_to_move)
            r2_best_route, r2_best_cost = sampled_best_insertion_route(
                item_to_move, r2, robots, current_time, rng, samples=Config.INSERTION_SAMPLES
            )
            cand_cost = r2_best_cost

            r1.route = old_r1
            r2.route = old_r2

            # tabu move는 원칙적으로 금지, 단 aspiration: global best 개선이면 허용
            if is_tabu and cand_cost >= global_best_cost:
                continue

            if cand_cost < best_cost:
                best_cost = cand_cost
                best_move = (item_to_move, r1, r2, r2_best_route, cand_cost, is_tabu)

        if best_move is None:
            break

        item_to_move, r1, r2, best_r2_route, cand_cost, was_tabu = best_move

        r1.route = remove_item_tasks_from_route(r1.route, item_to_move)
        if item_to_move in r1.assigned_items:
            r1.assigned_items.remove(item_to_move)

        cancel_return_if_needed(r2, log_stack, current_time, reason="tabu moved task in")

        r2.route = best_r2_route
        if item_to_move not in r2.assigned_items:
            r2.assigned_items.append(item_to_move)

        item_to_move.assigned_robot_id = r2.id
        tabu_list.append((item_to_move.id, r1.id, r2.id))

        current_cost = cand_cost
        if cand_cost < global_best_cost:
            global_best_cost = cand_cost

        log_stack.append(
            f"[TABU_REASSIGN] t={int(current_time):>3} | item#{item_to_move.id} "
            f"{item_to_move.type}({item_to_move.delivery_node}) R{r1.id}->{r2.id} | cost {cand_cost:.1f}"
            f"{' (asp)' if was_tabu else ''}"
        )


# =============================================================================
# 9) Assignment (Tabu policy only here)
# =============================================================================
def assign_order_tabu(new_items, robots, log_stack, current_time, timing_stats, tabu_list, rng):
    for item in new_items:
        start = time.perf_counter()

        candidates = []
        for r in robots:
            if not check_feasibility(r, item):
                continue
            travel, _, _ = simulate_route_projection(r, current_time)
            est = travel + len(r.route) * 0.01
            candidates.append((r, est))

        if not candidates:
            log_stack.append(f"[ASSIGN_FAIL] t={int(current_time):>3} | {item.type} -> {item.delivery_node}")
            timing_stats["event_ms"].append((time.perf_counter() - start) * 1000.0)
            continue

        candidates.sort(key=lambda x: x[1])
        best_robot = candidates[0][0]

        if len(candidates) > 1:
            first = candidates[0]
            second = candidates[1]
            if (second[1] - first[1]) < Config.C_THRESH:
                if len(second[0].assigned_items) < len(first[0].assigned_items):
                    best_robot = second[0]

        cancel_return_if_needed(best_robot, log_stack, current_time, reason="assigned new order")

        # best insertion on chosen robot
        best_route, _ = sampled_best_insertion_route(
            item, best_robot, robots, current_time, rng, samples=Config.INSERTION_SAMPLES
        )
        best_robot.route = best_route

        item.assigned_robot_id = best_robot.id
        item.status = "ASSIGNED"
        best_robot.assigned_items.append(item)

        log_stack.append(f"[ASSIGN] t={int(current_time):>3} | {item.type} -> {item.delivery_node} | R{best_robot.id}")

        # localized tabu improvement
        run_localized_tabu_search(item, robots, log_stack, current_time, tabu_list, rng)

        timing_stats["event_ms"].append((time.perf_counter() - start) * 1000.0)


# =============================================================================
# 10) Order event stream generator (seed-fixed)
# =============================================================================
def make_order_event_stream(seed: int):
    rng = RNG(seed)

    events = []
    order_counter = 0
    item_id_counter = 1

    for t in range(1, Config.RUN_DURATION_MIN + 1):
        if t > Config.ORDER_CUTOFF_MIN:
            continue
        if rng.random() > Config.ORDER_PROB_PER_MIN:
            continue

        order_counter += 1
        order_id = 100000 + order_counter

        r = rng.random()
        if r < 0.65:
            item_type = "Coffee"
            vol = Config.VOL_COFFEE
            shop = "S_3"
            qty = rng.choice([1, 1, 2, 2, 3])
        elif r < 0.90:
            item_type = "Book"
            vol = Config.VOL_BOOK
            shop = "S_2"
            qty = rng.choice([1, 1, 2])
        else:
            item_type = "Flower"
            vol = Config.VOL_FLOWER
            shop = "S_1"
            qty = 1

        target_apt = rng.choice(APT_NODES)

        specs = []
        for _ in range(qty):
            specs.append({
                "item_id": item_id_counter,
                "order_id": order_id,
                "type": item_type,
                "vol": vol,
                "pickup": shop,
                "delivery": target_apt,
                "create_time": t,
            })
            item_id_counter += 1

        events.append({"t": t, "order_id": order_id, "specs": specs})

    return events


# =============================================================================
# 11) Movement rules
# =============================================================================
def maybe_schedule_battery_swap(robot: Robot, current_time, log_stack):
    if robot.battery < Config.BATTERY_MAX_MINUTES * Config.BATTERY_THRESH and not robot.route:
        if robot.current_node != "N_0":
            robot.route.append({"node": "N_0", "type": "RETURN", "item": None, "service_time": 0})
            log_stack.append(f"[BATT] t={int(current_time):>3} | R{robot.id} low batt -> RETURN")
        else:
            robot.route.append({"node": "N_0", "type": "SWAP", "item": None, "service_time": Config.TIME_SWAP})

def maybe_schedule_return_to_depot(robot: Robot, current_time, log_stack):
    if robot.route:
        return
    if robot.current_node == "N_0":
        return
    robot.route.append({"node": "N_0", "type": "RETURN", "item": None, "service_time": 0})
    log_stack.append(f"[RETURN] t={int(current_time):>3} | R{robot.id} idle -> N_0")


# =============================================================================
# 12) Metrics
# =============================================================================
def p95(arr):
    if not arr:
        return 0.0
    s = sorted(arr)
    return float(s[int(0.95 * (len(s) - 1))])

def summarize(policy_name, robots, items, timing):
    delivered = [it for it in items if it.status == "DELIVERED" and it.arrival_time is not None]
    lead_times = [it.lead_time() for it in delivered if it.lead_time() is not None]
    delays = [it.delay() for it in delivered if it.delay() is not None]

    total_items = len(items)
    delivered_items = len(delivered)
    undelivered_items = len([it for it in items if it.status != "DELIVERED"])

    on_time = len([it for it in delivered if it.arrival_time <= it.due_time])
    on_time_rate = (on_time / delivered_items * 100.0) if delivered_items > 0 else 0.0

    swaps = sum(r.swap_count for r in robots)

    throughput = defaultdict(int)
    for it in delivered:
        throughput[it.assigned_robot_id] += 1
    thr_list = [throughput[i] for i in range(Config.ROBOT_COUNT)]
    thr_mean = sum(thr_list) / len(thr_list) if thr_list else 0.0
    thr_var = sum((x - thr_mean) ** 2 for x in thr_list) / len(thr_list) if thr_list else 0.0
    thr_std = math.sqrt(thr_var)

    ev = timing["event_ms"]
    avg_evt = sum(ev) / len(ev) if ev else 0.0
    p95_evt = p95(ev) if ev else 0.0

    return {
        "policy": policy_name,
        "total_items": total_items,
        "delivered_items": delivered_items,
        "undelivered_items": undelivered_items,
        "on_time_rate_pct": round(on_time_rate, 2),
        "avg_delay_min": round(sum(delays) / len(delays), 3) if delays else 0.0,
        "max_delay_min": round(max(delays), 3) if delays else 0.0,
        "p95_lead_time_min": round(p95(lead_times), 3) if lead_times else 0.0,
        "max_lead_time_min": round(max(lead_times), 3) if lead_times else 0.0,
        "battery_swaps": swaps,
        "robot_throughput_std": round(thr_std, 3),
        "avg_event_compute_ms": round(avg_evt, 3),
        "p95_event_compute_ms": round(p95_evt, 3),
    }, thr_list


# =============================================================================
# 13) Headless Tabu simulation (summary only)
# =============================================================================
def run_tabu_headless(events, seed):
    rng_policy = RNG(seed + 999)

    robots = [Robot(i) for i in range(Config.ROBOT_COUNT)]
    items = []
    timing = {"event_ms": []}
    log = []
    tabu_list = deque(maxlen=Config.TABU_SIZE)

    idx = 0
    for t in range(1, Config.RUN_DURATION_MIN + 1):
        while idx < len(events) and events[idx]["t"] == t:
            ev = events[idx]
            idx += 1

            new_items = []
            for sp in ev["specs"]:
                it = Item(sp["item_id"], sp["order_id"], sp["type"], sp["vol"], sp["pickup"], sp["delivery"], sp["create_time"])
                new_items.append(it)
                items.append(it)

            assign_order_tabu(new_items, robots, log, t, timing, tabu_list, rng_policy)

        for r in robots:
            maybe_schedule_battery_swap(r, t, log)
            maybe_schedule_return_to_depot(r, t, log)

            if not r.route:
                continue

            target_node = r.route[0]["node"]
            tx, ty = NODES[target_node]
            dx = tx - r.pos[0]
            dy = ty - r.pos[1]
            dist = math.sqrt(dx * dx + dy * dy)

            move_dist = Config.ROBOT_SPEED_MPM
            if dist <= move_dist:
                r.pos = [tx, ty]
                r.current_node = target_node
                task = r.route.pop(0)

                st = task.get("service_time", 0)
                if st > 0:
                    for _ in range(int(st)):
                        r.route.insert(0, {"node": r.current_node, "type": "WAIT", "item": None, "service_time": 0})

                if task["type"] == "PICKUP":
                    it = task["item"]
                    it.status = "LOADED"
                    r.current_load += it.volume
                    r.items_on_board.append(it)

                elif task["type"] == "DELIVERY":
                    it = task["item"]
                    it.status = "DELIVERED"
                    it.arrival_time = t
                    r.current_load -= it.volume
                    if it in r.items_on_board:
                        r.items_on_board.remove(it)

                elif task["type"] == "SWAP":
                    r.swap_count += 1
                    r.battery = Config.BATTERY_MAX_MINUTES
            else:
                ratio = move_dist / dist
                r.pos[0] += dx * ratio
                r.pos[1] += dy * ratio

            r.battery -= 1
            if r.battery < 0:
                r.battery = 0

    summary, thr = summarize("Tabu(best-insertion)", robots, items, timing)
    return summary, thr


# =============================================================================
# 14) Animation (figure1) - seed=42 only, with live KPI panels + log
# =============================================================================
def run_tabu_animation(events, seed):
    fig1 = plt.figure(figsize=(16, 8))
    outer = fig1.add_gridspec(1, 2, width_ratios=[2.4, 1.25], wspace=0.15)

    ax_map = fig1.add_subplot(outer[0, 0])

    right = outer[0, 1].subgridspec(3, 1, height_ratios=[1.0, 1.0, 1.55], hspace=0.12)
    ax_kpi1 = fig1.add_subplot(right[0, 0])
    ax_kpi2 = fig1.add_subplot(right[1, 0])
    ax_text = fig1.add_subplot(right[2, 0])

    # map
    ax_map.set_xlim(0, 800)
    ax_map.set_ylim(0, 400)
    ax_map.set_title(f"Tabu Policy Animation (seed={seed})")

    for nid, (x, y) in NODES.items():
        if nid == "N_0":
            ax_map.scatter(x, y, c='red', marker='s', s=140)
            ax_map.text(x, y + 8, nid, fontsize=9, ha='center')
        elif nid.startswith("S"):
            ax_map.scatter(x, y, c='orange', marker='D', s=95)
            ax_map.text(x, y + 8, nid, fontsize=8, ha='center')
        else:
            ax_map.scatter(x, y, c='skyblue', marker='o', s=25, alpha=0.6)
    ax_map.grid(True, linestyle="--", alpha=0.25)

    cmap = plt.cm.get_cmap("tab20", Config.ROBOT_COUNT)
    robot_colors = [cmap(i) for i in range(Config.ROBOT_COUNT)]

    rng_policy = RNG(seed + 999)
    robots = [Robot(i, robot_colors[i]) for i in range(Config.ROBOT_COUNT)]
    items = []
    timing = {"event_ms": []}
    tabu_list = deque(maxlen=Config.TABU_SIZE)

    persistent_event_log = []
    persistent_summary_lines = []

    persistent_event_log.append(f"=== START (Tabu, seed={seed}) ===")
    persistent_event_log.append(f"[INFO] 운영시간 07:00~22:00 (900분), 주문마감 21:30 (870분)")
    persistent_event_log.append(f"[INFO] Tabu uses best-insertion (sample={Config.INSERTION_SAMPLES})")

    robot_scatter = ax_map.scatter([], [], s=55, marker='^')
    trail_lines = []
    for i in range(Config.ROBOT_COUNT):
        (line,) = ax_map.plot([], [], linewidth=2.0, alpha=0.75, color=robot_colors[i])
        trail_lines.append(line)

    hud_text = ax_map.text(
        0.02, 0.97, "", transform=ax_map.transAxes, va="top",
        bbox=dict(boxstyle="round", fc="white", alpha=0.9)
    )

    # text panel
    ax_text.set_title("Summary (persistent) + Event Log (live)")
    ax_text.axis("off")
    summary_text = ax_text.text(
        0.02, 0.98, "", transform=ax_text.transAxes, va="top",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round", fc="#eef6ff", alpha=1.0)
    )
    event_text = ax_text.text(
        0.02, 0.52, "", transform=ax_text.transAxes, va="top",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round", fc="#f7f7f7", alpha=1.0)
    )

    # KPI1: On-time + Avg delay
    ax_kpi1.set_title("Live KPI-1 (cumulative): On-time & Avg delay")
    ax_kpi1.set_xlabel("t (min from 07:00)")
    ax_kpi1.set_ylabel("On-time rate (%)")
    ax_kpi1.grid(True, linestyle="--", alpha=0.3)
    ax_kpi1b = ax_kpi1.twinx()
    ax_kpi1b.set_ylabel("Avg delay (min)")

    (kpi1_line_ontime,) = ax_kpi1.plot([], [], linewidth=2.0, label="On-time (%)")
    (kpi1_line_avgdelay,) = ax_kpi1b.plot([], [], linewidth=2.0, label="Avg delay (min)")
    ax_kpi1.legend([kpi1_line_ontime, kpi1_line_avgdelay],
                   ["On-time (%)", "Avg delay (min)"], loc="upper right")

    # KPI2: Delivered count + p95 lead
    ax_kpi2.set_title("Live KPI-2 (cumulative): Delivered & p95 lead time")
    ax_kpi2.set_xlabel("t (min from 07:00)")
    ax_kpi2.set_ylabel("Delivered count")
    ax_kpi2.grid(True, linestyle="--", alpha=0.3)
    ax_kpi2b = ax_kpi2.twinx()
    ax_kpi2b.set_ylabel("p95 lead time (min)")

    (kpi2_line_delivered,) = ax_kpi2.plot([], [], linewidth=2.0, label="Delivered")
    (kpi2_line_p95lead,) = ax_kpi2b.plot([], [], linewidth=2.0, label="p95 lead (min)")
    ax_kpi2.legend([kpi2_line_delivered, kpi2_line_p95lead],
                   ["Delivered", "p95 lead (min)"], loc="upper right")

    # KPI buffers
    kpi_t = []
    kpi_on_time = []
    kpi_avg_delay = []
    kpi_delivered = []
    kpi_p95_lead = []

    idx = 0
    run_orders = []

    def compute_summary_line(t_now):
        delivered_list = [it for it in items if it.status == "DELIVERED" and it.arrival_time is not None]
        delays = [it.delay() for it in delivered_list if it.delay() is not None]
        avg_delay = sum(delays) / len(delays) if delays else 0.0
        max_delay = max(delays) if delays else 0.0
        swaps = sum(r.swap_count for r in robots)

        on_time = len([it for it in delivered_list if it.arrival_time <= it.due_time])
        on_time_rate = (on_time / len(delivered_list) * 100.0) if delivered_list else 0.0

        active_items = len([it for it in items if it.status != "DELIVERED"])
        return (
            f"t={int(t_now):>3} ({minute_to_clock_str(int(t_now))}) | "
            f"Active={active_items} Delivered={len(delivered_list)} | "
            f"OnTime={on_time_rate:.1f}% | avgDelay={avg_delay:.2f} maxDelay={max_delay:.1f} | swaps={swaps}"
        )

    def update(frame):
        nonlocal idx
        t = frame + 1

        # new orders at time t
        while idx < len(events) and events[idx]["t"] == t:
            ev = events[idx]
            idx += 1

            new_items = []
            for sp in ev["specs"]:
                it = Item(sp["item_id"], sp["order_id"], sp["type"], sp["vol"], sp["pickup"], sp["delivery"], sp["create_time"])
                new_items.append(it)
                items.append(it)

            rec = {
                "time_min": int(t),
                "time_clock": minute_to_clock_str(int(t)),
                "order_id": ev["order_id"],
                "item_type": new_items[0].type,
                "qty": len(new_items),
                "pickup": new_items[0].pickup_node,
                "delivery": new_items[0].delivery_node
            }
            run_orders.append(rec)

            persistent_event_log.append(
                f"[NEW] {rec['time_clock']} | O{rec['order_id']} | {rec['item_type']} x{rec['qty']} | "
                f"{rec['pickup']} -> {rec['delivery']}"
            )

            assign_order_tabu(new_items, robots, persistent_event_log, t, timing, tabu_list, rng_policy)

        if int(t) == Config.ORDER_CUTOFF_MIN + 1:
            persistent_event_log.append(f"[CUTOFF] {minute_to_clock_str(int(t))} 이후 신규 주문 마감")

        # move robots
        positions = []
        for r in robots:
            maybe_schedule_battery_swap(r, t, persistent_event_log)
            maybe_schedule_return_to_depot(r, t, persistent_event_log)

            if not r.route:
                positions.append(r.pos)
                r.trail.append(tuple(r.pos))
                if len(r.trail) > Config.TRAIL_MAX_POINTS:
                    r.trail.pop(0)
                continue

            target_node = r.route[0]["node"]
            tx, ty = NODES[target_node]
            dx = tx - r.pos[0]
            dy = ty - r.pos[1]
            dist = math.sqrt(dx * dx + dy * dy)

            move_dist = Config.ROBOT_SPEED_MPM
            if dist <= move_dist:
                r.pos = [tx, ty]
                r.current_node = target_node
                task = r.route.pop(0)

                st = task.get("service_time", 0)
                if st > 0:
                    for _ in range(int(st)):
                        r.route.insert(0, {"node": r.current_node, "type": "WAIT", "item": None, "service_time": 0})

                if task["type"] == "PICKUP":
                    it = task["item"]
                    it.status = "LOADED"
                    r.current_load += it.volume
                    r.items_on_board.append(it)

                elif task["type"] == "DELIVERY":
                    it = task["item"]
                    it.status = "DELIVERED"
                    it.arrival_time = t
                    r.current_load -= it.volume
                    if it in r.items_on_board:
                        r.items_on_board.remove(it)

                    lead = int(it.arrival_time - it.create_time)
                    delay = int(max(0.0, it.arrival_time - it.due_time))
                    persistent_event_log.append(
                        f"[DELIV] {minute_to_clock_str(int(t))} | O{it.order_id} "
                        f"{it.type} -> {it.delivery_node} | lead={lead}m delay={delay}m | R{r.id}"
                    )

                elif task["type"] == "SWAP":
                    r.swap_count += 1
                    r.battery = Config.BATTERY_MAX_MINUTES
                    persistent_event_log.append(f"[BATT] t={int(t):>3} | R{r.id} swap done (+1)")
            else:
                ratio = move_dist / dist
                r.pos[0] += dx * ratio
                r.pos[1] += dy * ratio

            r.battery -= 1
            if r.battery < 0:
                r.battery = 0

            positions.append(r.pos)
            r.trail.append(tuple(r.pos))
            if len(r.trail) > Config.TRAIL_MAX_POINTS:
                r.trail.pop(0)

        # draw
        robot_scatter.set_offsets(positions)
        robot_scatter.set_color([r.color for r in robots])
        for i, r in enumerate(robots):
            xs = [p[0] for p in r.trail]
            ys = [p[1] for p in r.trail]
            trail_lines[i].set_data(xs, ys)

        delivered_cnt = len([it for it in items if it.status == "DELIVERED"])
        active_cnt = len([it for it in items if it.status != "DELIVERED"])
        swaps_total = sum(r.swap_count for r in robots)
        evs = timing["event_ms"]
        avg_evt = sum(evs) / len(evs) if evs else 0.0

        hud_text.set_text(
            f"Tabu Policy (best-insertion) | seed={seed}\n"
            f"t={t}/{Config.RUN_DURATION_MIN} ({minute_to_clock_str(int(t))}) | cutoff=21:30(t={Config.ORDER_CUTOFF_MIN})\n"
            f"Active={active_cnt} Delivered={delivered_cnt} | Swaps={swaps_total} | Avg event compute={avg_evt:.2f} ms\n"
            f"Ncand={Config.N_CANDIDATE} I_local={Config.I_LOCAL} tabu={Config.TABU_SIZE} insSample={Config.INSERTION_SAMPLES}"
        )

        # KPI update
        delivered_list = [it for it in items if it.status == "DELIVERED" and it.arrival_time is not None]
        if delivered_list:
            on_time = len([it for it in delivered_list if it.arrival_time <= it.due_time])
            on_time_rate = (on_time / len(delivered_list)) * 100.0

            delays = [it.delay() for it in delivered_list if it.delay() is not None]
            avg_delay = (sum(delays) / len(delays)) if delays else 0.0

            lead_times = [it.lead_time() for it in delivered_list if it.lead_time() is not None]
            p95_lead = p95(lead_times) if lead_times else 0.0
        else:
            on_time_rate = 0.0
            avg_delay = 0.0
            p95_lead = 0.0

        kpi_t.append(t)
        kpi_on_time.append(on_time_rate)
        kpi_avg_delay.append(avg_delay)
        kpi_delivered.append(len(delivered_list))
        kpi_p95_lead.append(p95_lead)

        kpi1_line_ontime.set_data(kpi_t, kpi_on_time)
        kpi1_line_avgdelay.set_data(kpi_t, kpi_avg_delay)
        kpi2_line_delivered.set_data(kpi_t, kpi_delivered)
        kpi2_line_p95lead.set_data(kpi_t, kpi_p95_lead)

        xmax = min(Config.RUN_DURATION_MIN, max(60, t))
        ax_kpi1.set_xlim(0, xmax)
        ax_kpi2.set_xlim(0, xmax)

        ax_kpi1.set_ylim(0, 100)
        max_ad = max(kpi_avg_delay) if kpi_avg_delay else 1.0
        ax_kpi1b.set_ylim(0, max(1.0, max_ad * 1.15))

        ax_kpi2.set_ylim(0, max(10, (max(kpi_delivered) * 1.15) if kpi_delivered else 10))
        max_p95 = max(kpi_p95_lead) if kpi_p95_lead else 1.0
        ax_kpi2b.set_ylim(0, max(1.0, max_p95 * 1.15))

        if frame % 30 == 0:
            persistent_summary_lines.append(compute_summary_line(t))

        summary_lines = persistent_summary_lines[-Config.SUMMARY_MAX_LINES:] or ["(no summary yet)"]
        summary_text.set_text("SUMMARY (persistent)\n" + "\n".join(summary_lines))

        recent_events = persistent_event_log[-Config.LOG_LINES_RECENT:]
        event_text.set_text("EVENT LOG (recent)\n" + "\n".join(recent_events))

        return (
            [robot_scatter, hud_text, summary_text, event_text,
             kpi1_line_ontime, kpi1_line_avgdelay, kpi2_line_delivered, kpi2_line_p95lead]
            + trail_lines
        )

    ani = animation.FuncAnimation(
        fig1, update,
        frames=Config.RUN_DURATION_MIN,
        interval=60,
        blit=False,
        repeat=False
    )

    return fig1, ani, run_orders


# =============================================================================
# 15) Headless with records (for animation seed JSON dump)
# =============================================================================
def run_tabu_headless_with_records(events, seed):
    rng_policy = RNG(seed + 999)
    robots = [Robot(i) for i in range(Config.ROBOT_COUNT)]
    items = []
    timing = {"event_ms": []}
    log = []
    tabu_list = deque(maxlen=Config.TABU_SIZE)

    idx = 0
    for t in range(1, Config.RUN_DURATION_MIN + 1):
        while idx < len(events) and events[idx]["t"] == t:
            ev = events[idx]
            idx += 1

            new_items = []
            for sp in ev["specs"]:
                it = Item(sp["item_id"], sp["order_id"], sp["type"], sp["vol"], sp["pickup"], sp["delivery"], sp["create_time"])
                new_items.append(it)
                items.append(it)

            assign_order_tabu(new_items, robots, log, t, timing, tabu_list, rng_policy)

        for r in robots:
            maybe_schedule_battery_swap(r, t, log)
            maybe_schedule_return_to_depot(r, t, log)

            if not r.route:
                continue

            target_node = r.route[0]["node"]
            tx, ty = NODES[target_node]
            dx = tx - r.pos[0]
            dy = ty - r.pos[1]
            dist = math.sqrt(dx * dx + dy * dy)

            move_dist = Config.ROBOT_SPEED_MPM
            if dist <= move_dist:
                r.pos = [tx, ty]
                r.current_node = target_node
                task = r.route.pop(0)

                st = task.get("service_time", 0)
                if st > 0:
                    for _ in range(int(st)):
                        r.route.insert(0, {"node": r.current_node, "type": "WAIT", "item": None, "service_time": 0})

                if task["type"] == "PICKUP":
                    it = task["item"]
                    it.status = "LOADED"
                    r.current_load += it.volume
                    r.items_on_board.append(it)

                elif task["type"] == "DELIVERY":
                    it = task["item"]
                    it.status = "DELIVERED"
                    it.arrival_time = t
                    r.current_load -= it.volume
                    if it in r.items_on_board:
                        r.items_on_board.remove(it)

                elif task["type"] == "SWAP":
                    r.swap_count += 1
                    r.battery = Config.BATTERY_MAX_MINUTES
            else:
                ratio = move_dist / dist
                r.pos[0] += dx * ratio
                r.pos[1] += dy * ratio

            r.battery -= 1
            if r.battery < 0:
                r.battery = 0

    summ, thr = summarize("Tabu(best-insertion)", robots, items, timing)

    item_records = []
    for it in items:
        item_records.append({
            "item_id": it.id,
            "order_id": it.order_id,
            "item_type": it.type,
            "volume": it.volume,
            "pickup": it.pickup_node,
            "delivery": it.delivery_node,
            "create_time_min": float(it.create_time),
            "due_time_min": float(it.due_time),
            "assigned_robot_id": it.assigned_robot_id,
            "arrival_time_min": None if it.arrival_time is None else float(it.arrival_time),
            "lead_time_min": it.lead_time(),
            "delay_min": it.delay(),
            "status": it.status,
        })

    return summ, thr, item_records


# =============================================================================
# 16) Main
# =============================================================================
def main():
    print("=== Tabu-only multi-seed experiment ===")
    print(f"Seeds = {Config.SEEDS}")
    print(f"Insertion samples = {Config.INSERTION_SAMPLES}")
    print("Generating streams and running headless sims...")

    streams_by_seed = {}
    results_by_seed = []

    for seed in Config.SEEDS:
        events = make_order_event_stream(seed)
        streams_by_seed[seed] = events

        tabu_summary, tabu_thr = run_tabu_headless(events, seed)
        results_by_seed.append({
            "seed": seed,
            "tabu_summary": tabu_summary,
            "tabu_throughput": tabu_thr
        })

        print(
            f"- seed={seed} | ontime={tabu_summary['on_time_rate_pct']}% "
            f"| p95Lead={tabu_summary['p95_lead_time_min']} "
            f"| maxDelay={tabu_summary['max_delay_min']} "
            f"| avgEvtMs={tabu_summary['avg_event_compute_ms']}"
        )

    # animate only seed=42 (Config.ANIMATE_SEED)
    animate_seed = Config.ANIMATE_SEED
    if animate_seed not in streams_by_seed:
        raise ValueError(f"ANIMATE_SEED={animate_seed} not in SEEDS={Config.SEEDS}")

    events_anim = streams_by_seed[animate_seed]
    print(f"\nCreating animation for seed={animate_seed} (Tabu only)...")
    fig1, ani, anim_orders = run_tabu_animation(events_anim, animate_seed)

    # for JSON: detailed records for animation seed
    anim_summary, anim_thr, anim_item_records = run_tabu_headless_with_records(events_anim, animate_seed)

    payload = {
        "config": {k: getattr(Config, k) for k in dir(Config) if k.isupper()},
        "seeds": Config.SEEDS,
        "streams_by_seed": streams_by_seed,
        "results_by_seed": results_by_seed,
        "animation": {
            "seed": animate_seed,
            "policy": "Tabu(best-insertion)",
            "orders": anim_orders,
            "items": anim_item_records,
            "summary": anim_summary,
            "throughput": anim_thr
        }
    }

    with open(Config.OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved: {Config.OUT_JSON}")

    plt.show()


if __name__ == "__main__":
    main()
