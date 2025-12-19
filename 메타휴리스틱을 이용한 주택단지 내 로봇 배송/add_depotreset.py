import math
import random
import json
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# =============================================================================
# 1) Config
# =============================================================================
class Config:
    RUNS = 3

    # 07:00 ~ 22:00 => 900분 운영
    RUN_DURATION_MIN = 900

    # 21:30 => 870분 이후 신규 주문 생성 금지
    ORDER_CUTOFF_MIN = 870

    ORDER_PROB_PER_MIN = 0.10

    ROBOT_COUNT = 15
    ROBOT_SPEED_KMH = 5.0
    ROBOT_SPEED_MPM = (ROBOT_SPEED_KMH * 1000) / 60.0  # 83.33 m/min

    ROBOT_CAPACITY = 10
    BATTERY_MAX_MINUTES = 6 * 60
    BATTERY_THRESH = 0.1
    TIME_SWAP = 3
    TIME_PICKUP = 1
    TIME_DELIVERY = 1

    VOL_COFFEE = 1
    VOL_FLOWER = 3
    VOL_BOOK = 2

    GAP_MAX = 5
    C_THRESH = 2
    TW = 15

    W1 = 80
    W2 = 15
    W3 = 5

    N_CANDIDATE = 5
    I_LOCAL = 50
    TABU_SIZE = 10

    TRAIL_MAX_POINTS = 80

    LOG_LINES_RECENT = 16
    SUMMARY_MAX_LINES = 10


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

def get_dist(n1_id, n2_id):
    p1 = NODES[n1_id]
    p2 = NODES[n2_id]
    dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    return dist * 1.2

APT_NODES = [k for k in NODES.keys() if k.startswith("A_")]


# =============================================================================
# 3) Classes
# =============================================================================
class Item:
    def __init__(self, order_id, item_type, volume, pickup_node, delivery_node, create_time):
        self.id = int(random.random() * 100000)
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

    def delay(self):
        if self.arrival_time is None:
            return None
        return max(0.0, self.arrival_time - self.due_time)

class Robot:
    def __init__(self, r_id, color):
        self.id = r_id
        self.current_node = "N_0"
        self.pos = list(NODES["N_0"])
        self.battery = Config.BATTERY_MAX_MINUTES
        self.current_load = 0

        self.route = []
        self.items_on_board = []
        self.assigned_items = []

        self.state = "IDLE"
        self.dist_to_next = 0

        self.swap_count = 0
        self.color = color

        self.trail = [tuple(self.pos)]

    def calculate_estimated_arrival_to_end(self):
        if not self.route:
            return 0.0
        time_accum = 0.0
        curr = self.current_node
        if self.state == "MOVING" and len(self.route) > 0:
            time_accum += self.dist_to_next / Config.ROBOT_SPEED_MPM
            curr = self.route[0]['node']
            start_idx = 1
        else:
            start_idx = 0

        for i in range(start_idx, len(self.route)):
            nxt = self.route[i]['node']
            time_accum += get_dist(curr, nxt) / Config.ROBOT_SPEED_MPM
            time_accum += self.route[i].get('service_time', 0)
            curr = nxt
        return time_accum


# =============================================================================
# 4) Algorithms + RETURN helpers
# =============================================================================
def check_feasibility(robot, new_item):
    if robot.battery < Config.BATTERY_MAX_MINUTES * Config.BATTERY_THRESH:
        return False
    if robot.current_load + new_item.volume > Config.ROBOT_CAPACITY:
        return False
    return True

def is_pure_return_route(route):
    # RETURN 또는 WAIT만 있는 경우를 "순수 복귀"로 간주
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

def run_localized_tabu_search(trigger_item, robots, log_stack, run_id, current_time):
    pickup_loc = NODES[trigger_item.pickup_node]
    dists = []
    for r in robots:
        d = math.sqrt((r.pos[0]-pickup_loc[0])**2 + (r.pos[1]-pickup_loc[1])**2)
        dists.append((r, d))
    dists.sort(key=lambda x: x[1])
    target_robots = [x[0] for x in dists[:Config.N_CANDIDATE]]

    for _ in range(Config.I_LOCAL):
        if len(target_robots) < 2:
            return
        r1, r2 = random.sample(target_robots, 2)
        movable = [t for t in r1.route if t['type'] == 'PICKUP' and t['item'].status != "LOADED"]
        if not movable:
            continue
        task = random.choice(movable)
        item_to_move = task['item']

        if random.random() < 0.10:
            old_r = r1.id
            new_r = r2.id

            r1.route = [t for t in r1.route if t['item'] != item_to_move]
            if item_to_move in r1.assigned_items:
                r1.assigned_items.remove(item_to_move)

            # ✅ RETURN 중 로봇이 local 범위에 들어와서 선택되면, 복귀 취소하고 새 일 받게 함
            cancel_return_if_needed(r2, log_stack, current_time, reason="tabu moved task in")

            r2.route.append({'node': item_to_move.pickup_node, 'type': 'PICKUP', 'item': item_to_move,
                             'service_time': Config.TIME_PICKUP})
            r2.route.append({'node': item_to_move.delivery_node, 'type': 'DELIVERY', 'item': item_to_move,
                             'service_time': Config.TIME_DELIVERY})
            r2.assigned_items.append(item_to_move)
            item_to_move.assigned_robot_id = r2.id

            log_stack.append(
                f"[TABU] t={int(current_time):>3} | {item_to_move.type}({item_to_move.delivery_node}) R{old_r} -> R{new_r}"
            )

def assign_order_greedy(new_items, robots, log_stack, run_id, current_time, timing_stats):
    for item in new_items:
        start = time.perf_counter()

        candidates = []
        for r in robots:
            if not check_feasibility(r, item):
                continue
            est = r.calculate_estimated_arrival_to_end()
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

        # ✅ RETURN 중인 로봇이 배정되면, 복귀 취소하고 작업 우선
        cancel_return_if_needed(best_robot, log_stack, current_time, reason="assigned new order")

        best_robot.route.append({'node': item.pickup_node, 'type': 'PICKUP', 'item': item,
                                 'service_time': Config.TIME_PICKUP})
        best_robot.route.append({'node': item.delivery_node, 'type': 'DELIVERY', 'item': item,
                                 'service_time': Config.TIME_DELIVERY})

        item.assigned_robot_id = best_robot.id
        item.status = "ASSIGNED"
        best_robot.assigned_items.append(item)

        log_stack.append(f"[ASSIGN] t={int(current_time):>3} | {item.type} -> {item.delivery_node} | R{best_robot.id}")
        run_localized_tabu_search(item, robots, log_stack, run_id, current_time)

        timing_stats["event_ms"].append((time.perf_counter() - start) * 1000.0)


# =============================================================================
# 5) Order generator (주문 마감 반영)
# =============================================================================
def make_order(current_time):
    if current_time > Config.ORDER_CUTOFF_MIN:
        return None, []

    if random.random() > Config.ORDER_PROB_PER_MIN:
        return None, []

    order_id = int(current_time * 1000) + random.randint(0, 99)

    r = random.random()
    if r < 0.65:
        item_type = "Coffee"
        vol = Config.VOL_COFFEE
        shop = "S_3"
        qty = random.choice([1, 1, 2, 2, 3])
    elif r < 0.90:
        item_type = "Book"
        vol = Config.VOL_BOOK
        shop = "S_2"
        qty = random.choice([1, 1, 2])
    else:
        item_type = "Flower"
        vol = Config.VOL_FLOWER
        shop = "S_1"
        qty = 1

    target_apt = random.choice(APT_NODES)
    items = [Item(order_id, item_type, vol, shop, target_apt, current_time) for _ in range(qty)]
    return order_id, items


# =============================================================================
# 6) Simulation + animation
# =============================================================================
def run_simulation_with_animation():
    all_runs_records = []
    all_runs_summaries = []

    cmap = plt.cm.get_cmap("tab20", Config.ROBOT_COUNT)
    robot_colors = [cmap(i) for i in range(Config.ROBOT_COUNT)]

    persistent_event_log = []
    persistent_summary_lines = []

    state = {
        "run_id": 1,
        "t": 0.0,
        "robots": [],
        "items": [],
        "run_orders": [],
        "timing": {"event_ms": []},
        "final_dump_done": False,  # ✅ run3 끝날 때 1번만 dump 보장
    }

    def reset_run(run_id: int):
        random.seed(100 + run_id)
        state["run_id"] = run_id
        state["t"] = 0.0
        state["robots"] = [Robot(i, robot_colors[i]) for i in range(Config.ROBOT_COUNT)]
        state["items"] = []
        state["run_orders"] = []
        state["timing"] = {"event_ms": []}
        # final_dump_done은 전체 실행에 대해 한 번만 쓰는 플래그이므로 reset 안 함

        persistent_event_log.append(f"=== RUN {run_id} START ===")
        persistent_event_log.append(f"[INFO] 운영시간 07:00~22:00 (900분), 주문마감 21:30 (870분)")

    def compute_run_summary():
        delivered = [it for it in state["items"] if it.status == "DELIVERED" and it.arrival_time is not None]
        delays = [max(0.0, it.arrival_time - it.due_time) for it in delivered]
        avg_delay = sum(delays) / len(delays) if delays else 0.0
        max_delay = max(delays) if delays else 0.0

        swaps = sum(r.swap_count for r in state["robots"])
        total_orders = len(state["run_orders"])
        total_items = len(state["items"])

        ev = state["timing"]["event_ms"]
        avg_event_ms = sum(ev) / len(ev) if ev else 0.0
        p95_event_ms = sorted(ev)[int(0.95 * (len(ev)-1))] if len(ev) >= 2 else (ev[0] if ev else 0.0)

        undelivered = len([it for it in state["items"] if it.status != "DELIVERED"])

        return {
            "run_id": state["run_id"],
            "total_orders": total_orders,
            "total_items": total_items,
            "delivered_items": len(delivered),
            "undelivered_items": undelivered,
            "avg_delay_min": round(avg_delay, 3),
            "max_delay_min": round(max_delay, 3),
            "battery_swaps": swaps,
            "avg_event_compute_ms": round(avg_event_ms, 3),
            "p95_event_compute_ms": round(p95_event_ms, 3),
        }

    def dump_run():
        # ✅ run3 마지막 프레임에서 2번 dump되는 것 방지
        # (혹시라도 외부에서 재호출될 수 있어 방어)
        if state["final_dump_done"] and state["run_id"] == Config.RUNS:
            return

        # orders + items 모두 저장 (✅ 추가)
        run_items = []
        for it in state["items"]:
            lead = None if it.arrival_time is None else float(it.arrival_time - it.create_time)
            delay = None if it.arrival_time is None else float(max(0.0, it.arrival_time - it.due_time))
            run_items.append({
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
                "lead_time_min": lead,
                "delay_min": delay,
                "status": it.status,
            })

        all_runs_records.append({
            "run_id": state["run_id"],
            "orders": state["run_orders"],
            "items": run_items,
        })

        summ = compute_run_summary()
        all_runs_summaries.append(summ)

        print(f"\n===== RUN {summ['run_id']} SUMMARY =====")
        print(f"total_orders={summ['total_orders']}, total_items={summ['total_items']}, delivered_items={summ['delivered_items']}, undelivered_items={summ['undelivered_items']}")
        print(f"avg_delay={summ['avg_delay_min']} min, max_delay={summ['max_delay_min']} min, battery_swaps={summ['battery_swaps']}")
        print(f"avg_event_compute={summ['avg_event_compute_ms']} ms, p95_event_compute={summ['p95_event_compute_ms']} ms")
        print("=====================================\n")

        persistent_summary_lines.append(
            f"RUN {summ['run_id']}: orders={summ['total_orders']} items={summ['total_items']} "
            f"deliv={summ['delivered_items']} undeliv={summ['undelivered_items']} "
            f"avgDelay={summ['avg_delay_min']} maxDelay={summ['max_delay_min']} swaps={summ['battery_swaps']} "
            f"avgEvtMs={summ['avg_event_compute_ms']} p95={summ['p95_event_compute_ms']}"
        )
        persistent_event_log.append(f"=== RUN {summ['run_id']} END ===")

        # run3 dump 완료 플래그
        if state["run_id"] == Config.RUNS:
            state["final_dump_done"] = True

    # init first run
    reset_run(1)

    # ---------------- figure layout ----------------
    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.4, 1.0])
    ax_map = fig.add_subplot(gs[0, 0])
    ax_log = fig.add_subplot(gs[0, 1])

    ax_map.set_xlim(0, 800)
    ax_map.set_ylim(0, 400)
    ax_map.set_title("Event-Driven Localized Tabu Search (3 Runs)")

    # draw nodes
    for nid, (x, y) in NODES.items():
        if nid == "N_0":
            ax_map.scatter(x, y, c='red', marker='s', s=140)
            ax_map.text(x, y+8, nid, fontsize=9, ha='center')
        elif nid.startswith("S"):
            ax_map.scatter(x, y, c='orange', marker='D', s=95)
            ax_map.text(x, y+8, nid, fontsize=8, ha='center')
        else:
            ax_map.scatter(x, y, c='skyblue', marker='o', s=25, alpha=0.6)

    robot_scatter = ax_map.scatter([], [], s=55, marker='^')
    trail_lines = []
    for i in range(Config.ROBOT_COUNT):
        (line,) = ax_map.plot([], [], linewidth=2.0, alpha=0.75, color=robot_colors[i])
        trail_lines.append(line)

    hud_text = ax_map.text(
        0.02, 0.97, "", transform=ax_map.transAxes, va="top",
        bbox=dict(boxstyle="round", fc="white", alpha=0.9)
    )
    ax_map.grid(True, linestyle="--", alpha=0.25)

    ax_log.set_title("Summary (persistent) + Event Log (live)")
    ax_log.axis("off")

    summary_text = ax_log.text(
        0.02, 0.98, "", transform=ax_log.transAxes, va="top",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round", fc="#eef6ff", alpha=1.0)
    )

    event_text = ax_log.text(
        0.02, 0.52, "", transform=ax_log.transAxes, va="top",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round", fc="#f7f7f7", alpha=1.0)
    )

    total_frames = Config.RUNS * Config.RUN_DURATION_MIN

    def maybe_schedule_battery_swap(robot: Robot, current_time):
        # 배터리 낮을 때 예외적 return/swap (그대로 유지)
        if robot.battery < Config.BATTERY_MAX_MINUTES * Config.BATTERY_THRESH and not robot.route:
            if robot.current_node != "N_0":
                robot.route.append({"node": "N_0", "type": "RETURN", "item": None, "service_time": 0})
                persistent_event_log.append(f"[BATT] t={int(current_time):>3} | R{robot.id} low batt -> RETURN")
            else:
                robot.route.append({"node": "N_0", "type": "SWAP", "item": None, "service_time": Config.TIME_SWAP})

    def maybe_schedule_return_to_depot(robot: Robot, current_time, log_stack):
        # 기본 정책: 일이 끝나서 route가 비고 N_0가 아니면 복귀
        if robot.route:
            return
        if robot.current_node == "N_0":
            return
        robot.route.append({"node": "N_0", "type": "RETURN", "item": None, "service_time": 0})
        log_stack.append(f"[RETURN] t={int(current_time):>3} | R{robot.id} idle -> N_0")

    def minute_to_clock_str(minute_from_0700: int):
        total = 7 * 60 + minute_from_0700
        hh = (total // 60) % 24
        mm = total % 60
        return f"{hh:02d}:{mm:02d}"

    def update(frame):
        run_index = frame // Config.RUN_DURATION_MIN
        minute_in_run = frame % Config.RUN_DURATION_MIN

        # run switch: run1->run2, run2->run3로 넘어갈 때만 dump
        if minute_in_run == 0:
            if frame != 0:
                dump_run()
            new_run_id = run_index + 1
            reset_run(new_run_id)

        # time
        state["t"] += 1.0

        # 1) new order
        order_id, new_items = make_order(state["t"])
        if new_items:
            rec = {
                "time_min": int(state["t"]),
                "time_clock": minute_to_clock_str(int(state["t"])),
                "order_id": order_id,
                "item_type": new_items[0].type,
                "qty": len(new_items),
                "pickup": new_items[0].pickup_node,
                "delivery": new_items[0].delivery_node
            }
            state["run_orders"].append(rec)
            persistent_event_log.append(
                f"[NEW] {rec['time_clock']} | O{order_id} | {rec['item_type']} x{rec['qty']} | "
                f"{rec['pickup']} -> {rec['delivery']}"
            )

            assign_order_greedy(
                new_items, state["robots"], persistent_event_log,
                state["run_id"], state["t"], state["timing"]
            )
            state["items"].extend(new_items)
        else:
            if int(state["t"]) == Config.ORDER_CUTOFF_MIN + 1:
                persistent_event_log.append(f"[CUTOFF] {minute_to_clock_str(int(state['t']))} 이후 신규 주문 마감")

        # 2) move robots
        positions = []
        for r in state["robots"]:
            maybe_schedule_battery_swap(r, state["t"])
            maybe_schedule_return_to_depot(r, state["t"], persistent_event_log)

            if not r.route:
                r.state = "IDLE"
                positions.append(r.pos)
                r.trail.append(tuple(r.pos))
                if len(r.trail) > Config.TRAIL_MAX_POINTS:
                    r.trail.pop(0)
                continue

            r.state = "MOVING"
            target_node = r.route[0]['node']
            tx, ty = NODES[target_node]

            dx = tx - r.pos[0]
            dy = ty - r.pos[1]
            dist = math.sqrt(dx*dx + dy*dy)

            move_dist = Config.ROBOT_SPEED_MPM
            if dist <= move_dist:
                r.pos = [tx, ty]
                r.current_node = target_node
                task = r.route.pop(0)

                st = task.get("service_time", 0)
                if st > 0:
                    for _ in range(int(st)):
                        r.route.insert(0, {"node": r.current_node, "type": "WAIT", "item": None, "service_time": 0})

                if task['type'] == 'PICKUP':
                    item = task['item']
                    item.status = "LOADED"
                    r.current_load += item.volume
                    r.items_on_board.append(item)

                elif task['type'] == 'DELIVERY':
                    item = task['item']
                    item.status = "DELIVERED"
                    item.arrival_time = state["t"]
                    r.current_load -= item.volume
                    if item in r.items_on_board:
                        r.items_on_board.remove(item)

                    # ✅ 추가 1) 배송 완료 로그(리드타임/지연) 기록
                    lead = int(item.arrival_time - item.create_time)
                    delay = int(max(0.0, item.arrival_time - item.due_time))
                    persistent_event_log.append(
                        f"[DELIV] {minute_to_clock_str(int(state['t']))} | O{item.order_id} "
                        f"{item.type} -> {item.delivery_node} | lead={lead}m delay={delay}m | R{r.id}"
                    )

                elif task['type'] == 'SWAP':
                    r.swap_count += 1
                    r.battery = Config.BATTERY_MAX_MINUTES
                    persistent_event_log.append(f"[BATT] t={int(state['t']):>3} | R{r.id} swap done (+1)")

            else:
                ratio = move_dist / dist
                r.pos[0] += dx * ratio
                r.pos[1] += dy * ratio
                r.dist_to_next = dist - move_dist

            # ⚠️ 배터리 감소 로직은 요청대로 기존 그대로 유지
            r.battery -= 1
            if r.battery < 0:
                r.battery = 0

            positions.append(r.pos)

            r.trail.append(tuple(r.pos))
            if len(r.trail) > Config.TRAIL_MAX_POINTS:
                r.trail.pop(0)

        # draw robots + trails
        robot_scatter.set_offsets(positions)
        robot_scatter.set_color([r.color for r in state["robots"]])

        for i, r in enumerate(state["robots"]):
            xs = [p[0] for p in r.trail]
            ys = [p[1] for p in r.trail]
            trail_lines[i].set_data(xs, ys)

        # HUD
        active_items = len([it for it in state["items"] if it.status != "DELIVERED"])
        delivered_items = len([it for it in state["items"] if it.status == "DELIVERED"])
        swaps_total = sum(r.swap_count for r in state["robots"])
        ev = state["timing"]["event_ms"]
        avg_evt = (sum(ev) / len(ev)) if ev else 0.0

        now_clock = minute_to_clock_str(int(state["t"]))
        hud_text.set_text(
            f"RUN {state['run_id']}/{Config.RUNS} | t={int(state['t'])}/{Config.RUN_DURATION_MIN} min ({now_clock})\n"
            f"Orders allowed until 21:30 (t={Config.ORDER_CUTOFF_MIN}) | Active={active_items} Delivered={delivered_items}\n"
            f"BatterySwaps={swaps_total} | Avg event compute={avg_evt:.2f} ms | Ncand={Config.N_CANDIDATE} I_local={Config.I_LOCAL}"
        )

        # log panel
        summary_lines = persistent_summary_lines[-Config.SUMMARY_MAX_LINES:]
        if not summary_lines:
            summary_lines = ["(no completed runs yet)"]
        summary_text.set_text("SUMMARY (persistent)\n" + "\n".join(summary_lines))

        recent_events = persistent_event_log[-Config.LOG_LINES_RECENT:]
        event_text.set_text("EVENT LOG (recent)\n" + "\n".join(recent_events))

        # ✅ 마지막 프레임: run3 결과까지 dump하고 저장하고 끝
        if frame == total_frames - 1:
            dump_run()  # run3 summary가 반드시 찍힘

            payload = {
                "config": {k: getattr(Config, k) for k in dir(Config) if k.isupper()},
                "runs": all_runs_records,
                "summaries": all_runs_summaries
            }
            with open("experiment_order_logs.json", "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print("✅ Saved: experiment_order_logs.json")

        return [robot_scatter, hud_text, summary_text, event_text] + trail_lines

    ani = animation.FuncAnimation(
        fig, update,
        frames=total_frames,
        interval=60,
        blit=False,
        repeat=False
    )
    plt.show()


if __name__ == "__main__":
    run_simulation_with_animation()
