"""
ss.py — Animated Resource Allocation Monitor  (v2 – full color edition)
ncurses-based multi-scenario simulator with dynamic resources,
optimal scoring and defuse planning.

Controls: Q / ESC → quit   SPACE / RIGHT → skip to next scenario
"""

import curses
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# ── Timing ─────────────────────────────────────────────────────────────────────
SCENARIO_DURATION = 40.0
TICK_MS = 50
BAR_WIDTH = 28

# ── Color pair IDs ─────────────────────────────────────────────────────────────
C_NORMAL = 1
C_HEADER = 2
C_UNAL = 3  # Unallocated   – white
C_UNDER = 4  # Underutilized – yellow
C_OPTIMAL = 5  # Optimal       – green
C_OVER = 6  # Overutilized  – red
C_BORDER = 7  # borders       – blue
C_LABEL = 8  # labels        – white
C_TIMER = 9  # timer         – magenta
C_DEFUSE = 10  # defuse header – cyan
C_WARN = 11  # critical warn – red bold
C_TREND_UP = 12  # rising trend  – red
C_TREND_DN = 13  # falling trend – green
C_TREND_FL = 14  # flat trend    – dim
C_SHED = 15  # SHED action   – red
C_ACTIVATE = 16  # ACTIVATE      – green
C_BOOST = 17  # BOOST         – yellow
C_THROTTLE = 18  # THROTTLE      – magenta
C_SPARK = 19  # sparkline     – blue
C_THEME_R = 20  # theme red
C_THEME_Y = 21  # theme yellow
C_THEME_G = 22  # theme green
C_THEME_M = 23  # theme magenta
C_THEME_C = 24  # theme cyan
C_SCORE_LO = 25  # score low     – red on black
C_SCORE_MID = 26  # score mid     – yellow on black
C_SCORE_HI = 27  # score high    – green on black

STATUS_COLORS: Dict[str, int] = {
    "Unallocated"  : C_UNAL,
    "Underutilized": C_UNDER,
    "Optimal"      : C_OPTIMAL,
    "Overutilized" : C_OVER,
}

STATUS_ICONS: Dict[str, str] = {
    "Unallocated"  : "[ ]",
    "Underutilized": "[v]",
    "Optimal"      : "[*]",
    "Overutilized" : "[^]",
}

# mini overview chars
OVERVIEW_CHARS: Dict[str, str] = {
    "Unallocated"  : "U",
    "Underutilized": "v",
    "Optimal"      : "*",
    "Overutilized" : "^",
}

# sparkline levels (5 tiers)
SPARK_CHARS = " ._-^"

# ── Scenario templates ─────────────────────────────────────────────────────────
SCENARIO_TEMPLATES = [
    {
        "name"   : "CRISIS ZONE",
        "desc"   : "High stress · multiple overutilized units",
        "bias"   : +1.8, "volatility": 1.4,
        "theme_c": C_THEME_R,
        "units"  : [
            ("Alpha", 95.0), ("Beta", 88.0), ("Gamma", 12.0),
            ("Delta", 0.0), ("Epsilon", 75.0), ("Zeta", 92.0)
        ],
    },
    {
        "name"   : "IDLE SYSTEM",
        "desc"   : "Most resources unused · underutilization crisis",
        "bias"   : -1.6, "volatility": 0.8,
        "theme_c": C_THEME_Y,
        "units"  : [
            ("Node-1", 5.0), ("Node-2", 0.0), ("Node-3", 8.0),
            ("Node-4", 0.0), ("Node-5", 55.0), ("Node-6", 0.0), ("Node-7", 3.0)
        ],
    },
    {
        "name"   : "BALANCED STATE",
        "desc"   : "Near-optimal distribution across all units",
        "bias"   : 0.0, "volatility": 0.5,
        "theme_c": C_THEME_G,
        "units"  : [
            ("Sector-A", 65.0), ("Sector-B", 70.0), ("Sector-C", 60.0),
            ("Sector-D", 55.0), ("Sector-E", 72.0)
        ],
    },
    {
        "name"   : "MIXED WARFARE",
        "desc"   : "Chaotic allocation · partial defuse active",
        "bias"   : +0.4, "volatility": 2.0,
        "theme_c": C_THEME_M,
        "units"  : [
            ("Unit-X1", 90.0), ("Unit-X2", 15.0), ("Unit-X3", 0.0),
            ("Unit-X4", 68.0), ("Unit-X5", 85.0), ("Unit-X6", 20.0),
            ("Unit-X7", 0.0), ("Unit-X8", 72.0)
        ],
    },
    {
        "name"   : "CASCADE FAILURE",
        "desc"   : "Resources cascading · imminent total collapse",
        "bias"   : +2.8, "volatility": 1.2,
        "theme_c": C_THEME_R,
        "units"  : [
            ("Core-1", 99.0), ("Core-2", 95.0), ("Core-3", 40.0),
            ("Core-4", 10.0), ("Core-5", 0.0)
        ],
    },
    {
        "name"       : "RESOURCE STORM",
        "desc"       : "Volatile state · all metrics unpredictable",
        "bias"       : 0.0, "volatility": 3.5,
        "theme_c"    : C_THEME_C,
        "random_init": True,
        "units"      : [
            ("Storm-A", 50.0), ("Storm-B", 50.0), ("Storm-C", 50.0),
            ("Storm-D", 50.0), ("Storm-E", 50.0), ("Storm-F", 50.0)
        ],
    },
    {
        "name"   : "RECOVERY PHASE",
        "desc"   : "Post-crisis redistribution in progress",
        "bias"   : -0.9, "volatility": 1.1,
        "theme_c": C_THEME_C,
        "units"  : [
            ("Recov-A", 82.0), ("Recov-B", 78.0), ("Recov-C", 35.0),
            ("Recov-D", 10.0), ("Recov-E", 65.0), ("Recov-F", 0.0)
        ],
    },
    {
        "name"   : "SURGE EVENT",
        "desc"   : "Sudden demand spike · rapid reallocation needed",
        "bias"   : +3.0, "volatility": 2.5,
        "theme_c": C_THEME_M,
        "units"  : [
            ("Hub-1", 30.0), ("Hub-2", 25.0), ("Hub-3", 15.0),
            ("Hub-4", 40.0), ("Hub-5", 20.0)
        ],
    },
]


# ── Data model ─────────────────────────────────────────────────────────────────

def _classify(r: float) -> str:
    if r <= 0:
        return "Unallocated"
    if r <= 40:
        return "Underutilized"
    if r <= 79:
        return "Optimal"
    return "Overutilized"


@dataclass
class Unit:
    name: str
    resource: float
    status: str = field(init=False, default="")
    _drift: float = field(init=False, default=0.0)
    _hist: deque = field(init=False, default_factory=lambda: deque(maxlen=14))

    def __post_init__(self):
        self.status = _classify(self.resource)

    def tick(self, bias: float, volatility: float) -> None:
        noise = random.gauss(bias, 4.5 * volatility)
        mean_pull = (50.0 - self.resource) * 0.008
        self._drift = self._drift * 0.82 + (noise + mean_pull) * 0.18
        self.resource = max(0.0, min(100.0, self.resource + self._drift * 0.55))
        self.status = _classify(self.resource)
        self._hist.append(self.resource)

    @property
    def trend(self) -> str:
        h = list(self._hist)
        if len(h) < 5:
            return " ~ "
        diff = sum(h[-3:]) / 3 - sum(h[:3]) / 3
        if diff > 5:
            return "+++"
        if diff > 1.5:
            return " + "
        if diff < -5:
            return "---"
        if diff < -1.5:
            return " - "
        return " = "

    @property
    def trend_c(self) -> int:
        h = list(self._hist)
        if len(h) < 5:
            return C_LABEL
        diff = sum(h[-3:]) / 3 - sum(h[:3]) / 3
        if diff > 1.5:
            return C_TREND_UP
        if diff < -1.5:
            return C_TREND_DN
        return C_TREND_FL

    @property
    def spark(self) -> str:
        """8-char sparkline from history."""
        h = list(self._hist)
        if not h:
            return "        "
        chars = [SPARK_CHARS[min(4, int(v / 20))] for v in h[-8:]]
        return "".join(chars).ljust(8)


def _make_units(template: dict) -> List[Unit]:
    units = []
    for name, val in template["units"]:
        if template.get("random_init"):
            val = random.uniform(0, 100)
        units.append(Unit(name=name, resource=float(val)))
    return units


# ── Metrics & planning ─────────────────────────────────────────────────────────

def compute_metrics(units: List[Unit]) -> dict:
    n = len(units)
    counts: Dict[str, int] = {s: 0 for s in STATUS_COLORS}
    for u in units:
        counts[u.status] += 1

    total = sum(u.resource for u in units)
    avg = total / n if n else 0.0

    opt_frac = counts["Optimal"] / n
    over_penalty = counts["Overutilized"] / n * 0.55
    under_penalty = counts["Underutilized"] / n * 0.25
    unal_penalty = counts["Unallocated"] / n * 0.35
    raw = opt_frac - over_penalty - under_penalty - unal_penalty
    score = max(0, min(100, int(raw * 100)))

    defuse_rate = avg
    non_opt = n - counts["Optimal"]
    eta = (non_opt * avg / (defuse_rate + 1e-6)) if defuse_rate > 0 else 999.9
    eta = min(eta, 999.9)

    risk = ("LOW" if score >= 70 else
            "MODERATE" if score >= 40 else
            "HIGH" if score >= 15 else
            "CRITICAL")

    return {
        "n"          : n, "total": total, "avg": avg,
        "counts"     : counts, "score": score,
        "defuse_rate": defuse_rate, "eta": eta, "risk": risk
    }


def build_defuse_plan(units: List[Unit]) -> List[Tuple[str, str]]:
    overloaded = sorted(
        [u for u in units if u.status == "Overutilized"],
        key=lambda u: u.resource, reverse=True,
    )
    unallocated = [u for u in units if u.status == "Unallocated"]
    underutilized = [u for u in units if u.status == "Underutilized"]

    plan: List[Tuple[str, str]] = []
    step = 1
    receivers = unallocated + underutilized

    for i, src in enumerate(overloaded):
        excess = src.resource - 70.0
        if i < len(receivers):
            dst = receivers[i]
            plan.append(("SHED", f" {step}. SHED {excess:5.1f}u  {src.name:<10} -> {dst.name}"))
        else:
            plan.append(("THROTTLE", f" {step}. THROTTLE    {src.name:<10} ({src.resource:.1f}% -> 70%)"))
        step += 1

    for u in unallocated:
        plan.append(("ACTIVATE", f" {step}. ACTIVATE    {u.name:<10} allocate ~50u baseline"))
        step += 1

    for u in underutilized:
        boost = 50.0 - u.resource
        plan.append(("BOOST", f" {step}. BOOST +{boost:4.1f}u  {u.name:<10} ({u.resource:.1f}% -> 50%)"))
        step += 1

    if not plan:
        plan.append(("OK", " [OK] SYSTEM OPTIMAL — no redistribution needed"))

    return plan[:9]


# ── ncurses helpers ────────────────────────────────────────────────────────────

def init_colors() -> None:
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(C_NORMAL, curses.COLOR_WHITE, -1)
    curses.init_pair(C_HEADER, curses.COLOR_CYAN, -1)
    curses.init_pair(C_UNAL, curses.COLOR_WHITE, -1)
    curses.init_pair(C_UNDER, curses.COLOR_YELLOW, -1)
    curses.init_pair(C_OPTIMAL, curses.COLOR_GREEN, -1)
    curses.init_pair(C_OVER, curses.COLOR_RED, -1)
    curses.init_pair(C_BORDER, curses.COLOR_BLUE, -1)
    curses.init_pair(C_LABEL, curses.COLOR_WHITE, -1)
    curses.init_pair(C_TIMER, curses.COLOR_MAGENTA, -1)
    curses.init_pair(C_DEFUSE, curses.COLOR_CYAN, -1)
    curses.init_pair(C_WARN, curses.COLOR_RED, -1)
    curses.init_pair(C_TREND_UP, curses.COLOR_RED, -1)
    curses.init_pair(C_TREND_DN, curses.COLOR_GREEN, -1)
    curses.init_pair(C_TREND_FL, curses.COLOR_WHITE, -1)
    curses.init_pair(C_SHED, curses.COLOR_RED, -1)
    curses.init_pair(C_ACTIVATE, curses.COLOR_GREEN, -1)
    curses.init_pair(C_BOOST, curses.COLOR_YELLOW, -1)
    curses.init_pair(C_THROTTLE, curses.COLOR_MAGENTA, -1)
    curses.init_pair(C_SPARK, curses.COLOR_BLUE, -1)
    curses.init_pair(C_THEME_R, curses.COLOR_RED, -1)
    curses.init_pair(C_THEME_Y, curses.COLOR_YELLOW, -1)
    curses.init_pair(C_THEME_G, curses.COLOR_GREEN, -1)
    curses.init_pair(C_THEME_M, curses.COLOR_MAGENTA, -1)
    curses.init_pair(C_THEME_C, curses.COLOR_CYAN, -1)
    curses.init_pair(C_SCORE_LO, curses.COLOR_RED, -1)
    curses.init_pair(C_SCORE_MID, curses.COLOR_YELLOW, -1)
    curses.init_pair(C_SCORE_HI, curses.COLOR_GREEN, -1)


def _put(win, y: int, x: int, text: str, attr: int = 0) -> None:
    try:
        h, w = win.getmaxyx()
        if y < 0 or y >= h or x < 0 or x >= w:
            return
        avail = w - x - 1
        if avail <= 0:
            return
        win.addstr(y, x, text[:avail], attr)
    except curses.error:
        pass


def _hline(win, y: int, ch: str = "-", color: int = C_BORDER) -> None:
    try:
        h, w = win.getmaxyx()
        if 0 <= y < h:
            win.addstr(y, 0, ch * (w - 1), curses.color_pair(color))
    except curses.error:
        pass


def _gradient_bar(win, y: int, x: int, value: float, width: int) -> None:
    """3-zone gradient bar: yellow(0-40%) | green(40-79%) | red(80-100%)."""
    filled = max(0, min(width, int(value / 100.0 * width)))
    low_end = int(0.40 * width)
    mid_end = int(0.79 * width)

    zones = [
        (0, low_end, C_UNDER, "="),
        (low_end, mid_end, C_OPTIMAL, "#"),
        (mid_end, width, C_OVER, "!"),
    ]
    col = x
    for zs, ze, cid, ch in zones:
        zw = ze - zs
        if zw <= 0:
            continue
        zf = max(0, min(zw, filled - zs))
        if zf > 0:
            _put(
                win, y, col, ch * zf,
                             curses.color_pair(cid) | curses.A_BOLD,
            )
        rem = zw - zf
        if rem > 0:
            _put(
                win, y, col + zf, "." * rem,
                curses.color_pair(C_BORDER),
            )
        col += zw


def _score_bar(win, y: int, x: int, score: int, width: int) -> None:
    """Gradient score bar: red zone | yellow zone | green zone."""
    filled = int(score / 100 * width)
    low_end = int(0.30 * width)
    mid_end = int(0.60 * width)

    zones = [
        (0, low_end, C_SCORE_LO, "#"),
        (low_end, mid_end, C_SCORE_MID, "#"),
        (mid_end, width, C_SCORE_HI, "#"),
    ]
    col = x
    for zs, ze, cid, ch in zones:
        zw = ze - zs
        if zw <= 0:
            continue
        zf = max(0, min(zw, filled - zs))
        if zf > 0:
            _put(
                win, y, col, ch * zf,
                             curses.color_pair(cid) | curses.A_BOLD,
            )
        rem = zw - zf
        if rem > 0:
            _put(
                win, y, col + zf, "-" * rem,
                curses.color_pair(C_BORDER),
            )
        col += zw


PLAN_COLORS: Dict[str, int] = {
    "SHED"    : C_SHED,
    "THROTTLE": C_THROTTLE,
    "ACTIVATE": C_ACTIVATE,
    "BOOST"   : C_BOOST,
    "OK"      : C_OPTIMAL,
}

# ── Genetic Algorithm ──────────────────────────────────────────────────────────
GA_POP_SIZE = 40
GA_MUTATION_RATE = 0.25
GA_MUTATION_STD = 10.0
GA_ELITE_K = 5
GA_TOURNAMENT_K = 4
GA_CONVERGENCE = 88.0  # score % to mark as converged


@dataclass
class GAState:
    generation: int = 0
    best_score: float = 0.0
    best_genes: list = field(default_factory=list)
    avg_score: float = 0.0
    diversity: float = 100.0
    history: deque = field(default_factory=lambda: deque(maxlen=32))
    converged: bool = False
    lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False, compare=False,
    )


def _ga_fitness(genes: list) -> float:
    """Fitness: fraction optimal − weighted penalties + center-bonus."""
    n = len(genes)
    if n == 0:
        return 0.0
    counts = {s: 0 for s in STATUS_COLORS}
    for v in genes:
        counts[_classify(v)] += 1
    opt_frac = counts["Optimal"] / n
    over_penalty = counts["Overutilized"] / n * 0.55
    under_penalty = counts["Underutilized"] / n * 0.25
    unal_penalty = counts["Unallocated"] / n * 0.35
    raw = opt_frac - over_penalty - under_penalty - unal_penalty
    # Bonus: distance from center of optimal band (60 %)
    center = sum(
        (1.0 - abs(v - 60.0) / 60.0
         for v in genes if 41 <= v <= 79),
    )
    return max(0.0, min(1.0, raw + (center / n) * 0.15))


class GeneticAlgorithm:
    def __init__(self, n_units: int, state: GAState):
        self.n = n_units
        self.state = state
        # Seed: 1/3 fully random, rest biased toward optimal band
        self.pop: List[list] = []
        for i in range(GA_POP_SIZE):
            if i < GA_POP_SIZE // 3:
                genes = [random.uniform(0, 100) for _ in range(n_units)]
            else:
                genes = [random.uniform(41, 79) for _ in range(n_units)]
            self.pop.append(genes)

    def _tournament(self, fits: list) -> int:
        k = min(GA_TOURNAMENT_K, len(fits))
        cands = random.sample(range(len(fits)), k)
        return max(cands, key=lambda i: fits[i])

    def _crossover(self, p1: list, p2: list) -> list:
        """BLX-α blend crossover."""
        alpha = 0.3
        child = []
        for x, y in zip(p1, p2):
            lo = min(x, y) - alpha * abs(x - y)
            hi = max(x, y) + alpha * abs(x - y)
            child.append(max(0.0, min(100.0, random.uniform(lo, hi))))
        return child

    def _mutate(self, genes: list) -> list:
        result = genes[:]
        for i in range(len(result)):
            if random.random() < GA_MUTATION_RATE:
                result[i] = max(
                    0.0, min(
                        100.0,
                        result[i] + random.gauss(0, GA_MUTATION_STD),
                    ),
                )
        return result

    def step(self) -> None:
        fits = [_ga_fitness(ind) for ind in self.pop]
        order = sorted(range(len(fits)), key=lambda i: fits[i], reverse=True)
        s_pop = [self.pop[i] for i in order]
        s_fit = [fits[i] for i in order]

        best_f = s_fit[0]
        avg_f = sum(s_fit) / len(s_fit)
        var = sum((f - avg_f) ** 2 for f in s_fit) / len(s_fit)

        with self.state.lock:
            self.state.generation += 1
            if best_f * 100.0 > self.state.best_score:
                self.state.best_score = best_f * 100.0
                self.state.best_genes = s_pop[0][:]
            self.state.avg_score = avg_f * 100.0
            self.state.diversity = (var ** 0.5) * 100.0
            self.state.history.append(self.state.best_score)
            self.state.converged = (self.state.best_score >= GA_CONVERGENCE)

        # Next generation with elitism
        new_pop = s_pop[:GA_ELITE_K]
        while len(new_pop) < GA_POP_SIZE:
            p1 = s_pop[self._tournament(s_fit)]
            p2 = s_pop[self._tournament(s_fit)]
            child = self._mutate(self._crossover(p1, p2))
            new_pop.append(child)
        self.pop = new_pop


def _ga_worker(ga: GeneticAlgorithm, stop: threading.Event) -> None:
    while not stop.is_set():
        ga.step()
        time.sleep(0.005 if not ga.state.converged else 0.04)


# ── Catastrophic Events ────────────────────────────────────────────────────────

@dataclass
class EventState:
    name: str = ""
    desc: str = ""
    color: int = C_OVER
    end_time: float = 0.0

    @property
    def active(self) -> bool:
        return bool(self.name) and time.time() < self.end_time

    @property
    def remaining(self) -> float:
        return max(0.0, self.end_time - time.time())


# ── Event effect functions ─────────────────────────────────────────────────────

def _evt_blackout(units: List["Unit"]) -> None:
    """Half the units suddenly lose all power."""
    victims = random.sample(units, max(1, len(units) // 2))
    for u in victims:
        u.resource = random.uniform(0.0, 6.0)
        u._drift = -12.0


def _evt_meltdown(units: List["Unit"]) -> None:
    """Most-loaded unit goes critical, radiating heat to neighbours."""
    if not units:
        return
    victim = max(units, key=lambda u: u.resource)
    victim.resource = 100.0
    victim._drift = +14.0
    for u in units:
        if u is not victim:
            u.resource = min(100.0, u.resource + random.uniform(8.0, 24.0))
            u._drift = max(u._drift, +5.0)


def _evt_emp_pulse(units: List["Unit"]) -> None:
    """Electromagnetic surge — every unit resets to chaos."""
    for u in units:
        u.resource = random.uniform(0.0, 100.0)
        u._drift = random.gauss(0.0, 10.0)


def _evt_cascade(units: List["Unit"]) -> None:
    """Sequential collapse — units drain in order from highest to lowest."""
    for i, u in enumerate(sorted(units, key=lambda u: u.resource, reverse=True)):
        u.resource = max(0.0, u.resource - (i + 1) * 18.0)
        u._drift = -10.0


def _evt_demand_surge(units: List["Unit"]) -> None:
    """Sudden demand spike — all units pushed toward overload."""
    for u in units:
        u.resource = min(100.0, u.resource + random.uniform(28.0, 48.0))
        u._drift = +8.0


def _evt_failover(units: List["Unit"]) -> None:
    """Emergency failover — highest unit dumps load to the coldest standby."""
    if len(units) < 2:
        return
    src = max(units, key=lambda u: u.resource)
    dst = min(units, key=lambda u: u.resource)
    combined = src.resource + dst.resource
    src.resource = combined * 0.08
    dst.resource = min(100.0, combined * 0.92)
    src._drift = -14.0
    dst._drift = +14.0


def _evt_fragmentation(units: List["Unit"]) -> None:
    """Split failure — alternating units go critical / dead."""
    for i, u in enumerate(units):
        u.resource = 97.0 if i % 2 == 0 else 1.0
        u._drift = +8.0 if i % 2 == 0 else -8.0


def _evt_total_collapse(units: List["Unit"]) -> None:
    """Everything fails simultaneously."""
    for u in units:
        u.resource = random.uniform(0.0, 4.0)
        u._drift = -15.0


CATASTROPHIC_EVENTS = [
    {
        "name" : "!! BLACKOUT !!", "desc": "Power failure — units collapsing to zero",
        "color": C_OVER, "duration": (3.0, 4.5), "effect": _evt_blackout
    },
    {
        "name" : "!! MELTDOWN !!", "desc": "Critical overload radiating to adjacent units",
        "color": C_WARN, "duration": (2.5, 4.0), "effect": _evt_meltdown
    },
    {
        "name" : "!! EMP PULSE !!", "desc": "Electromagnetic surge — all values randomized",
        "color": C_THEME_M, "duration": (2.0, 3.0), "effect": _evt_emp_pulse
    },
    {
        "name" : "!! CASCADE FAIL !!", "desc": "Sequential failure — drain spreading unit by unit",
        "color": C_OVER, "duration": (3.5, 5.0), "effect": _evt_cascade
    },
    {
        "name" : "!! DEMAND SURGE !!", "desc": "Sudden spike — all units pushed toward overload",
        "color": C_THEME_R, "duration": (2.0, 3.5), "effect": _evt_demand_surge
    },
    {
        "name" : "!! FAILOVER !!", "desc": "Emergency dump — load transferred to cold standby",
        "color": C_THEME_Y, "duration": (2.0, 3.0), "effect": _evt_failover
    },
    {
        "name" : "!! FRAGMENTATION !!", "desc": "Split failure — alternating overload and void",
        "color": C_THEME_M, "duration": (2.5, 4.0), "effect": _evt_fragmentation
    },
    {
        "name" : "!! TOTAL COLLAPSE !!", "desc": "Everything fails simultaneously — restart imminent",
        "color": C_WARN, "duration": (4.0, 6.0), "effect": _evt_total_collapse
    },
]


def _flash_event_alert(win, event: dict) -> None:
    """Full-screen 4-flash dramatic alert when a catastrophic event fires."""
    h, w = win.getmaxyx()
    color = event.get("color", C_OVER)
    name = event["name"]
    desc = event["desc"]
    fill = curses.color_pair(color) | curses.A_REVERSE
    title_attr = curses.color_pair(color) | curses.A_BOLD | curses.A_REVERSE
    desc_attr = curses.color_pair(color) | curses.A_BOLD

    for _ in range(4):
        win.erase()
        for y in range(h):
            try:
                win.addstr(y, 0, " " * (w - 1), fill)
            except curses.error:
                pass
        _put(win, h // 2 - 1, max(0, (w - len(name)) // 2), name, title_attr)
        _put(win, h // 2, max(0, (w - len(desc)) // 2), desc, desc_attr)
        _put(win, h // 2 + 1, max(0, (w - 18) // 2), ">>> ALERT ACTIVE <<<", title_attr)
        win.refresh()
        time.sleep(0.11)
        win.erase()
        win.refresh()
        time.sleep(0.07)


# ── GA result flash ───────────────────────────────────────────────────────────

def _flash_ga_result(win, state: GAState, template: dict, theme_c: int) -> None:
    """Show GA best allocation for ~2 s between scenarios."""
    h, w = win.getmaxyx()
    genes = list(state.best_genes)
    units_def = template["units"]
    end_time = time.time() + 2.2
    f = 0
    while time.time() < end_time:
        win.erase()
        _hline(win, 0, "=", theme_c)
        title = (f" GA OPTIMAL SOLUTION  "
                 f"[Score: {state.best_score:.1f}/100  "
                 f"Gen: {state.generation}  "
                 f"{'CONVERGED' if state.converged else 'BEST SO FAR'}] ")
        _put(
            win, 0, max(0, (w - len(title)) // 2),
            title, curses.color_pair(theme_c) | curses.A_BOLD | curses.A_REVERSE,
        )

        row = 2
        _put(
            win, row, 2,
            "RECOMMENDED ALLOCATION  (Genetic Algorithm Best Fit)",
            curses.color_pair(C_DEFUSE) | curses.A_BOLD,
        )
        row += 1
        _put(
            win, row, 2,
            f"  {'UNIT':<12}  {'GA TARGET':>9}   {'DELTA':>8}   {'STATUS':<14}  BAR",
            curses.color_pair(C_LABEL) | curses.A_BOLD,
        )
        row += 1

        for i, (name, orig) in enumerate(units_def):
            if row >= h - 3 or i >= len(genes):
                break
            v = genes[i]
            status = _classify(v)
            icon = STATUS_ICONS.get(status, "[ ]")
            color = curses.color_pair(STATUS_COLORS.get(status, C_NORMAL))
            delta = v - orig
            ds = f"+{delta:5.1f}" if delta >= 0 else f"{delta:6.1f}"
            prefix = f"  {name:<12}  {v:7.1f}%   {ds}%   {icon} {status:<12} "
            _put(win, row, 0, prefix, color)
            _gradient_bar(win, row, len(prefix), v, BAR_WIDTH)
            row += 1

        pulse = "(*)" if (f // 4) % 2 == 0 else "( )"
        _put(
            win, h - 1, 2,
            f" {pulse}  NEXT SCENARIO LOADING...  "
            f"  Diversity: {state.diversity:.1f}  Avg: {state.avg_score:.1f}/100 ",
                 curses.color_pair(C_TIMER) | curses.A_BOLD,
        )
        win.refresh()
        f += 1
        time.sleep(0.05)
        key = win.getch()
        if key != -1:
            break


# ── Transition flash ───────────────────────────────────────────────────────────

def _flash_transition(win, next_name: str, theme_c: int) -> None:
    h, w = win.getmaxyx()
    msg = f"  >> NEXT: {next_name} <<  "
    half = len(msg) // 2
    for i in range(6):
        win.erase()
        attr = curses.color_pair(theme_c) | curses.A_BOLD | curses.A_REVERSE
        _put(win, h // 2 - 1, max(0, w // 2 - half - 2), " " * (len(msg) + 4), attr)
        _put(win, h // 2, max(0, w // 2 - half), msg, attr)
        _put(win, h // 2 + 1, max(0, w // 2 - half - 2), " " * (len(msg) + 4), attr)
        win.refresh()
        time.sleep(0.09)
        win.erase()
        win.refresh()
        time.sleep(0.07)


# ── Main draw ──────────────────────────────────────────────────────────────────

def draw_frame(
    win,
    scen_idx: int,
    total_scen: int,
    template: dict,
    units: List[Unit],
    metrics: dict,
    plan: list,
    elapsed: float,
    frame: int,
    ga_state: Optional[GAState] = None,
    event_state: Optional[EventState] = None,
) -> None:
    win.erase()
    h, w = win.getmaxyx()
    remaining = max(0.0, SCENARIO_DURATION - elapsed)
    score = metrics["score"]
    counts = metrics["counts"]
    risk = metrics["risk"]
    theme_c = template.get("theme_c", C_HEADER)
    # Flash theme color to event color while catastrophe active
    eff_theme = (event_state.color
                 if event_state is not None and event_state.active
                 else theme_c)
    row = 0

    # ── Themed title bar ──────────────────────────────────────────────────────
    title = f" RESOURCE ALLOCATION MONITOR  [ {scen_idx + 1}/{total_scen} ] "
    _hline(win, row, "=", eff_theme)
    _put(
        win, row, max(0, (w - len(title)) // 2),
        title, curses.color_pair(eff_theme) | curses.A_BOLD | curses.A_REVERSE,
    )
    row += 1

    _put(
        win, row, 2,
        f"{template['name']}  |  {template['desc']}",
        curses.color_pair(eff_theme) | curses.A_BOLD,
    )

    # System overview: one colored char per unit on same row, right-aligned
    overview_x = w - len(units) - 4
    _put(win, row, overview_x, "[", curses.color_pair(C_LABEL))
    for i, u in enumerate(units):
        ch = OVERVIEW_CHARS.get(u.status, "?")
        color = curses.color_pair(STATUS_COLORS.get(u.status, C_LABEL)) | curses.A_BOLD
        _put(win, row, overview_x + 1 + i, ch, color)
    _put(win, row, overview_x + 1 + len(units), "]", curses.color_pair(C_LABEL))
    row += 1

    _hline(win, row, "-", eff_theme)
    row += 1

    # ── Catastrophic Event banner ─────────────────────────────────────────────
    if event_state is not None and event_state.active:
        rem_e = event_state.remaining
        blink = (frame // 3) % 2 == 0
        ev_fill = curses.color_pair(event_state.color) | curses.A_REVERSE
        ev_attr = curses.color_pair(event_state.color) | curses.A_BOLD
        if blink:
            ev_attr |= curses.A_REVERSE
        banner = (f"  {event_state.name}  —  {event_state.desc}"
                  f"  [{rem_e:.1f}s]  ")
        try:
            win.addstr(row, 0, " " * (w - 1), ev_fill)
        except curses.error:
            pass
        _put(win, row, max(0, (w - len(banner)) // 2), banner, ev_attr)
        row += 1
        _hline(win, row, "-", event_state.color)
        row += 1

    # ── Unit table ────────────────────────────────────────────────────────────
    _put(
        win, row, 0,
        f"  {'UNIT':<12}  {'RES':>6}   {'STATUS':<15}  {'ALLOCATION BAR':<{BAR_WIDTH}}  {'TRD'}  {'HISTORY'}",
        curses.color_pair(C_LABEL) | curses.A_BOLD,
    )
    row += 1

    for u in units:
        if row >= h - 18:
            break
        icon = STATUS_ICONS.get(u.status, "[ ]")
        ucolor = curses.color_pair(STATUS_COLORS.get(u.status, C_NORMAL))
        prefix = f"  {u.name:<12}  {u.resource:5.1f}%   {icon} {u.status:<12} "
        _put(win, row, 0, prefix, ucolor)
        col = len(prefix)

        _gradient_bar(win, row, col, u.resource, BAR_WIDTH)
        col += BAR_WIDTH + 1

        # Trend
        _put(
            win, row, col + 1, u.trend,
                      curses.color_pair(u.trend_c) | curses.A_BOLD,
        )
        col += 6

        # Sparkline
        _put(
            win, row, col, u.spark,
            curses.color_pair(C_SPARK) | curses.A_BOLD,
        )
        row += 1

    row += 1
    _hline(win, row, "-", eff_theme)
    row += 1

    # ── Metrics row ───────────────────────────────────────────────────────────
    risk_color = {
        "LOW" : C_OPTIMAL, "MODERATE": C_UNDER,
        "HIGH": C_OVER, "CRITICAL": C_WARN,
    }.get(risk, C_LABEL)
    risk_attr = curses.color_pair(risk_color) | curses.A_BOLD
    if risk == "CRITICAL" and (frame // 5) % 2 == 0:
        risk_attr |= curses.A_REVERSE

    # color each metric separately
    _put(win, row, 2, "Total:", curses.color_pair(C_LABEL))
    _put(win, row, 9, f"{metrics['total']:7.1f}u", curses.color_pair(C_UNDER) | curses.A_BOLD)
    _put(win, row, 18, "Avg:", curses.color_pair(C_LABEL))
    _put(win, row, 23, f"{metrics['avg']:5.1f}u", curses.color_pair(C_OPTIMAL) | curses.A_BOLD)
    _put(win, row, 31, "Rate:", curses.color_pair(C_LABEL))
    _put(win, row, 37, f"{metrics['defuse_rate']:5.1f}", curses.color_pair(C_HEADER) | curses.A_BOLD)
    _put(win, row, 43, "ETA:", curses.color_pair(C_LABEL))
    _put(win, row, 48, f"{metrics['eta']:6.1f}t", curses.color_pair(C_TIMER) | curses.A_BOLD)
    _put(win, row, w - 18, f"RISK: {risk:<8}", risk_attr)
    row += 1

    # ── Gradient score bar ────────────────────────────────────────────────────
    sbw = 36
    _put(win, row, 2, "Score [", curses.color_pair(C_LABEL) | curses.A_BOLD)
    _score_bar(win, row, 9, score, sbw)
    _put(
        win, row, 9 + sbw, f"] {score:3d}/100",
                  curses.color_pair(C_LABEL) | curses.A_BOLD,
    )
    row += 1

    # ── Status distribution colored mini-bar ──────────────────────────────────
    total_u = len(units)
    _put(win, row, 2, "Dist  ", curses.color_pair(C_LABEL))
    dx = 8
    for status, cid, ch in [
        ("Unallocated", C_UNAL, "U"),
        ("Underutilized", C_UNDER, "v"),
        ("Optimal", C_OPTIMAL, "*"),
        ("Overutilized", C_OVER, "^"),
    ]:
        cnt = counts[status]
        label = f"{ch}x{cnt}"
        _put(win, row, dx, label, curses.color_pair(cid) | curses.A_BOLD)
        dx += len(label) + 1
        # mini block bar
        blocks = "#" * cnt
        _put(
            win, row, dx, f"[{blocks:<{total_u}}]",
            curses.color_pair(cid),
        )
        dx += total_u + 4

    row += 1
    _hline(win, row, "-", eff_theme)
    row += 1

    # ── Defuse plan (colored by action type) ──────────────────────────────────
    _put(
        win, row, 2, "DEFUSE PLAN",
        curses.color_pair(C_DEFUSE) | curses.A_BOLD,
    )
    row += 1

    for action, line in plan:
        if row >= h - 4:
            break
        plan_color = PLAN_COLORS.get(action, C_DEFUSE)
        _put(
            win, row, 2, line,
            curses.color_pair(plan_color) | curses.A_BOLD,
        )
        row += 1

    # ── GA Optimizer panel ────────────────────────────────────────────────────
    if ga_state is not None and row < h - 6:
        with ga_state.lock:
            gen = ga_state.generation
            best_s = ga_state.best_score
            avg_s = ga_state.avg_score
            div = ga_state.diversity
            conv = ga_state.converged
            best_g = list(ga_state.best_genes)
            hist = list(ga_state.history)

        _hline(win, row, "-", C_DEFUSE)
        row += 1

        conv_str = "CONVERGED!" if conv else "evolving.."
        conv_color = C_OPTIMAL if conv else C_UNDER
        conv_attr = curses.color_pair(conv_color) | curses.A_BOLD
        if conv and (frame // 5) % 2 == 0:
            conv_attr |= curses.A_REVERSE

        _put(win, row, 2, "GA", curses.color_pair(C_DEFUSE) | curses.A_BOLD)
        _put(win, row, 5, f"Gen:{gen:5d}", curses.color_pair(C_HEADER) | curses.A_BOLD)
        _put(win, row, 16, f"Best:{best_s:5.1f}", curses.color_pair(C_OPTIMAL) | curses.A_BOLD)
        _put(win, row, 27, f"Avg:{avg_s:5.1f}", curses.color_pair(C_UNDER) | curses.A_BOLD)
        _put(win, row, 37, f"Div:{div:4.1f}", curses.color_pair(C_SPARK) | curses.A_BOLD)
        _put(win, row, 47, f"[{conv_str}]", conv_attr)
        row += 1

        if row < h - 4:
            spark_hist = "".join(
                (SPARK_CHARS[min(4, int(v / 20))] for v in hist[-22:]),
            ) if hist else ""
            _put(win, row, 2, "Fit  [", curses.color_pair(C_LABEL))
            _score_bar(win, row, 8, int(best_s), 24)
            _put(win, row, 33, f"] {best_s:5.1f}%  Evol:", curses.color_pair(C_LABEL))
            _put(
                win, row, 50, spark_hist.ljust(22),
                curses.color_pair(C_SPARK) | curses.A_BOLD,
            )
            row += 1

        if row < h - 4 and best_g:
            parts = [f"{name}:{best_g[i]:4.1f}"
                     for i, (name, _) in enumerate(template["units"])
                     if i < len(best_g)]
            _put(
                win, row, 2, "Best: " + "  ".join(parts),
                             curses.color_pair(C_OPTIMAL) | curses.A_BOLD,
            )
            row += 1

    # ── Themed footer ─────────────────────────────────────────────────────────
    _hline(win, h - 2, "=", eff_theme)
    _put(
        win, h - 2, w - 30,
        "  Q/ESC=quit  SPACE=skip  ",
        curses.color_pair(C_LABEL),
    )

    prog_w = max(4, w - 34)
    prog_fill = int((elapsed / SCENARIO_DURATION) * prog_w)
    # gradient progress bar: theme color for filled portion
    pulse = "(*)" if (frame // 6) % 2 == 0 else "( )"
    timer_str = f" {pulse} {remaining:4.1f}s left "
    _put(
        win, h - 1, 0, timer_str,
             curses.color_pair(C_TIMER) | curses.A_BOLD,
    )
    tx = len(timer_str)
    _put(
        win, h - 1, tx, "[",
        curses.color_pair(C_BORDER),
    )
    _put(
        win, h - 1, tx + 1, "|" * prog_fill,
         curses.color_pair(eff_theme) | curses.A_BOLD)
    _put(
        win, h - 1, tx + 1 + prog_fill, "." * (prog_w - prog_fill),
        curses.color_pair(C_BORDER),
    )
    _put(
        win, h - 1, tx + 1 + prog_w, "]",
        curses.color_pair(C_BORDER),
    )

    next_name = SCENARIO_TEMPLATES[(scen_idx + 1) % total_scen]["name"]
    next_str = f" Next: {next_name} "
    _put(
        win, h - 1, w - len(next_str) - 1,
        next_str, curses.color_pair(C_LABEL),
    )

    win.refresh()


# ── Main loop ──────────────────────────────────────────────────────────────────

def main(stdscr) -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(TICK_MS)
    init_colors()

    total = len(SCENARIO_TEMPLATES)
    scen_idx = 0
    frame = 0

    while True:
        template = SCENARIO_TEMPLATES[scen_idx]
        units = _make_units(template)
        bias = template.get("bias", 0.0)
        vol = template.get("volatility", 1.0)
        theme_c = template.get("theme_c", C_HEADER)
        scen_start = time.time()

        # ── Start GA thread for this scenario ──────────────────────────────
        ga_state = GAState()
        ga = GeneticAlgorithm(len(units), ga_state)
        ga_stop = threading.Event()
        ga_thread = threading.Thread(
            target=_ga_worker, args=(ga, ga_stop), daemon=True, name="ga-worker",
        )
        ga_thread.start()

        # ── Catastrophic event scheduling ───────────────────────────────────
        event_state      = EventState()
        next_event_time  = time.time() + random.uniform(2.5, 5.0)

        while True:
            elapsed = time.time() - scen_start
            if elapsed >= SCENARIO_DURATION:
                break

            # Trigger a new catastrophic event?
            now = time.time()
            if now >= next_event_time and not event_state.active:
                if random.random() < 0.70:   # 70 % chance to fire
                    evt = random.choice(CATASTROPHIC_EVENTS)
                    evt["effect"](units)
                    dur = random.uniform(*evt["duration"])
                    event_state = EventState(
                        name=evt["name"],
                        desc=evt["desc"],
                        color=evt["color"],
                        end_time=now + dur,
                    )
                    _flash_event_alert(stdscr, evt)
                next_event_time = now + random.uniform(3.0, 6.5)

            for u in units:
                u.tick(bias, vol)

            metrics = compute_metrics(units)
            plan    = build_defuse_plan(units)

            draw_frame(
                stdscr, scen_idx, total,
                template, units, metrics, plan,
                elapsed, frame, ga_state, event_state,
            )
            frame += 1

            key = stdscr.getch()
            if key in (ord("q"), ord("Q"), 27):
                ga_stop.set()
                ga_thread.join(timeout=0.5)
                return
            if key in (ord(" "), curses.KEY_RIGHT):
                break

        # ── Stop GA, show its best solution ────────────────────────────────
        ga_stop.set()
        ga_thread.join(timeout=0.5)

        _flash_ga_result(stdscr, ga_state, template, theme_c)

        next_idx = (scen_idx + 1) % total
        next_tmpl = SCENARIO_TEMPLATES[next_idx]
        _flash_transition(
            stdscr, next_tmpl["name"],
            next_tmpl.get("theme_c", C_HEADER),
        )
        scen_idx = next_idx


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass
