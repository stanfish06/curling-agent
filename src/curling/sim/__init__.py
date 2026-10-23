import ctypes as C
import json
import subprocess
from pathlib import Path

import gymnasium as gym
import numpy as np

from curling.model.physics import (
    BACK_LINE_Y,
    HOG_LINE_Y,
    HOUSE_RADIUS,
    MAX_SPEED,
    NOISE_ANGLE,
    NOISE_SPEED,
    SHEET_WIDTH,
    STONE_RADIUS,
    TEE_LINE_Y,
)

ROOT = Path(__file__).parent / "DigitalCurling"
PRESET = "linux-x64"
LIB = ROOT / "build" / PRESET / "lib" / "libdigitalcurling_plugin_loader.so"
N_STONES = 16
STONE_FIELDS = 6
MODE_FULL, MODE_OUT_STONE, MODE_COLLISION = 1, 2, 4
OK = 0


class Uuid(C.Structure):
    _fields_ = [("bytes", C.c_uint8 * 16)]


class Vec2(C.Structure):
    _fields_ = [("x", C.c_float), ("y", C.c_float)]


class StoneState(C.Structure):
    _fields_ = [
        ("position", Vec2),
        ("angle", C.c_float),
        ("velocity", Vec2),
        ("angular_velocity", C.c_float),
    ]


class Stones(C.Structure):
    _fields_ = [("stones", StoneState * N_STONES)]


class Shot(C.Structure):
    _fields_ = [
        ("translational_velocity", C.c_float),
        ("angular_velocity", C.c_float),
        ("release_angle", C.c_float),
    ]


SIGNATURES = {
    "dc_loader_get_last_error_message": [C.c_char_p, C.c_size_t, C.POINTER(C.c_size_t)],
    "dc_loader_snapshot_get_data": [
        C.c_void_p,
        C.c_char_p,
        C.c_size_t,
        C.POINTER(C.c_size_t),
    ],
    "dc_loader_destroy_snapshot": [C.c_void_p],
    "dc_loader_create_player_factory": [C.c_char_p, C.c_char_p, C.POINTER(Uuid)],
    "dc_loader_create_player": [C.POINTER(Uuid), C.POINTER(Uuid)],
    "dc_loader_remove_player_instance": [C.POINTER(Uuid)],
    "dc_loader_player_play": [C.POINTER(Uuid), C.POINTER(Shot), C.POINTER(Shot)],
    "dc_loader_create_simulator_factory": [C.c_char_p, C.c_char_p, C.POINTER(Uuid)],
    "dc_loader_create_simulator": [C.POINTER(Uuid), C.POINTER(Uuid)],
    "dc_loader_remove_simulator_instance": [C.POINTER(Uuid)],
    "dc_loader_simulator_set_stones": [C.POINTER(Uuid), C.POINTER(Stones)],
    "dc_loader_simulator_step": [C.POINTER(Uuid), C.c_int, C.c_float],
    "dc_loader_simulator_simulate": [C.POINTER(Uuid), C.c_int, C.c_float],
    "dc_loader_simulator_get_stones": [C.POINTER(Uuid), C.POINTER(Stones)],
    "dc_loader_simulator_get_collisions_snapshot": [
        C.POINTER(Uuid),
        C.POINTER(C.c_void_p),
        C.POINTER(C.c_size_t),
    ],
    "dc_loader_simulator_are_all_stones_stopped": [
        C.POINTER(Uuid),
        C.POINTER(C.c_bool),
    ],
    "dc_loader_simulator_get_seconds_per_frame": [
        C.POINTER(Uuid),
        C.POINTER(C.c_float),
    ],
    "dc_loader_simulator_calculate_shot": [
        C.POINTER(Uuid),
        C.POINTER(Vec2),
        C.c_float,
        C.c_float,
        C.POINTER(Shot),
    ],
}

_lib = None


def build() -> Path:
    subprocess.run(
        ["cmake", "--preset", PRESET, "-DDIGITALCURLING_BUILD_TEST=OFF"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(ROOT / "build" / PRESET), "-j"], cwd=ROOT, check=True
    )
    return LIB


def lib() -> C.CDLL:
    global _lib
    if _lib is None:
        if not LIB.exists():
            build()
        _lib = C.CDLL(str(LIB))
        for name, argtypes in SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes, fn.restype = argtypes, C.c_int
    return _lib


def check(code: int):
    if code != OK:
        n = C.c_size_t()
        lib().dc_loader_get_last_error_message(None, 0, C.byref(n))
        buf = C.create_string_buffer(n.value)
        lib().dc_loader_get_last_error_message(buf, n.value, C.byref(n))
        raise RuntimeError(f"DigitalCurling error {code}: {buf.value.decode()}")


def snapshot(handle: C.c_void_p, size: int) -> str:
    buf = C.create_string_buffer(size + 1)
    n = C.c_size_t()
    check(lib().dc_loader_snapshot_get_data(handle, buf, size + 1, C.byref(n)))
    check(lib().dc_loader_destroy_snapshot(handle))
    return buf.value.decode()


def pack(pos, vel, spin, active) -> Stones:
    s = Stones()
    for i in np.flatnonzero(active):
        s.stones[i].position = Vec2(float(pos[i, 0]), float(pos[i, 1]))
        s.stones[i].velocity = Vec2(float(vel[i, 0]), float(vel[i, 1]))
        s.stones[i].angular_velocity = float(spin[i])
    return s


def unpack(s: Stones):
    a = np.ctypeslib.as_array(s.stones).view(np.float32).reshape(N_STONES, STONE_FIELDS)
    pos, vel, spin = a[:, 0:2].copy(), a[:, 3:5].copy(), a[:, 5].copy()
    active = np.abs(a).sum(1) > 0
    return pos.astype(float), vel.astype(float), spin.astype(float), active


class Simulator:
    def __init__(
        self, seconds_per_frame: float = 0.001, sheet_width: float = SHEET_WIDTH
    ):
        self.sheet_width = sheet_width
        self.factory, self.id = Uuid(), Uuid()
        config = json.dumps({"type": "fcv1", "seconds_per_frame": seconds_per_frame})
        check(
            lib().dc_loader_create_simulator_factory(
                b"fcv1", config.encode(), C.byref(self.factory)
            )
        )
        check(lib().dc_loader_create_simulator(C.byref(self.factory), C.byref(self.id)))

    def __del__(self):
        if _lib is not None and hasattr(self, "id"):
            _lib.dc_loader_remove_simulator_instance(C.byref(self.id))

    def set_stones(self, pos, vel=None, spin=None, active=None):
        pos = np.asarray(pos, float)
        vel = np.zeros_like(pos) if vel is None else np.asarray(vel, float)
        spin = np.zeros(N_STONES) if spin is None else np.asarray(spin, float)
        active = ~np.isnan(pos[:, 0]) if active is None else np.asarray(active, bool)
        check(
            lib().dc_loader_simulator_set_stones(
                C.byref(self.id), C.byref(pack(pos, vel, spin, active))
            )
        )

    def stones(self):
        s = Stones()
        check(lib().dc_loader_simulator_get_stones(C.byref(self.id), C.byref(s)))
        return unpack(s)

    def step(self, frames: int = 1):
        check(
            lib().dc_loader_simulator_step(C.byref(self.id), frames, self.sheet_width)
        )

    def simulate(self, mode: int = MODE_FULL):
        check(
            lib().dc_loader_simulator_simulate(C.byref(self.id), mode, self.sheet_width)
        )

    def stopped(self) -> bool:
        out = C.c_bool()
        check(
            lib().dc_loader_simulator_are_all_stones_stopped(
                C.byref(self.id), C.byref(out)
            )
        )
        return out.value

    def seconds_per_frame(self) -> float:
        out = C.c_float()
        check(
            lib().dc_loader_simulator_get_seconds_per_frame(
                C.byref(self.id), C.byref(out)
            )
        )
        return out.value

    def collisions(self) -> list:
        handle, size = C.c_void_p(), C.c_size_t()
        check(
            lib().dc_loader_simulator_get_collisions_snapshot(
                C.byref(self.id), C.byref(handle), C.byref(size)
            )
        )
        return json.loads(snapshot(handle, size.value))

    def calculate_shot(
        self, target, target_speed: float, spin: float
    ) -> tuple[float, float, float]:
        out = Shot()
        check(
            lib().dc_loader_simulator_calculate_shot(
                C.byref(self.id),
                C.byref(Vec2(float(target[0]), float(target[1]))),
                target_speed,
                spin,
                C.byref(out),
            )
        )
        return out.translational_velocity, out.angular_velocity, out.release_angle

    def throw(
        self,
        pos,
        active,
        index: int,
        speed: float,
        spin: float,
        angle: float,
        record_every: int = 0,
    ):
        pos, active = np.array(pos, float), np.array(active, bool)
        vel, spins = np.zeros((N_STONES, 2)), np.zeros(N_STONES)
        pos[index], active[index] = 0.0, True
        vel[index] = speed * np.cos(angle), speed * np.sin(angle)
        spins[index] = spin
        self.set_stones(pos, vel, spins, active)
        before, others = pos.copy(), active.copy()
        others[index] = False
        frames = []
        if record_every:
            while not self.stopped():
                self.step(record_every)
                frames.append(self.stones()[0])
        else:
            self.simulate()
        pos, vel, spins, active = self.stones()
        still = others & active
        hit = still.sum() < others.sum() or bool(
            np.any(np.abs(pos[still] - before[still]) > 1e-6)
        )
        y = pos[:, 1]
        active &= y < BACK_LINE_Y + STONE_RADIUS
        if not hit and y[index] <= HOG_LINE_Y + STONE_RADIUS:
            active[index] = False
        pos[~active] = np.nan
        return pos, active, (np.stack(frames) if frames else None)


class Player:
    def __init__(
        self,
        kind: str = "normal_dist",
        max_speed: float = MAX_SPEED,
        stddev_speed: float = NOISE_SPEED,
        stddev_angle: float = NOISE_ANGLE,
        seed: int | None = None,
    ):
        config = {"type": kind, "gender": "unknown"}
        if kind == "normal_dist":
            config |= {
                "max_speed": max_speed,
                "stddev_speed": stddev_speed,
                "stddev_angle": stddev_angle,
                "seed": seed,
            }
        self.factory, self.id = Uuid(), Uuid()
        check(
            lib().dc_loader_create_player_factory(
                kind.encode(), json.dumps(config).encode(), C.byref(self.factory)
            )
        )
        check(lib().dc_loader_create_player(C.byref(self.factory), C.byref(self.id)))

    def __del__(self):
        if _lib is not None and hasattr(self, "id"):
            _lib.dc_loader_remove_player_instance(C.byref(self.id))

    def play(
        self, speed: float, spin: float, angle: float
    ) -> tuple[float, float, float]:
        out = Shot()
        check(
            lib().dc_loader_player_play(
                C.byref(self.id), C.byref(Shot(speed, spin, angle)), C.byref(out)
            )
        )
        return out.translational_velocity, out.angular_velocity, out.release_angle


N_SHOTS = 16
STONES_PER_TEAM = 8
TEE = np.array([0.0, TEE_LINE_Y])
HANDLE = np.pi / 2
SPEED_RANGE = (2.15, 3.4)
ANGLE_RANGE = 0.04
TAKEOUT_SPEED = 1.5
GUARD_Y = (TEE_LINE_Y - 4.5, TEE_LINE_Y - 2.5)
ACTION_LOW = np.array([1.5, -HANDLE, np.pi / 2 - 0.1])
ACTION_HIGH = np.array([MAX_SPEED, HANDLE, np.pi / 2 + 0.1])
STONE_WIDTH = 2 * STONE_RADIUS
MOVED = 0.01
FREEZE_GAP = (0.05, 0.10)
FRONT_PAST_HOG = (1.5, 0.5)
WEIGHTS = (0.5, 0.3, 0.2)
FOUR_FOOT = 0.610
CORRIDOR = FOUR_FOOT
HAMMER = 1
EARLY_END = 3
LATE_END = 7
LEAD = 3
FGZ_SHOTS = 5
DOUBLE_SIDE, DOUBLE_LINE = 0.915, 1.83
OFFENCE = {"draw", "guard", "front", "freeze", "raise"}
CCA = "Curling Canada, Free Guard Zone Strategy guide"
USA = "USA Curling Level I manual p.31"
WCF = "World Curling Rules of Curling 2024"
KATOH = "Katoh et al., IPSJ Journal 57(11) 2016"
RULES = {
    "S1": ("shot style fits the offence/defence situation", CCA),
    "S7": ("guards when behind, keep it clean when ahead", USA),
    "A1": ("thrower counts shot stone", KATOH),
    "P1": ("hammer team keeps the centre open, guards to the side", f"{USA}; {CCA}"),
    "P2": ("non-hammer team plays the centre", f"{USA}; {CCA}"),
    "P6": ("non-hammer four-foot stone in front of the tee", KATOH),
    "P7": ("no own pair left open for a double", KATOH),
    "R1": ("defence without hammer: hit opponent stones in the house", CCA),
    "R6": ("hammer after shot 5: remove opponent centre guards", USA),
    "R7": ("defence: peel the lone guard covering an opponent house stone", CCA),
    "R5": ("big lead late: throw-through is acceptable", CCA),
    "FGZ": ("no opponent free-guard-zone stone removed before shot 6", f"{WCF} R6(b)"),
    "NoTick": ("no opponent centre-line guard moved off before shot 6", f"{WCF} R7"),
    "E4": ("final end first stone without hammer fits the score", CCA),
    "E1": ("end objective: hammer scores 2+ or blanks", f"{USA}; {CCA}"),
    "E1n": ("end objective: non-hammer steals or forces 1", f"{USA}; {CCA}"),
}


def team_of(shot: int) -> int:
    return shot % 2


def stone_of(shot: int) -> int:
    return team_of(shot) * STONES_PER_TEAM + shot // 2


def in_house(pos: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pos - TEE, axis=-1) <= HOUSE_RADIUS + STONE_RADIUS


def end_score(pos: np.ndarray, active: np.ndarray) -> int:
    d = np.linalg.norm(np.nan_to_num(pos) - TEE, axis=-1)
    d[~active | ~in_house(np.nan_to_num(pos))] = np.inf
    d0, d1 = d[:STONES_PER_TEAM], d[STONES_PER_TEAM:]
    if d0.min() < d1.min():
        return int((d0 < d1.min()).sum())
    if d1.min() < d0.min():
        return -int((d1 < d0.min()).sum())
    return 0


def dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=-1)


def shot_type(
    before_pos: np.ndarray,
    before_active: np.ndarray,
    pos: np.ndarray,
    active: np.ndarray,
    shot: int,
) -> tuple[str, np.ndarray]:
    s = stone_of(shot)
    none = np.array([], int)
    if shot == N_SHOTS - 1:
        return "last", none
    others = before_active.copy()
    others[s] = False
    removed = others & ~active
    moved = others & active & (np.nan_to_num(dist(pos, before_pos)) > MOVED)
    opp = np.arange(N_STONES) // STONES_PER_TEAM != team_of(shot)
    if (
        not (removed & opp).any()
        and not moved.any()
        and (active[s] or not removed.any())
    ):
        if not active[s]:
            return "through", none
        if in_house(pos[s]):
            idx = np.flatnonzero(others & active)
            gap = dist(pos[idx], pos[s]) - STONE_WIDTH
            if idx.size and gap.min() <= FREEZE_GAP[1]:
                return "freeze", idx[[gap.argmin()]]
            return "draw", none
        guarded = (
            others
            & active
            & in_house(np.nan_to_num(before_pos))
            & (np.abs(before_pos[:, 0] - pos[s, 0]) <= STONE_WIDTH)
            & (before_pos[:, 1] > pos[s, 1])
        )
        return ("guard", np.flatnonzero(guarded)) if guarded.any() else ("front", none)
    if (removed & opp).sum() >= 2:
        return "double", np.flatnonzero(removed & opp)
    if removed.any() and not active[s]:
        return "clearing", np.flatnonzero(removed)
    if (removed & opp).any():
        t = np.flatnonzero(removed & opp)
        stick = dist(pos[s], before_pos[t[0]]) <= STONE_WIDTH
        return ("takeout" if stick else "hit_roll"), t
    return "raise", np.flatnonzero(moved)


def execution_score(
    task: str,
    targets: np.ndarray,
    before_pos: np.ndarray,
    before_active: np.ndarray,
    int_pos: np.ndarray,
    int_active: np.ndarray,
    pos: np.ndarray,
    active: np.ndarray,
    shot: int,
) -> int | None:
    s = stone_of(shot)
    house = bool(active[s] and in_house(pos[s]))
    if task == "through":
        return None
    if task == "last":
        sign = 1 if team_of(shot) == 0 else -1
        now, want, got = (
            sign * end_score(p, a)
            for p, a in (
                (before_pos, before_active),
                (int_pos, int_active),
                (pos, active),
            )
        )
        return 4 if got >= want else 2 if got > now else 0
    if task == "draw":
        if not house:
            return 0
        d = dist(pos[s], int_pos[s])
        return 4 if d <= STONE_WIDTH else 3 if d <= 2 * STONE_WIDTH else 2
    if task == "freeze":
        t = targets[0]
        if not (active[s] and active[t]):
            return 0
        gap = dist(pos[s], pos[t]) - STONE_WIDTH
        return 4 if gap <= FREEZE_GAP[0] else 2 if gap <= FREEZE_GAP[1] else 0
    if task == "guard":
        live = targets[active[targets] & (pos[targets, 1] > pos[s, 1])]
        if not active[s] or house or not live.size:
            return 0
        lateral = np.abs(pos[live, 0] - pos[s, 0]).min()
        return 4 if lateral <= STONE_WIDTH else 2 if lateral <= 2 * STONE_WIDTH else 0
    if task == "front":
        if not active[s] or house:
            return 0
        past = pos[s, 1] - HOG_LINE_Y
        return 4 if past >= FRONT_PAST_HOG[0] else 3 if past >= FRONT_PAST_HOG[1] else 2
    out = int((~active[targets]).sum())
    if task in ("takeout", "hit_roll"):
        if out < len(targets):
            return 0
        if not active[s]:
            return 2
        tol = STONE_WIDTH / 2 if task == "takeout" else STONE_WIDTH
        return 4 if dist(pos[s], int_pos[s]) <= tol else 3
    if task == "double":
        return 4 if out >= 2 else 2 if out else 0
    if task == "clearing":
        return 4 if out == len(targets) else 2 if out else 0
    ok = active[targets] & (
        np.nan_to_num(dist(pos[targets], int_pos[targets]), nan=np.inf) <= STONE_WIDTH
    )
    return 4 if ok.all() else 2 if ok.any() else 0


def in_fgz(pos: np.ndarray) -> np.ndarray:
    y = pos[..., 1]
    return (y > HOG_LINE_Y) & (y < TEE_LINE_Y) & ~in_house(pos)


def in_four_foot(pos: np.ndarray) -> np.ndarray:
    return dist(pos, TEE) <= FOUR_FOOT + STONE_RADIUS


def situation(team: int, end_no: int, ends: int, diff: int) -> str:
    hammer = team == HAMMER
    if end_no == ends:
        return (
            ("defence" if diff >= 0 else "offence")
            if hammer
            else ("offence" if diff <= 0 else "defence")
        )
    if end_no <= EARLY_END:
        return "offence" if hammer else "defence"
    if hammer:
        return "defence" if diff >= LEAD else "offence"
    return "offence" if diff < 0 else "defence"


def double_exposed(pos: np.ndarray, active: np.ndarray, s: int) -> bool:
    mates = np.flatnonzero(
        active & (np.arange(N_STONES) // STONES_PER_TEAM == s // STONES_PER_TEAM)
    )
    mates = mates[mates != s]
    d = dist(pos[mates], pos[s])
    cos = np.abs(pos[mates, 1] - pos[s, 1]) / np.maximum(d, 1e-9)
    return bool((d < DOUBLE_SIDE + DOUBLE_LINE * cos).any())


def strategy_rules(
    task: str,
    before_pos: np.ndarray,
    before_active: np.ndarray,
    int_pos: np.ndarray,
    int_active: np.ndarray,
    pos: np.ndarray,
    active: np.ndarray,
    shot: int,
    end_no: int,
    ends: int,
    diff: int,
) -> tuple[str, list[dict]]:
    team, s = team_of(shot), stone_of(shot)
    hammer, final = team == HAMMER, end_no == ends
    sign = 1 if team == 0 else -1
    label = situation(team, end_no, ends, diff)
    opp = np.arange(N_STONES) // STONES_PER_TEAM != team
    out = []

    def rule(rid: str, ok: bool):
        text, source = RULES[rid]
        out.append({"id": rid, "ok": bool(ok), "text": text, "source": source})

    if shot == N_SHOTS - 1:
        got = sign * end_score(pos, active)
        rule("E1", got >= 2 or got == 0 or (got == 1 and final and diff >= 0))
        return label, out
    b_fgz = before_active & opp & in_fgz(before_pos)
    b_house = before_active & opp & in_house(before_pos)
    b_centre = b_fgz & (np.abs(before_pos[:, 0]) <= CORRIDOR)
    rule("S1", (task in OFFENCE) == (label == "offence"))
    if task in ("guard", "front") and diff != 0:
        rule("S7", diff < 0)
    count = sign * end_score(pos, active)
    if count:
        rule("A1", count > 0)
    if active[s]:
        centre = abs(pos[s, 0]) <= CORRIDOR
        if hammer and in_fgz(pos[s]):
            rule("P1", not centre)
        if not hammer and (in_fgz(pos[s]) or in_four_foot(pos[s])):
            rule("P2", centre)
        if not hammer and in_four_foot(pos[s]):
            rule("P6", pos[s, 1] < TEE_LINE_Y)
        if np.delete(active & ~opp, s).any():
            rule("P7", not double_exposed(pos, active, s))
    if label == "defence" and not hammer and b_house.any():
        rule("R1", (b_house & ~int_active).any())
    if hammer and shot >= FGZ_SHOTS and b_centre.any():
        cleared = ~int_active | (
            np.abs(np.nan_to_num(int_pos[:, 0], nan=np.inf)) > CORRIDOR
        )
        rule("R6", (b_centre & cleared).any())
    if (
        label == "defence"
        and shot >= FGZ_SHOTS
        and b_fgz.sum() == 1
        and b_house.sum() == 1
    ):
        rule("R7", (b_fgz & ~int_active).any())
    if end_no >= LATE_END and diff >= LEAD and task == "through":
        rule("R5", True)
    if shot < FGZ_SHOTS and b_fgz.any():
        rule("FGZ", not (b_fgz & ~active).any())
    on_line = b_fgz & (np.abs(before_pos[:, 0]) <= STONE_RADIUS)
    if shot < FGZ_SHOTS and on_line.any():
        kept = (
            active
            & in_fgz(pos)
            & (np.abs(np.nan_to_num(pos[:, 0], nan=np.inf)) <= STONE_RADIUS)
        )
        rule("NoTick", (on_line <= kept).all())
    if final and not hammer and shot == 0:
        centre_guard = task in ("guard", "front") and abs(int_pos[s, 0]) <= CORRIDOR
        four_foot = task == "draw" and in_four_foot(int_pos[s])
        if diff <= 0:
            rule("E4", centre_guard)
        elif diff == 1:
            rule("E4", centre_guard or four_foot)
        else:
            rule("E4", four_foot or task == "through")
    return label, out


def rate_shot(
    before_pos: np.ndarray,
    before_active: np.ndarray,
    int_pos: np.ndarray,
    int_active: np.ndarray,
    pos: np.ndarray,
    active: np.ndarray,
    shot: int,
    v_now: float,
    v_best: float,
    v_intended: float,
    v_actual: float,
    end_no: int = 1,
    ends: int = 10,
    score_diff: int = 0,
) -> dict:
    diff = score_diff if team_of(shot) == 0 else -score_diff
    task, targets = shot_type(before_pos, before_active, int_pos, int_active, shot)
    execution = execution_score(
        task, targets, before_pos, before_active, int_pos, int_active, pos, active, shot
    )
    label, rules = strategy_rules(
        task,
        before_pos,
        before_active,
        int_pos,
        int_active,
        pos,
        active,
        shot,
        end_no,
        ends,
        diff,
    )
    strategy = 4 * sum(r["ok"] for r in rules) / len(rules) if rules else None
    loss = max(float(v_best - v_intended), 0.0)
    parts = [
        (w, x)
        for w, x in zip(
            WEIGHTS,
            (
                None if execution is None else execution / 4,
                None if strategy is None else strategy / 4,
                float(np.exp(-loss)),
            ),
        )
        if x is not None
    ]
    return {
        "shot": shot,
        "team": team_of(shot),
        "task": task,
        "targets": targets.tolist(),
        "execution": execution,
        "situation": label,
        "strategy": strategy,
        "rules": rules,
        "decision_loss": loss,
        "execution_delta": float(v_actual - v_intended),
        "shot_value": float(v_actual - v_now),
        "overall": 100 * sum(w * x for w, x in parts) / sum(w for w, _ in parts),
    }


def end_summary(ratings: list[dict], score: int, end_no: int = 1) -> dict:
    out = {"shot_pct": {}, "strategy_pct": {}, "rules": []}
    for team in (0, 1):
        mine = [r for r in ratings if r["team"] == team]
        pts = [r["execution"] for r in mine if r["execution"] is not None]
        out["shot_pct"][team] = 100 * sum(pts) / (4 * len(pts)) if pts else None
        checks = [c["ok"] for r in mine for c in r["rules"]]
        out["strategy_pct"][team] = 100 * sum(checks) / len(checks) if checks else None
    hammer_score = score if HAMMER == 0 else -score
    ok = None
    if hammer_score < 0 or hammer_score == 1:
        ok = True
    elif hammer_score >= 2 and end_no > EARLY_END:
        ok = False
    if ok is not None:
        text, source = RULES["E1n"]
        out["rules"].append(
            {"id": "E1n", "team": 1 - HAMMER, "ok": ok, "text": text, "source": source}
        )
    return out


def observation(
    pos: np.ndarray,
    active: np.ndarray,
    shot: int,
    score_diff: int = 0,
    ends_left: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    sign = np.where(np.arange(N_STONES) < STONES_PER_TEAM, 1.0, -1.0)
    xy = np.nan_to_num(pos) / np.array([SHEET_WIDTH / 2, 1.0])
    xy[:, 1] = (xy[:, 1] - TEE_LINE_Y) / HOUSE_RADIUS
    stones = np.stack(
        [active.astype(float), sign * active, xy[:, 0] * active, xy[:, 1] * active], -1
    )
    context = np.array(
        [shot / (N_SHOTS - 1), float(team_of(shot) == 1), score_diff / 8, ends_left],
        float,
    )
    return stones.astype(np.float32), context.astype(np.float32)


class End:
    def __init__(
        self,
        seed: int | None = None,
        noise: bool = True,
        end_no: int = 1,
        ends: int = 10,
        score_diff: int = 0,
    ):
        self.end_no, self.ends, self.score_diff = end_no, ends, score_diff
        self.sim = Simulator()
        self.player = Player(seed=seed) if noise else Player("identical")
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.pos = np.full((N_STONES, 2), np.nan)
        self.active = np.zeros(N_STONES, bool)
        self.shot = 0
        return self.observe()

    @property
    def done(self) -> bool:
        return self.shot >= N_SHOTS

    @property
    def team(self) -> int:
        return team_of(self.shot)

    def observe(self) -> tuple[np.ndarray, np.ndarray]:
        return observation(self.pos, self.active, min(self.shot, N_SHOTS - 1))

    def score(self) -> int:
        return end_score(self.pos, self.active)

    def shot_to(self, target, speed: float, spin: float):
        try:
            return self.sim.calculate_shot(target, speed, spin)
        except RuntimeError:
            return None

    def targeted(self) -> list[tuple[float, float, float]]:
        out = []
        for spin in (HANDLE, -HANDLE):
            out.append(self.shot_to(TEE, 0.0, spin))
            gy = self.rng.uniform(*GUARD_Y)
            out.append(self.shot_to((self.rng.uniform(-0.6, 0.6), gy), 0.0, spin))
        for i in np.flatnonzero(self.active):
            if (i < STONES_PER_TEAM) != (self.team == 0):
                out.append(
                    self.shot_to(
                        self.pos[i], TAKEOUT_SPEED, self.rng.choice([HANDLE, -HANDLE])
                    )
                )
        return [a for a in out if a is not None]

    def random(self, n: int) -> list[tuple[float, float, float]]:
        return [
            (
                float(self.rng.uniform(*SPEED_RANGE)),
                float(self.rng.choice([HANDLE, -HANDLE])),
                float(np.pi / 2 + self.rng.uniform(-ANGLE_RANGE, ANGLE_RANGE)),
            )
            for _ in range(n)
        ]

    def candidates(self, k: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        acts = self.targeted()
        self.rng.shuffle(acts)
        acts = acts[: k // 2] + self.random(k - min(len(acts), k // 2))
        acts = np.array(acts[:k])
        nxt_pos, nxt_active = [], []
        for a in acts:
            p, act, _ = self.sim.throw(self.pos, self.active, stone_of(self.shot), *a)
            nxt_pos.append(p)
            nxt_active.append(act)
        return acts, np.stack(nxt_pos), np.stack(nxt_active)

    def step(self, action) -> tuple[tuple[np.ndarray, np.ndarray], int, bool]:
        speed, spin, angle = self.player.play(*[float(x) for x in action])
        self.pos, self.active, _ = self.sim.throw(
            self.pos, self.active, stone_of(self.shot), speed, spin, angle
        )
        self.shot += 1
        return self.observe(), self.score() if self.done else 0, self.done

    def rated_step(self, net, acts, npos, nact, j: int):
        import torch

        from curling.model.trainer import utilities

        before_pos, before_active, shot = self.pos.copy(), self.active.copy(), self.shot
        sign = 1.0 if self.team == 0 else -1.0
        out = self.step(acts[j])
        layouts = [(before_pos, before_active, shot)]
        layouts += [(p, a, shot + 1) for p, a in zip(npos, nact)]
        layouts += [(self.pos, self.active, shot + 1)]
        obs = [observation(p, a, min(t, N_SHOTS - 1)) for p, a, t in layouts]
        stones = torch.as_tensor(np.stack([o[0] for o in obs]))
        context = torch.as_tensor(np.stack([o[1] for o in obs]))
        with torch.no_grad():
            v = sign * utilities(net, stones, context).numpy()
        if self.done:
            v[1:] = [sign * end_score(p, a) for p, a, _ in layouts[1:]]
        rating = rate_shot(
            before_pos,
            before_active,
            npos[j],
            nact[j],
            self.pos,
            self.active,
            shot,
            v[0],
            v[1:-1].max(),
            v[1 + j],
            v[-1],
            self.end_no,
            self.ends,
            self.score_diff,
        )
        return out, rating


class CurlingEnv(gym.Env):
    def __init__(self, seed: int | None = None, noise: bool = True):
        self.end = End(seed, noise)
        self.observation_space = gym.spaces.Dict(
            {
                "stones": gym.spaces.Box(-np.inf, np.inf, (N_STONES, 4), np.float32),
                "context": gym.spaces.Box(-np.inf, np.inf, (4,), np.float32),
            }
        )
        self.action_space = gym.spaces.Box(
            ACTION_LOW.astype(np.float32), ACTION_HIGH.astype(np.float32)
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        stones, context = self.end.reset()
        return {"stones": stones, "context": context}, {"team": self.end.team}

    def step(self, action):
        (stones, context), reward, done = self.end.step(action)
        return (
            {"stones": stones, "context": context},
            float(reward),
            done,
            False,
            {"team": self.end.team if not done else None},
        )
