import torch
from torch import nn

G = 9.80665
STONE_MASS = 19.96
STONE_RADIUS = 0.145
SHEET_WIDTH = 4.75
HOG_LINE_Y = 32.004
TEE_LINE_Y = 38.405
BACK_LINE_Y = 40.234
HACK_Y = 42.063
BACK_BOARD_Y = 43.892
HOUSE_RADIUS = 1.829
MAX_SPEED = 4.0
HANDLE_SPIN = torch.pi / 2
DT = 0.001
NOISE_SPEED = 0.0076
NOISE_ANGLE = 0.0018
FCV1 = {
    "friction_a": 0.00200985,
    "friction_b": 0.06385782,
    "friction_c": 0.00626286,
    "yaw_k": 0.00820,
    "yaw_p": 0.8,
    "spin_k": 0.025,
    "contact_friction": 0.2,
    "time_scale": 1.0,
}
FCV1_RESTITUTION = 1.0


def release(
    speed: torch.Tensor, angle: torch.Tensor, handle: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    vel = speed[..., None] * torch.stack([torch.cos(angle), torch.sin(angle)], -1)
    return vel, HANDLE_SPIN * torch.sign(handle)


def out_of_sheet(pos: torch.Tensor) -> torch.Tensor:
    x, y = pos[..., 0], pos[..., 1]
    return (
        (x.abs() > SHEET_WIDTH / 2 - STONE_RADIUS)
        | (y > BACK_BOARD_Y - STONE_RADIUS)
        | (y < 0)
    )


def in_play(pos: torch.Tensor) -> torch.Tensor:
    x, y = pos[..., 0], pos[..., 1]
    return (
        (x.abs() <= SHEET_WIDTH / 2 - STONE_RADIUS)
        & (y > HOG_LINE_Y + STONE_RADIUS)
        & (y < BACK_LINE_Y + STONE_RADIUS)
    )


HEIGHT_UNIT = 1e-3


def _param(v, learn: bool, unit: float = 1.0) -> nn.Parameter:
    return nn.Parameter(
        torch.tensor(v, dtype=torch.float32) / unit, requires_grad=learn
    )


class Tilt(nn.Module):
    def __init__(
        self, x: float = 0.0, y: float = 0.0, learn: bool = False, unit: float = 1.0
    ):
        super().__init__()
        self.unit = unit
        self.slope = _param([x, y], learn, unit)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.unit * (self.slope[0] * x + self.slope[1] * y)

    def grad(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.unit * self.slope.expand(*x.shape, 2)


class Bump(nn.Module):
    def __init__(
        self,
        at: list[float],
        sigma: float,
        amp: float,
        learn: bool = False,
        unit: float = 1.0,
    ):
        super().__init__()
        self.unit = unit
        self.at = _param(at, learn)
        self.sigma = _param(sigma, learn)
        self.amp = _param(amp, learn, unit)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        d2 = (x - self.at[0]) ** 2 + (y - self.at[1]) ** 2
        return self.unit * self.amp * torch.exp(-d2 / (2 * self.sigma**2))

    def grad(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        d = torch.stack([x - self.at[0], y - self.at[1]], -1)
        return -self(x, y)[..., None] * d / self.sigma**2


class Sink(Bump):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -super().forward(x, y)


class Ridge(nn.Module):
    def __init__(
        self,
        axis: str,
        at: float,
        width: float,
        amp: float,
        learn: bool = False,
        unit: float = 1.0,
    ):
        super().__init__()
        self.axis = axis
        self.unit = unit
        self.at = _param(at, learn)
        self.width = _param(width, learn)
        self.amp = _param(amp, learn, unit)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        u = x if self.axis == "y" else y
        g = torch.exp(-((u - self.at) ** 2) / (2 * self.width**2))
        return self.unit * self.amp * g

    def grad(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        u = x if self.axis == "y" else y
        du = -self(x, y) * (u - self.at) / self.width**2
        zero = torch.zeros_like(du)
        return torch.stack([du, zero] if self.axis == "y" else [zero, du], -1)


class Noise(nn.Module):
    def __init__(
        self,
        seed: int,
        scale: float,
        amp: float,
        n: int = 16,
        learn: bool = False,
        unit: float = 1.0,
    ):
        super().__init__()
        self.unit = unit
        g = torch.Generator().manual_seed(seed)
        th = torch.rand(n, generator=g) * 2 * torch.pi
        self.register_buffer("k", torch.stack([th.cos(), th.sin()], -1))
        self.register_buffer("phi", torch.rand(n, generator=g) * 2 * torch.pi)
        self.scale = _param(scale, learn)
        self.amp = _param(amp, learn, unit)

    def phase(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        k = 2 * torch.pi / self.scale * self.k
        return x[..., None] * k[:, 0] + y[..., None] * k[:, 1] + self.phi

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n = self.k.shape[0]
        a = self.unit * self.amp * (2 / n) ** 0.5
        return a * torch.cos(self.phase(x, y)).sum(-1)

    def grad(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n = self.k.shape[0]
        k = 2 * torch.pi / self.scale * self.k
        s = torch.sin(self.phase(x, y))
        a = self.unit * self.amp * (2 / n) ** 0.5
        return -a * (s[..., None] * k).sum(-2)


PRIMITIVES = {"tilt": Tilt, "bump": Bump, "sink": Sink, "ridge": Ridge, "noise": Noise}
FIELDS = ("height", "friction", "curl")


class Surface(nn.Module):
    def __init__(self, spec: dict[str, list[dict]] | None = None):
        super().__init__()
        spec = spec or {}
        self.fields = nn.ModuleDict(
            {
                f: nn.ModuleList(
                    [
                        PRIMITIVES[k](**p, unit=HEIGHT_UNIT if f == "height" else 1.0)
                        for t in spec.get(f, [])
                        for k, p in t.items()
                    ]
                )
                for f in FIELDS
            }
        )

    def field(self, name: str, pos: torch.Tensor) -> torch.Tensor:
        x, y = pos[..., 0], pos[..., 1]
        out = torch.zeros_like(x)
        for m in self.fields[name]:
            out = out + m(x, y)
        return out

    def slope(self, pos: torch.Tensor) -> torch.Tensor:
        x, y = pos[..., 0], pos[..., 1]
        out = torch.zeros_like(pos)
        for m in self.fields["height"]:
            out = out + m.grad(x, y)
        return out

    def friction(self, pos: torch.Tensor) -> torch.Tensor:
        return 1 + self.field("friction", pos)

    def curl(self, pos: torch.Tensor) -> torch.Tensor:
        return 1 + self.field("curl", pos)


class Physics(nn.Module):
    def __init__(
        self,
        params: dict[str, float] = FCV1,
        restitution: float = FCV1_RESTITUTION,
        surface: Surface | None = None,
    ):
        super().__init__()
        self.surface = surface or Surface()
        self.log = nn.ParameterDict(
            {k: nn.Parameter(torch.tensor(v).log()) for k, v in params.items()}
        )
        self.restitution_logit = nn.Parameter(
            torch.tensor(restitution).clamp(1e-4, 1 - 1e-4).logit()
        )

    def __getattr__(self, name: str):
        if name in FCV1:
            return self.log[name].exp()
        return super().__getattr__(name)

    @property
    def restitution(self) -> torch.Tensor:
        return torch.sigmoid(self.restitution_logit)

    def friction(self, speed: torch.Tensor) -> torch.Tensor:
        return self.friction_a / (speed + self.friction_b) + self.friction_c

    def yaw_rate(self, speed: torch.Tensor, spin: torch.Tensor) -> torch.Tensor:
        return torch.sign(spin) * self.yaw_k * speed.clamp(min=1e-6).pow(-self.yaw_p)

    def spin_decay(self, speed: torch.Tensor) -> torch.Tensor:
        return self.spin_k / speed.clamp(min=0.001)

    def slide(
        self, pos: torch.Tensor, vel: torch.Tensor, spin: torch.Tensor, dt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vel = vel - G * self.surface.slope(pos) * dt
        speed = vel.norm(dim=-1)
        moving = speed > 1e-7
        s = torch.where(moving, speed, torch.ones_like(speed))
        new_speed = (
            speed - self.friction(s) * self.surface.friction(pos) * G * dt
        ).clamp(min=0)
        yaw = self.yaw_rate(s, spin) * self.surface.curl(pos) * dt
        e_long = vel / s[..., None]
        e_trans = torch.stack([-e_long[..., 1], e_long[..., 0]], -1)
        new_vel = new_speed[..., None] * (
            torch.cos(yaw)[..., None] * e_long + torch.sin(yaw)[..., None] * e_trans
        )
        new_vel = torch.where(moving[..., None], new_vel, torch.zeros_like(new_vel))
        decay = self.spin_decay(s) * dt
        new_spin = torch.where(
            spin.abs() <= decay, torch.zeros_like(spin), spin - decay * torch.sign(spin)
        )
        return new_vel, new_spin

    def collide(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        spin: torch.Tensor,
        active: torch.Tensor,
        passes: int = 8,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = pos.shape[-2]
        iu, ju = torch.triu_indices(n, n, 1, device=pos.device)
        for _ in range(passes):
            d = pos[..., ju, :] - pos[..., iu, :]
            dist = d.norm(dim=-1)
            hit = (dist < 2 * STONE_RADIUS) & active[..., iu] & active[..., ju]
            if not hit.any():
                break
            nrm = d / dist.clamp(min=1e-9)[..., None]
            tan = torch.stack([-nrm[..., 1], nrm[..., 0]], -1)
            rel = vel[..., ju, :] - vel[..., iu, :]
            rel_contact = (
                rel - STONE_RADIUS * (spin[..., ju] + spin[..., iu])[..., None] * tan
            )
            vn = (rel_contact * nrm).sum(-1)
            vt = (rel_contact * tan).sum(-1)
            approaching = hit & (vn < 0)
            jn = torch.where(
                approaching,
                -(1 + self.restitution) * vn * STONE_MASS / 2,
                torch.zeros_like(vn),
            )
            jt = torch.maximum(
                torch.minimum(-vt * STONE_MASS / 6, self.contact_friction * jn),
                -self.contact_friction * jn,
            )
            impulse = jn[..., None] * nrm + jt[..., None] * tan
            push = (
                torch.where(hit, (2 * STONE_RADIUS - dist) / 2, torch.zeros_like(dist))[
                    ..., None
                ]
                * nrm
            )
            dspin = -2 * jt / (STONE_MASS * STONE_RADIUS)
            vel = vel.index_add(-2, iu, -impulse / STONE_MASS).index_add(
                -2, ju, impulse / STONE_MASS
            )
            spin = spin.index_add(-1, iu, dspin).index_add(-1, ju, dspin)
            pos = pos.index_add(-2, iu, -push).index_add(-2, ju, push)
        return pos, vel, spin

    def step(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        spin: torch.Tensor,
        active: torch.Tensor,
        dt: float = DT,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = dt * self.time_scale
        vel, spin = self.slide(pos, vel, spin, h)
        pos = pos + vel * h
        pos, vel, spin = self.collide(pos, vel, spin, active)
        active = active & ~out_of_sheet(pos)
        vel = vel * active[..., None]
        spin = spin * active
        return pos, vel, spin, active

    def forward(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        spin: torch.Tensor,
        active: torch.Tensor,
        dt: float = DT,
        max_steps: int = 60000,
        record_every: int = 0,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None
    ]:
        frames = []
        for i in range(max_steps):
            if record_every and i % record_every == 0:
                frames.append(pos)
            pos, vel, spin, active = self.step(pos, vel, spin, active, dt)
            if not (vel.abs() > 1e-7).any():
                break
        if record_every:
            frames.append(pos)
            return pos, vel, spin, active, torch.stack(frames)
        return pos, vel, spin, active, None
