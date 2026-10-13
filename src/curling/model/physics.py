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


class Physics(nn.Module):
    def __init__(
        self, params: dict[str, float] = FCV1, restitution: float = FCV1_RESTITUTION
    ):
        super().__init__()
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
        self, vel: torch.Tensor, spin: torch.Tensor, dt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        speed = vel.norm(dim=-1)
        moving = speed > 1e-7
        s = torch.where(moving, speed, torch.ones_like(speed))
        new_speed = (speed - self.friction(s) * G * dt).clamp(min=0)
        yaw = self.yaw_rate(s, spin) * dt
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
        vel, spin = self.slide(vel, spin, h)
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
