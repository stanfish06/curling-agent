import torch
from torch import nn

N_STONES = 16
STONE_FEATURES = 4
GLOBAL_FEATURES = 4
MAX_SCORE = 8
ACTION_DIM = 3


class StoneEncoder(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(STONE_FEATURES, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
        )
        self.edge = nn.Sequential(
            nn.Linear(2 * width + 3, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
        )
        self.update = nn.Sequential(nn.Linear(2 * width, width), nn.ReLU())

    def forward(self, stones: torch.Tensor) -> torch.Tensor:
        n = stones.shape[1]
        present = stones[..., :1]
        xy = stones[..., 2:4]
        h = self.phi(stones) * present
        rel = xy[:, None, :, :] - xy[:, :, None, :]
        dist = rel.norm(dim=-1, keepdim=True)
        direction = rel / dist.clamp(min=1e-6)
        pair = present[:, :, None, :] * present[:, None, :, :]
        pair = pair * (1 - torch.eye(n, device=stones.device))[None, :, :, None]
        hi = h[:, :, None, :].expand(-1, -1, n, -1)
        hj = h[:, None, :, :].expand(-1, n, -1, -1)
        m = self.edge(torch.cat([hi, hj, direction, dist], -1)) * pair
        h = self.update(torch.cat([h, m.sum(2)], -1)) * present
        return torch.cat([h.max(1).values, h.sum(1) / present.sum(1).clamp(min=1)], -1)


class PolicyValueNet(nn.Module):
    def __init__(self, width: int = 128):
        super().__init__()
        self.encoder = StoneEncoder(width)
        self.trunk = nn.Sequential(
            nn.Linear(2 * width + GLOBAL_FEATURES, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
        )
        self.score = nn.Sequential(
            nn.Linear(width + 2 * width + ACTION_DIM, width),
            nn.ReLU(),
            nn.Linear(width, 1),
        )
        self.value = nn.Linear(width, 2 * MAX_SCORE + 1)

    def forward(
        self,
        stones: torch.Tensor,
        context: torch.Tensor,
        actions: torch.Tensor,
        next_stones: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(torch.cat([self.encoder(stones), context], -1))
        k = actions.shape[1]
        z_next = self.encoder(next_stones.flatten(0, 1)).unflatten(0, (-1, k))
        logits = self.score(
            torch.cat([h[:, None].expand(-1, k, -1), z_next, actions], -1)
        ).squeeze(-1)
        return logits, self.value(h)
