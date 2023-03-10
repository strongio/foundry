import torch


class BSpline:
    """
    TODO: this includes a bias term
    """
    def __init__(self,
                 num_knots: int = 3,
                 degree: int = 3):
        self.num_knots = num_knots
        self.raw_knots = torch.nn.Parameter(torch.linspace(-1, 1, num_knots))
        self.degree = degree
        self.x_range_ = None

    @staticmethod
    def get_all_knots(base_knots: torch.Tensor, degree: int) -> torch.Tensor:
        base_knots = torch.sort(base_knots).values
        base_knots = torch.cat([torch.zeros(1), base_knots, torch.ones(1)])

        dist_min = base_knots[1] - base_knots[0]
        dist_max = base_knots[-1] - base_knots[-2]

        knots = torch.cat([
            torch.linspace(
                (base_knots[0] - degree * dist_min).item(),
                (base_knots[0] - dist_min).item(),
                steps=degree,
            ),
            base_knots,
            torch.linspace(
                (base_knots[-1] + dist_max).item(),
                (base_knots[-1] + degree * dist_max).item(),
                steps=degree,
            )
        ])

        return knots

    @property
    def all_knots(self) -> torch.Tensor:
        return self.get_all_knots(torch.sigmoid(self.raw_knots), degree=self.degree)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.x_range_ is None:
            self.x_range_ = (x.min(), x.max())

        # scale to 0-1:
        x = (x - self.x_range_[0]) / (self.x_range_[1] - self.x_range_[0])

        n = len(self.all_knots) - self.degree - 1
        above_mask = x > self.x_range_[1]
        below_mask = x < self.x_range_[0]
        inside_mask = ~above_mask & ~below_mask
        at_boundaries = torch.stack([self._make_col(torch.as_tensor(self.x_range_), i) for i in range(n)], 1)

        out = torch.zeros(x.shape + (n,), dtype=x.dtype, device=x.device)
        out[inside_mask] = torch.stack([self._make_col(x[inside_mask], i) for i in range(n)], 1)
        out[above_mask] = at_boundaries[1]
        out[below_mask] = at_boundaries[0]
        return out

    def _make_col(self, x: torch.Tensor, i: int, k: int = None):
        if k is None:
            k = self.degree

        if k == 0:
            out = torch.zeros_like(x)
            out[(self.all_knots[i] <= x) & (x < self.all_knots[i + 1])] = 1.0
            return out

        if self.all_knots[i + k] == self.all_knots[i]:
            c1 = torch.zeros_like(x)
        else:
            c1 = (x - self.all_knots[i]) / (self.all_knots[i + k] - self.all_knots[i]) * self._make_col(x, i=i, k=k - 1)

        if self.all_knots[i + k + 1] == self.all_knots[i + 1]:
            c2 = torch.zeros_like(x)
        else:
            c2 = (
                    (self.all_knots[i + k + 1] - x) / (self.all_knots[i + k + 1] - self.all_knots[i + 1])
                    * self._make_col(x, i=i + 1, k=k - 1)
            )

        return c1 + c2

    @torch.no_grad()
    def as_dataframe(self, x):
        import pandas as pd
        mat = self(x)
        df = pd.DataFrame(mat.numpy())
        df.columns = [f"x{i}" for i in range(df.shape[1])]
        df['x_orig'] = x.numpy()
        return df.melt(id_vars=['x_orig'])
