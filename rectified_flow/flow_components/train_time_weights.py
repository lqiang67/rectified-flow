import torch


class TrainTimeWeights:
    def __init__(
        self,
        weight: str = "uniform",
    ):
        self.weight = weight

    def __call__(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if self.weight == "uniform":
            wts = torch.ones_like(t)
        else:
            raise NotImplementedError(f"Time weight '{self.weight}' is not implemented.")
        
        return wts