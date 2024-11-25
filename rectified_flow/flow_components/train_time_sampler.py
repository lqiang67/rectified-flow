import torch


class TrainTimeSampler:
    def __init__(
        self,
        distribution: str = "uniform",
    ):
        self.distribution = distribution

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Sample time tensor for training

        Returns:
            torch.Tensor: Time tensor, shape (batch_size,)
        """
        if self.distribution == "uniform":
            t = torch.rand((batch_size,), device=device, dtype=dtype)
        else: # NOTE: will implement different time distributions
            raise NotImplementedError(f"Time distribution '{self.dist}' is not implemented.")
        
        return t


