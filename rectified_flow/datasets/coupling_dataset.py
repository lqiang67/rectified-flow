import torch
import torch.distributions as dist
from torch.utils.data import Dataset, DataLoader


class CouplingDataset(Dataset):
    def __init__(
        self, 
        D1,
        D0=None,
        reflow=False
    ):
        """
        A dataset that provides coupled samples from D0 and D1.

        Args:
            D1: Dataset or Tensor,
            D0: Dataset, Tensor, distribution for noise data samples.
                If None and reflow=False, defaults to standard normal.
            reflow: If True, D0 and D1 must have the same length and are paired.
        """
        if reflow:
            assert D0 is not None and len(D0) == len(D1), "D0 must exist, D0 and D1 must have the same length when reflow=True"

        self.D1 = D1
        self.D0 = D0 if D0 is not None else dist.Normal(0, 1)
        self.reflow = reflow

    def __len__(self):
        return len(self.D1)

    def __getitem__(self, index):
        X1 = self.D1[index]
        if self.reflow:
            X0 = self.D0[index]
        else:
            X0 = None  # we will leave to the dataloader to resolve this case
        return X0, X1


class DataPromptDataset(Dataset):
    def __init__(self, data, prompts):
        """
        Initialize the dataset with data and prompts.

        Args:
            data (torch.Tensor): A tensor of shape (num_data, data_dim).
            prompts (list): A list of length num_data containing prompt strings.
        """
        assert len(data) == len(prompts), "Data and prompts must have the same length."
        self.data = data
        self.prompts = prompts

    def __len__(self):
        """Return the number of data points."""
        return len(self.data)

    def __getitem__(self, index):
        """
        Retrieve a single data point and its corresponding prompt.

        Args:
            index (int): Index of the data point to retrieve.

        Returns:
            tuple: A tuple (data_point, prompt) where:
                - data_point is a tensor of shape (data_dim,).
                - prompt is a string corresponding to the data point.
        """
        data_point = self.data[index]
        prompt = self.prompts[index]
        return data_point, prompt


def rectified_flow_collate_fn(batch, D0):
    """
    Custom collate function to handle batch sampling for D0 when it is a distribution.
    """
    batch_size = len(batch)
    if isinstance(batch[0][1], tuple):
        X1 = torch.stack([item[1][0] for item in batch])
        prompts = [item[1][1] for item in batch]
    else:
        X1 = torch.stack([item[1] for item in batch])
        prompts = None

    # Sample X0 efficiently using batch sampling
    X0 = D0.sample(X1.shape)

    if prompts is None:
        return X0, X1
    else:
        return X0, X1, prompts


def reflow_collate_fn(batch):
    """
    Custom collate function to handle batch sampling for D0 when it is a distribution.
    """
    batch_size = len(batch)
    X0 = torch.stack([item[0] for item in batch])  # Collect X1 values

    if isinstance(batch[0][1], tuple):
        X1 = torch.stack([item[1][0] for item in batch])  # Collect X1 values
        prompts = [item[1][1] for item in batch]
    else:
        X1 = torch.stack([item[1] for item in batch])
        prompts = None

    if prompts is None:
        return X0, X1
    else:
        return X0, X1, prompts


class RectifiedFlowDataloader(DataLoader):
    def __init__(self, dataset, batch_size, **kwargs):
        """
        A DataLoader for the Rectified Flow setting, where D0 is assumed to be a distribution.
        
        Args:
            dataset: CouplingDataset with D0 as a distribution.
            batch_size: Number of samples per batch.
            kwargs: Additional arguments for the PyTorch DataLoader.
        """
        assert hasattr(dataset, "D0") and isinstance(dataset.D0, torch.distributions.Distribution), \
            "D0 must be a torch.distributions.Distribution for RectifiedFlowDataloader."
        super().__init__(dataset, batch_size=batch_size, collate_fn=lambda batch: rectified_flow_collate_fn(batch, dataset.D0), **kwargs)


class ReflowDataloader(DataLoader):
    def __init__(self, dataset, batch_size, **kwargs):
        assert dataset.reflow, "Reflow dataset should have reflow=True"
        super().__init__(dataset, batch_size=batch_size, collate_fn=reflow_collate_fn, **kwargs)