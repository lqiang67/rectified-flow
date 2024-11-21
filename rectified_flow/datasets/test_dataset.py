import pytest
import torch
from torch.utils.data import Dataset
from coupling_dataset import CouplingDataset, RectifiedFlowDataloader, ReflowDataloader, DataPromptDataset


def test_rectified_flow_without_prompts():
    # Rectified flow training without prompts
    dataset = CouplingDataset(D1=torch.randn(100, 784))
    dataloader = RectifiedFlowDataloader(dataset, batch_size=64)

    X0, X1 = next(iter(dataloader))
    assert X0.shape == X1.shape, "X0 and X1 shapes do not match"


def test_rectified_flow_with_prompts():
    # Rectified flow training with prompts
    
    D1 = DataPromptDataset(data=torch.randn(100, 784), prompts=[0 for _ in range(100)])
    dataset = CouplingDataset(D1=D1)
    dataloader = RectifiedFlowDataloader(dataset, batch_size=64)

    X0, X1, prompts = next(iter(dataloader))
    assert X0.shape == X1.shape, "X0 and X1 shapes do not match"
    assert X0.shape[0] == len(prompts)


def test_reflow_without_prompts():
    # Reflow without prompts
    dataset = CouplingDataset(
        D0=torch.randn(100, 784), 
        D1=torch.randn(100, 784), 
        reflow=True
    )
    dataloader = ReflowDataloader(dataset, batch_size=64)

    X0, X1 = next(iter(dataloader))
    assert X0.shape == X1.shape, "X0 and X1 shapes do not match"


def test_reflow_with_prompts():
    # Reflow with prompts
    dataset = CouplingDataset(
        D0=torch.randn(100, 784), 
        D1=DataPromptDataset(data=torch.randn(100, 784), prompts=[0 for _ in range(100)]),
        reflow=True
    )
    dataloader = ReflowDataloader(dataset, batch_size=64)

    X0, X1, prompts = next(iter(dataloader))
    assert X0.shape == X1.shape, "X0 and X1 shapes do not match"
    assert X0.shape[0] == len(prompts)