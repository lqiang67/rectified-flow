<div align="center">

<img src="https://github.com/lqiang67/rectified-flow/blob/main/assets/logo_header.png?raw=true" alt="Logo" style="width: 100%; height: auto;">

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) 
[![Blog](https://img.shields.io/badge/blog-blue)](https://rectifiedflow.github.io)
[![Lecture Notes](https://img.shields.io/badge/lecture%20notes-blue)](https://www.cs.utexas.edu/~lqiang/PDF/flow_book.pdf)
[![Email](https://img.shields.io/badge/email-blue)](mailto:rectifiedflow@gmail.com)

______________________________________________________________________
</div>


## Overview 

**RectifiedFlow** is a simple, unified PyTorch codebase for diffusion and flow models. It offers an easy-to-use platform for training and inference, focusing on simplicity, flexibility, and quick prototyping. The library includes:

- **Companion Resources**: Includes a [![Blog](https://img.shields.io/badge/blog-blue)](https://rectifiedflow.github.io), [![Lecture Notes](https://img.shields.io/badge/lecture%20notes-blue)](https://www.cs.utexas.edu/~lqiang/PDF/flow_book.pdf), and beginner-friendly [![Notebooks](https://img.shields.io/badge/Notebooks-orange)](https://github.com/lqiang67/rectified-flow/tree/main/examples) covering concepts from basics to advanced implementations.

- **Unified ODE Framework**: Train and infer rectified flow (RF) and diffusion models using a unified ODE approach, including 1-rectified flow from data (flow matching), reflow for speedup, diffusion as RF+Langevin, post-training conversion of affine interpolation schemes, analytic models, etc. 

- **Symbolic Algorithm Derivation**: We use a symbolic solver for affine interpolation to automate the derivation of algorithms and formulas, enabling easy model conversion between various forms like score functions, velocity fields, and noise predictions. This eliminates the need for manual derivation in both existing and new algorithms (e.g., the DDIM/DDPM coefficients). 

- **Easy Integration with SOTA Models**: Easily integrate state-of-the-art models, including the Flux series, for greater flexibility and compatibility.

---

# Contents
- [Installation](#installation)
- [Getting Started](#getting-started)
  - [Rectified Flow: A One-Minute Introduction](#rectified-flow-a-one-minute-introduction)
  - [A Quick Walkthrough](#a-quick-walkthrough)
  - [Interactive Tutorials](#interactive-tutorials)
  - [Training Pipelines](#training-pipelines)
- [Customize Your Component](#customize-your-component)
  - [Interpolation](#interpolation)
  - [Wrapping a New Velocity](#wrapping-a-new-velocity)
  - [Training Utilities](#training-utilities)
  - [Customized Samplers](#customized-samplers)
- [License](#license)


# Installation

You can install the `rectified-flow` package using `pip`:

```
pip install rectified-flow
```

Alternatively, you can install the package from source. Please run the following commands in the given order to install the dependency.

```
conda create -n rf python=3.10
conda activate rf
git clone https://github.com/lqiang67/rectified-flow.git
cd rectified-flow
pip install -r requirements.txt
```

Then install the `rectified-flow` package:

```
pip install -e .
```

# Getting Started

## **Rectified Flow: A One-Minute Introduction**

<div align="center">
<img src="https://github.com/lqiang67/rectified-flow/blob/main/assets/rf_toy_illustration.png?raw=true" alt="Header" style="width: 100%; height: auto;">
</div>

Consider the task of learning an ODE model $\mathrm d Z_t = v_t(Z_t)\mathrm d t$ that transforms a noise distribution $X_0 \sim \pi_0$ into a data distribution $X_1 \sim \pi_1$. We begin by drawing random pairs $(X_0, X_1)$, where $X_0$ and $X_1$ are independent by default, and then construct the interpolation $X_t = t X_1 + (1 - t) X_0.$ The rectified flow velocity is learned by minimizing

$$
\min_v \mathbb{E}_{X_0, X_1, t} \left[ \left\lVert \frac{\mathrm d}{\mathrm d t} X_t - v_t(X_t, t) \right\rVert^2 \right]
$$

where $t \sim \text{Uniform}([0, 1])$ and $\frac{\mathrm d}{\mathrm d t} X_t = X_1 - X_0$.

After training the model $v_t$, we can solve the ODE $\mathrm d Z_t = v_t(Z_t)  \mathrm d t$ with the initial condition $Z_0 \sim \pi_0$. This yields a set of pairs $(Z_0, Z_1)$, which can be treated as new data pairs $(X_0, X_1)$ to train a new model $v_t^{\text{reflow}}$. This "reflowed" model is known to produce straighter trajectories, allowing the ODE to be solved with fewer Euler steps and larger step sizes.

Although ultimately unnecessary in theory (see Chapter 3 of the lecture notes), the codebase supports a more general affine interpolation $X_t = \alpha_t X_1 + \beta_t X_0$ to ensure compatibility with existing algorithms.


## A Quick Walkthrough
The `RectifiedFlow` class serves as an **intermediary** for your training and inference processes. Each different velocity field should instantiate a separate `RectifiedFlow` class.

```python
from rectified_flow.rectified_flow import RectifiedFlow

model = YourVelocityFieldModel()

# Initialize RectifiedFlow with custom settings
rectified_flow = RectifiedFlow(
    data_shape=(32, 32),
    velocity_field=model,
    interp="straight",
    source_distribution="normal",
    is_independent_coupling=True,
    train_time_distribution="uniform",
    train_time_weight="uniform",
    criterion="mse",
    device=device,
)

# Or use the default settings
rectified_flow = RectifiedFlow(
    data_shape=(32, 32),
    velocity_field=model,
    device=device,
)
```

During training, you can easily compute the predefined loss by passing your target data samples `x_1`. If samples from source distribution `x_0` is not provided, it will be sampled by default from standard Gaussian. The `RectifiedFlow` class supports various pre-specified loss functions and interpolation methods, and it calculates the loss accordingly.

```python
loss = rectified_flow.get_loss(x_0=None, x_1=x_1, **kwargs)
```
This is implemented by:
```python
# Step 1: Interpolation
x_t, dot_x_t = self.get_interpolation(x_0, x_1, t)

# Step 2: Velocity Calculation
v_t = self.get_velocity(x_t, t, **kwargs)

# Step 3: Time Weights
time_weights = self.train_time_weight(t)

# Step 4: Loss Computation
return self.criterion(
    v_t=v_t,
    dot_x_t=dot_x_t,
    x_t=x_t,
    t=t,
    time_weights=time_weights,
)

```

After training, converting a pretrained rectified flow to another interpolation scheme (as long as alpha and beta are specified) can be done easily and automatically by:
```python
from rectified_flow.flow_components.interpolation_convertor import AffineInterpConverter

# Converting pretrained rf into spherical one
target_interp = AffineInterp("spherical")
converted_spherical_rf = AffineInterpConverter(rf, target_interp).transform_rectified_flow()
```

For sampling, import the desired sampler class and pass the `RectifiedFlow` instance to it.
```python
from rectified_flow.samplers import SDESampler

sde_sampler = SDESampler(rectified_flow=rectified_flow)

sde_sampler.sample_loop(
    num_samples=128,
    num_steps=100,
    seed=0,
)

traj = sde_sampler.trajectories
img = traj[-1]
```

## Interactive Tutorials

1. **Introduction with 2D Toy**: This [notebook](examples/train_2d_toys.ipynb) provides a 2D toy example to illustrate the fundamental concepts of Rectified Flow. It covers the basics of interpolation processes, the training and inference of rectified flow, and reflow, which straightens flow dynamics to achieve speedup.
2. **Samplers**: This [notebook](examples/samplers_2d_toys.ipynb) explores the samplers available in this repository using a 2D toy example. It illustrates the concepts and usage of both deterministic and stochastic samplers. Additionally, it demonstrates how to customize a sampler by inheriting from the $\texttt{Sampler}$ base class and discusses the effects of using stochastic samplers.
3. **Interpolation**: This [notebook](examples/interpolation_conversion.ipynb) introduces the idea that different affine interpolations can be converted from one form to another and provides a simple implementation to achieve this transformation. It also highlights the interesting observation that the same transformation applies to rectified flows — and, in fact, to their discretized trajectories produced by natural Euler samplers.
4. **Flux**: We provide a [notebook](examples/inference_flux_dev.ipynb) that shows how to easily interact with the wrapped Flux model using different samplers. Additionally, another [notebook](examples/editing_flux_dev.ipynb) demonstrates how to perform image editing task with Flux. All implementations in a clear and accesible manner.

## Training Pipelines

We provide **[Diffusers](https://github.com/huggingface/diffusers)-style** training scripts for [UNet](https://github.com/NVlabs/edm) and [DiT](https://github.com/facebookresearch/DiT) in this [directory](rectified_flow/pipelines). The training scripts utilizes **[Accelerate](https://github.com/huggingface/accelerate)** for multi-GPU training.

**Results Using this Training Scripts**:

- **UNet CIFAR10**: Trained for $300 \text{k}$ iterations with `batch_size=256`. You can download the model [here](https://drive.google.com/file/d/16DKMPtR1-Dhy21F8WNzkDtXc7bcoCFjW/view?usp=drive_link). $\text{FID}_{50\text{K}}=2.496$.
- **DiT CIFAR10**: Trained for $1000 \text{k}$ iterations with `batch_size=128`. You can download the model [here](). $\text{FID}_{50\text{K}}=3.678$.

**Loading a Pretrained Model**:

To construct a model from a pretrained checkpoint, simply run the following code:

```python
from rectified_flow.models.dit import DiT

model = DiT.from_pretrained(save_directory="PATH_TO_MODEL", filename="dit", use_ema=True).to(device)
```

# Customize Your Component

## Interpolation

The `AffineInterp` class manages the affine interpolation between the source distribution $\pi_0$ and the target distribution $\pi_1$. It offers two primary features:

1. **Automatic Interpolation Handling**: Given an affine interpolation $X_t=\alpha_tX_1 + \beta_t X_0$, providing $\alpha_t$ and $\beta_t$ functions (optionally along with their time-derivative functions $\dot \alpha_t$ and $\dot \beta_t$), `AffineInterp` computes the interpolated state $X_t$ and its time derivative $\dot X_t$. If the derivatives functions $\dot \alpha_t, \dot\beta_t$ are not supplied, they are calculated automatically using Pytorch automatic differentiation.

   ```python
   from rectified_flow.flow_components import AffineInterp
   
   alpha_function = lambda t: torch.sin(a * t) / torch.sin(a)
   beta_function = lambda t: torch.sin(a * (1 - t)) / torch.sin(a)
   
   interp = AffineInterp(alpha=alpha_function, beta=beta_function)
   
   x_t, dot_x_t = interp.forward(x_0, x_1, t)
   ```

2. **Automatic Solving of Unknown Variables**: Given any two of the four variables ($X_0,X_1,X_t,\dot X_t$), the class can automatically solve for the remaining unknowns using precomputed symbolic solvers from $X_t = \alpha_t X_1 + \beta_t X_0$, and $\dot{X}_t = \boldsymbol{\dot{\alpha}}_t X_1 + \boldsymbol{\dot{\beta}}_t X_0$. 
This feature is  convenient to avoid the hand derivation of the coefficients in DDIM like algorithms, and conversion between important quantities, such as the RF velocity, score fuction, and predicte noise and targets.
   ```python
   # Solve for x_0 and x_1 given x_t and dot_x_t
   interp.solve(t=t, x_t=x_t, dot_x_t=velocity)
   print(interp.x_0, interp.x_1)
   ```
   
   ```python
     # The inference step of DDIM as curved Euler sampler walking along the interopoliation curves 
    def step(self):
        t, t_next, x_t = self.t, self.t_next, self.x_t
        v_t = self.rectified_flow.get_velocity(x_t, t)
   
         # find the expected noise x_0_pred and data x_1_pred from the interpolation
        interp = self.interp_inference.solve(t, x_t=x_t, dot_x_t=v_t)
        x_1_pred = interp.x_1
        x_0_pred = interp.x_0
           
        # Get x_{t_next} on the interpolated curve
        self.x_t = self.interp_inference.solve(t_next, x_0=x_0_pred, x_1=x_1_pred).x_t
   ```   

## Wrapping a New Velocity

The `velocity_field` argument in the `RectifiedFlow` class accepts a neural network or any callable function that takes $x_t$ and $t$ as inputs. If you need to customize your model or convert it from other formats, we recommend using a velocity field wrapper to simplify the process.

1. **Reversing the Time Direction**  

In Rectified Flow, we use the convention of transforming the **noise (or source) distribution** $X_0 \sim \pi_0$ at time $t = 0$ to the **data (target) distribution** $X_1 \sim \pi_1$ a time $t=1$. In scenarios like Flux, where the velocity transitions from $\pi_1$ to $\pi_0$ for image generation, we need to reverse the ODE time direction. This can be easily achieved by using a simple wrapper:


   ```python
   # Reverse ODE time direction
   velocity = lambda x_t, t: -model(x_t, 1.0 - t)
   ```

2. **Reparameterizing for Noise Prediction**  
   Some works parameterize the model to predict noise instead of velocity. Using an `AffineInterpSolver`, you can automatically convert noise predictions into velocity predictions, bypassing the complexity of handling DDIM coefficients.

   ```python
   # Convert noise prediction to velocity prediction
   # Assume model is trained by minimizing ((x0 - model(x_t, t))**2).mean(), with x_t = a_t * x_1 + b_t*x_0, where x_0 is noise, and x_1 is data.
   velocity = lambda x_t, t: self.interp.solve(t=t, x_t=x_t, x_0=model(x_t, t)).dot_x_t
   ```

   
## Training Utilities

To tailor the training process to your specific requirements, you can customize these utilities by inheriting from their base classes and overriding their methods. Once customized, simply pass the instances to the `RectifiedFlow` class during initialization.

**Example: Custom Weighting Scheme**
```python
from rectified_flow.flow_components import TrainTimeWeight

class CustomTimeWeight(TrainTimeWeight):
    def __init__(self):
        super().__init__()

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        wts = torch.exp(t)
        return wts

# Initialize with custom exponential weighting
custom_time_weight = CustomTimeWeight()
```

## Customized Samplers

To create custom samplers with specific integration schemes, subclass the `Sampler` class and implement the `step` method tailored to your needs. The `step` method receives the current state `x_t`, `t`, and `t_next` from the base class and defines the integration scheme.

**Example: Euler Sampler**
```python
from rectified_flow.flow_components import Sampler

class EulerSampler(Sampler):
    def __init__(
        self,
        rectified_flow: RectifiedFlow,
        num_steps: int | None = None,
        time_grid: list[float] | torch.Tensor | None = None,
        record_traj_period: int = 1,
        callbacks: list[Callable] | None = None,
        num_samples: int | None = None,
    ):
        super().__init__(
            rectified_flow,
            num_steps,
            time_grid,
            record_traj_period,
            callbacks,
            num_samples,
        )

    def step(self, **model_kwargs):
        t, t_next, x_t = self.t, self.t_next, self.x_t
        v_t = self.rectified_flow.get_velocity(x_t, t, **model_kwargs)
        self.x_t = x_t + (t_next - t) * v_t
```
After defining your custom sampler, you can perform sampling just like with a standard sampler.

The following is a quick implementation of stochastic sampler which covers the DDPM sampling algorithm: 

```python
class MyStochasticSampler(Sampler):
    def __init__(self, rectified_flow: RectifiedFlow, noise_replacement_rate: Callable | str = lambda t, t_next: 0.5, **kwargs):
        super().__init__(rectified_flow=rectified_flow, **kwargs)

        if not (self.rectified_flow.independent_coupling and self.rectified_flow.is_pi_0_zero_mean_gaussian):
            import warnings
            warnings.warn("It is only theoretically correct to use this sampler when pi0 is a zero mean Gaussian and the coupling (X0, X1) is independent. Proceed at your own risk.")

        self.noise_replacement_rate = noise_replacement_rate

    def step(self, **model_kwargs):
        """Perform a single step of the sampling process."""
        t, t_next, x_t = self.t, self.t_next, self.x_t
        v_t = self.rectified_flow.get_velocity(x_t, t, **model_kwargs)

        # Given x_t and dot_x_t = vt, find the corresponding endpoints x_0 and x_1
        interp = self.rectified_flow.interp.solve(t, x_t=x_t, dot_x_t=v_t)
        x_1_pred = interp.x_1
        x_0_pred = interp.x_0

        # Randomize x_0_pred by replacing part of it with new noise
        noise = self.rectified_flow.sample_source_distribution(self.num_samples)
        noise_replacement_factor = self.noise_replacement_rate(t, t_next)
        x_0_pred_refreshed = (
            (1 - noise_replacement_factor) * x_0_pred + 
            (1 - (1 - noise_replacement_factor) ** 2) ** 0.5 * noise
        )

        # Interpolate to find x_t at t_next
        self.x_t = self.rectified_flow.interp.solve(t_next, x_0=x_0_pred_refreshed, x_1=x_1_pred).x_t
  ```

# Citation
If you find this repository useful for your research, please consider citing

```bibtex
@misc{lq2024rectifiedflow,
  author = {Qiang Liu, Runlong Liao, Bo Liu, Xixi Hu},
  title = {PyTorch RectifiedFlow},
  year = {2024},
  url = {https://github.com/lqiang67/rectified-flow}
}
```

# License

| Component | License                                                      |
| --------- | ------------------------------------------------------------ |
| Codebase  | [MIT License](LICENSE)                                       |

