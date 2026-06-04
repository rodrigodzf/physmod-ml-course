# Project Ideas

Each idea describes work that could be developed into a final project.

## Learning and Manipulating Modal Couplings

A possible project is to learn or design the coupling structure between resonant modes in a differentiable modal synthesiser. Building on the Week 3 idea that linear modes are independent until nonlinear forces make them interact, you could replace a fixed tension-modulation term with a learnable coupling map, inspired by Poirot et al.'s coupled resonant filter bank for nonlinear sound sources. The project would investigate whether these couplings can be inferred from audio while remaining physically interpretable, and whether they can be manipulated creatively to control energy transfer, pitch glide, roughness, decay, or impact-like nonlinear behaviour.

## Learning Basis Functions for Membranes

A possible project is to extend the Week 2 modal identification example, where a neural network jointly predicts spatial basis functions, modal frequencies and decays, from a 1D string to 2D resonators. Since the natural basis for real instruments or rooms is not always known in advance, you could first generate synthetic datasets for rectangular and circular membranes, then train a model to recover the spatial modes, frequencies and damping rates directly from simulated displacement data. The project would compare how well learned basis functions match the analytical membrane modes, and whether the learned representation remains useful when the geometry, boundary conditions or observation locations become less ideal.

## Comparing Losses for Modal and Physical Parameter Fitting

A possible project is to turn the Week 4 discussion of audio losses into a systematic benchmark for differentiable physical modelling. You could generate clean synthetic targets from string and membrane models, then fit either free modal parameters or lower-dimensional physical parameters using several losses: waveform MSE or L1, magnitude and log-magnitude STFT losses, multi-resolution STFT losses, spectral optimal transport, scattering-style losses, and possibly task-specific envelope or modal-domain losses. The project would evaluate how each objective performs when estimating modal frequencies, decay rates and gains, and then ask whether the same loss remains useful when fitting physical parameters such as string tension, stiffness and damping, or membrane wave speed and damping. Useful comparisons would include parameter error, audio reconstruction error, convergence reliability from different initialisations, optimisation time, memory use, and gradient stability. The goal would be to identify which losses are accurate, efficient and robust for modal parameter recovery, and where the loss that sounds best is not the one that recovers the underlying physics most reliably.
