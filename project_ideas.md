# Project Ideas

## Learning and Manipulating Modal Couplings

A possible project is to learn or design the coupling structure between resonant modes in a differentiable modal synthesiser. Building on the Week 3 idea that linear modes are independent until nonlinear forces make them interact, the student could replace a fixed tension-modulation term with a learnable coupling map, inspired by Poirot et al.'s coupled resonant filter bank for nonlinear sound sources. The project would investigate whether these couplings can be inferred from audio while remaining physically interpretable, and whether they can be manipulated creatively to control energy transfer, pitch glide, roughness, decay, or impact-like nonlinear behaviour.

## Learning Basis Functions for Membranes

A possible project is to extend the Week 2 modal identification example, where a neural network jointly predicts spatial basis functions, modal frequencies and decays, from a 1D string to 2D resonators. Since the natural basis for real instruments or rooms is not always known in advance, the student could first generate synthetic datasets for rectangular and circular membranes, then train a model to recover the spatial modes, frequencies and damping rates directly from simulated displacement data. The project would compare how well learned basis functions match the analytical membrane modes, and whether the learned representation remains useful when the geometry, boundary conditions or observation locations become less ideal.
