# gpuaffman_networks
**GPU accelerated Kauffman networks**

A lightweight library for simulating NK Boolean networks (Kauffman networks). All functions in the library are compatible with both numpy and cupy arrays,
so GPU acceleration of network evaluation is trivial. 

In this library, Kauffman networks are represented by tensors that specify the truth tables and connectivity used for state updates. It supports vectorized simulation
of arbitrary batches of networks on both CPUs and GPUs.

# Installation

Install using the provided setup.py (instructions [here](https://www.activestate.com/resources/quick-reads/how-to-manually-install-python-packages/)). In summary, in the directory ```gpuaffman_networks/```, run:

```console
pip install .
```

# Getting started

See the [example notebook](https://github.com/trevormccrt/gpuaffman_networks/blob/master/examples/ragged_network_properties.ipynb) that shows how to specify and evaluate networks using numpy or cupy

# Citing

If you find my package useful, please cite me!

```console
 @misc{McCourt2023,
  author = {McCourt, T},
  title = {gpuaffman_networks: GPU accelerated Kauffman networks},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/trevormccrt/gpuaffman_networks}}
}
```

