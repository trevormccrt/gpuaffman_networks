# gpuaffman_networks
**GPU accelerated Kauffman networks**

A lightweight library for simulating NK Boolean networks (Kauffman networks). All functions in the library are compatible with both numpy and cupy arrays,
so GPU acceleration of network evaluation is trivial. 

In this library, Kauffman networks are represented by tensors that specify the truth tables and connectivity used for state updates. We support vectorized simulation
of arbitrary batches of networks on both CPUs and GPUs.

# Installation

```console
pip install .
```

# Getting started

See the [example notebook](https://github.com/trevormccrt/gpuaffman_networks/blob/master/examples/ragged_network_properties.ipynb) that shows how to specify and evaluate networks using numpy or cupy
