<img src="https://github.com/TeamGraphix/graphix/raw/master/docs/logo/black_with_name.png" alt="logo" width="550">

![PyPI](https://img.shields.io/pypi/v/graphix)
![License](https://img.shields.io/github/license/TeamGraphix/graphix)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/graphix)
[![Downloads](https://static.pepy.tech/badge/graphix)](https://pepy.tech/project/graphix)
[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg)](https://unitary.fund/)
[![DOI](https://zenodo.org/badge/573466585.svg)](https://zenodo.org/badge/latestdoi/573466585)
[![CI](https://github.com/TeamGraphix/graphix/actions/workflows/ci.yml/badge.svg)](https://github.com/TeamGraphix/graphix/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/TeamGraphix/graphix/graph/badge.svg?token=E41MLUTYXU)](https://codecov.io/gh/TeamGraphix/graphix)
[![Documentation Status](https://readthedocs.org/projects/graphix/badge/?version=latest)](https://graphix.readthedocs.io/en/latest/?badge=latest)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Graphix** is a measurement-based quantum computing (MBQC) software package, featuring

- the measurement calculus framework with integrated graphical rewrite rules for Pauli measurement preprocessing
- circuit-to-pattern transpiler, graph-based deterministic pattern generator and manual pattern generation
- flow, gflow and pauliflow finding tools and graph visualization based on flows (see below)
- statevector, density matrix and tensornetwork pattern simulation backends
- QPU interface and fusion network extraction tool

## Installation

Install `graphix` with `pip`:

```bash
pip install graphix
```

Install together with device interface:

```bash
pip install graphix[extra]
```

this will install `graphix` and interface for [IBMQ](https://github.com/TeamGraphix/graphix-ibmq) and [Perceval](https://github.com/TeamGraphix/graphix-perceval) to run MBQC patterns on superconducting and optical QPUs and their simulators.

## Using graphix

### generating pattern from a circuit

```python
from graphix import Circuit

circuit = Circuit(4)
circuit.h(0)
...
pattern = circuit.transpile().pattern
pattern.draw_graph()
```

<img src="https://github.com/TeamGraphix/graphix/assets/33350509/de17c663-f607-44e2-945b-835f4082a940" alt="graph_flow" width="750">

<small>note: this graph is generated from QAOA circuit, see [our example code](examples/qaoa.py). Arrows indicate the [_causal flow_](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.74.052310) of MBQC and dashed lines are the other edges of the graph. the vertical dashed partitions and the labels 'l:n' below indicate the execution _layers_ or the order in the graph (measurements should happen from left to right, and nodes in the same layer can be measured simultaneously), based on the partial order associated with the (maximally-delayed) flow. </small>

### preprocessing Pauli measurements

```python
pattern.perform_pauli_measurements()
pattern.draw_graph()
```

<img src="https://github.com/TeamGraphix/graphix/assets/33350509/3c30a4c9-f912-4a36-925f-2ff446a07c68" alt="graph_gflow" width="140">

<small>(here, the graph is visualized based on [_generalized flow_](https://iopscience.iop.org/article/10.1088/1367-2630/9/8/250).)</small>

### simulating the pattern

```python
state_out = pattern.simulate_pattern(backend="statevector")
```

### and more..

- See [demos](https://graphix.readthedocs.io/en/latest/gallery/index.html) showing other features of `graphix`.

- Read the [tutorial](https://graphix.readthedocs.io/en/latest/tutorial.html) for more usage guides.

- For theoretical background, read our quick introduction into [MBQC](https://graphix.readthedocs.io/en/latest/intro.html) and [LC-MBQC](https://graphix.readthedocs.io/en/latest/lc-mbqc.html).

- Full API docs is [here](https://graphix.readthedocs.io/en/latest/references.html).

## Citing

> Shinichi Sunami and Masato Fukushima, Graphix. (2023) <https://doi.org/10.5281/zenodo.7861382>

## Contributing

We use [GitHub issues](https://github.com/TeamGraphix/graphix/issues) for tracking feature requests and bug reports.

## Discord Server

Please visit [Unitary Fund's Discord server](https://discord.com/servers/unitary-fund-764231928676089909), where you can find a channel for `graphix` to ask questions.

## Core Contributors (alphabetical order)

- Masato Fukushima (University of Tokyo, Fixstars Amplify)
- Maxime Garnier (Inria Paris)
- Thierry Martinez (Inria Paris)
- Sora Shiratani (University of Tokyo, Fixstars Amplify)
- Shinichi Sunami (University of Oxford)

## Acknowledgements

We are proud to be supported by [unitary fund microgrant program](https://unitary.fund/grants.html).

<p><a href="https://unitary.fund/grants.html">
<img src="https://user-images.githubusercontent.com/33350509/233384863-654485cf-b7d0-449e-8868-265c6fea2ced.png" alt="unitary-fund" width="150"/>
</a></p>

Special thanks to Fixstars Amplify:

<p><a href="https://amplify.fixstars.com/en/">
<img src="https://github.com/TeamGraphix/graphix/raw/master/docs/imgs/fam_logo.png" alt="amplify" width="200"/>
</a></p>

## License

[Apache License 2.0](LICENSE)
