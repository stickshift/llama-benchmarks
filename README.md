# llama-benchmarks

Utilities to help you experiment with and benchmark Meta's Llama models.

## Prerequisites

* Python 3.12
* uv (pip install uv)
* Llama models (Download from Meta https://www.llama.com/)

## Build and Test Workflow

### Configure Environment

```shell
source environment.sh
```

### Build Artifacts

Builds local Python venv, installs dependencies, and downloads MMLU dataset needed for tests.

```shell
make
```

### Run Tests

Runs test automation.

```shell
make tests
```

## Experiments

* [Exploring Order Dependency](experiments/20241107-exploring-order-dependency.ipynb)