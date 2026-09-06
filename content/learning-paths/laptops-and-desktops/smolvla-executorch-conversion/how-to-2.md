---
title: Set up the environment
weight: 3

### FIXED, DO NOT MODIFY
layout: learningpathall
---

## Download the project files

Create a working directory and download the project files for this Learning
Path:

```bash
mkdir smolvla-executorch-work
cd smolvla-executorch-work
curl -fsSLO https://raw.githubusercontent.com/ArmDeveloperEcosystem/arm-learning-paths/main/content/learning-paths/laptops-and-desktops/smolvla-executorch-conversion/download-project.sh
bash download-project.sh
cd smolvla-executorch-arm
```

The `download-project.sh` script fetches the project directory and configures its scripts to be executable.


## Set up the software environment
The `setup.sh` script:
- Pins ExecuTorch v1.4.1 at commit `e4d02f41f7909e8ed5bf4a14ffc520d733453d9f`.
- Builds the ExecuTorch and XNNPACK + KleidiAI runtime libraries and Python bindings.
- Installs relevant Python packages.
- Downloads the pinned SmolVLA checkpoint from Hugging Face.
- Creates a project-local virtual environment `.venv`.

```bash
./scripts/setup.sh
```
Activate the virtual environment and resolve repository-local path variables with:
```bash
source env.sh
```

## Verify the resources

Verify the pinned packages, ExecuTorch revision and Python binding, XNNPACK and KleidiAI runtime build, and SmolVLA checkpoint files:

```bash
python scripts/check_environment.py
```

A successful check looks like:

```output
Environment OK: aarch64, ExecuTorch e4d02f41
  Python: /path/to/smolvla-executorch-arm/.venv/bin/python
  ExecuTorch: /path/to/smolvla-executorch-arm/toolchain/executorch
  Runtime: /path/to/smolvla-executorch-arm/toolchain/executorch/cmake-out-xnnpack
  Checkpoint: /path/to/smolvla-executorch-arm/checkpoints/smolvla_base
```


## What you've accomplished and what's next
You have obtained the scripts to convert the model, built the ExecuTorch runtime and configured your environment.

Next, you'll work through the ExecuTorch pipeline to export and lower the FP32 SmolVLA for Arm CPU, and validate the converted model against the PyTorch reference.
