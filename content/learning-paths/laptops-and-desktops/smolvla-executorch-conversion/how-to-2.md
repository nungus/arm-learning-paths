---
title: Set up the environment
weight: 3

### FIXED, DO NOT MODIFY
layout: learningpathall
---

## Download the files

Create a working directory and download the project files for this Learning
Path:

```bash
mkdir smolvla-executorch-work
cd smolvla-executorch-work
curl -fsSLO https://raw.githubusercontent.com/ArmDeveloperEcosystem/arm-learning-paths/main/content/learning-paths/laptops-and-desktops/smolvla-executorch-conversion/download-project.sh
bash download-project.sh
cd smolvla-executorch-arm
```

The download script recreates the project directory structure and restores the
executable permissions on the shell and Python scripts.


## Set up the software environment
The `setup.sh` script:
- Pins ExecuTorch v1.4.1 at commit `e4d02f41f7909e8ed5bf4a14ffc520d733453d9f`.
- Builds the ExecuTorch and XNNPACK + KleidiAI runtime libraries and Python bindings.
- Installs relevant Python packages.
- Downloads a fixed SmolVLA checkpoint from Hugging Face.
- Creates a project-local virtual environment `.venv`.

```bash
./scripts/setup.sh
```
Activate the virtual environment and resolve repository-local path variables with:
```bash
source env.sh
```

## Verify the resources
TODO


## What you've accomplished and what's next
You have obtained the scripts to convert the model, and configured your environment for building the runtime.

Next, you'll work through the ExecuTorch pipeline to export and lower the FP32 SmolVLA, build the XNNPACK-backed ExecuTorch runner, and validate the converted model's output.
