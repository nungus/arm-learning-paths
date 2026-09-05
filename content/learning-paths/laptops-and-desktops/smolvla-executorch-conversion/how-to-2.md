---
title: Set up the environment
weight: 3

### FIXED, DO NOT MODIFY
layout: learningpathall
---

## Download the files
TODO (add files to the learning path files, commit them, and then try setting up the download from there)


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
