## Overview

UNav Server provides a serverless implementation for indoor navigation using computer vision techniques. It leverages Modal for deployment and offers features like visual localization, path planning, and navigation guidance.

## Prerequisites

- Python 3.10+
- Modal CLI
- A Modal account and token

## Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment:

- On macOS/Linux: `source .venv/bin/activate`
- On Windows: `.venv\Scripts\activate`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Deployment

Navigate to the folder

```bash
cd src/modal_functions/unav_v2
```

```bash
modal deploy unav_modal.py
```

`scaledown_window` and GPU type are configurable at deploy time using:

- `UNAV_SCALEDOWN_WINDOW` (default: `300`)
- `UNAV_GPU_TYPE` (default: `t4`, allowed: `a10`, `t4`, `any`)

```bash
UNAV_SCALEDOWN_WINDOW=600 UNAV_GPU_TYPE=t4 modal deploy unav_modal.py
```

You can also deploy from GitHub UI using:

- Actions -> `Deploy UNav v2 Modal` -> `Run workflow`
- Set `scaledown_window` input
- Set `gpu_type` input (`a10`, `t4`, or `any`)
- Ensure repository secrets `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` are configured

## To test the deployed unav code with a predefined function parameters

```bash
python test_modal_functions.py
```
