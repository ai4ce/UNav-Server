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

## To test the deployed unav code with a predefined function parameters

```bash
python test_modal_functions.py
```
