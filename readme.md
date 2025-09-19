## Overview

UNav Server provides a serverless implementation for indoor navigation using computer vision techniques. It leverages Modal for deployment and offers features like visual localization, path planning, and navigation guidance.

## Prerequisites

- Python 3.10+
- Modal CLI
- A Modal account and token


## Deployment

Navigate to the folder

``` 
cd src/modal_functions/unav_v2
```

```
pip install local_deployment_requirements.txt
```


```
modal deploy unav_modal.py
```



## Test the deployed unav code

Make sure you are inside the `src` folder

``` 
python modal_functions/test_modal_functions.py
```
