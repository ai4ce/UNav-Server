# UNav Server Communication Guide

## Overview

This document provides detailed instructions on how to communicate with the UNav server. The server offers various endpoints to manage settings, start/terminate the server, localize images, get floorplans, select destinations, and navigate paths.

## Base URL

The base URL for all API requests is:

http://128.122.136.173:5001

## Endpoints

### 1. Update Settings

**Endpoint:** `/settings`

**Method:** `POST`

**Description:** Setup desired navigation scene.

**Request Format:**

```json
{
    "place": "New_York_City",
    "building": "LightHouse",
    "floor": "6th_floor",
    "scale": 0.01098358101
}
```

### 2. Start Server

**Endpoint:** `/start`

**Method:** `POST`

**Description:** Start the server and initialize required components.

**Request Format:** None

**Response Format:**

```json
{
    "status": "server started"
}
```

### 3. Terminate Server

**Endpoint:** `/terminate`

**Method:** `POST`

**Description:** Terminate the server and clean up resources.

**Request Format:** None

**Response Format:**

```json
{
    "status": "server terminated"
}
```

### 4. Localize Image

**Endpoint:** `/localize`

**Method:** `POST`

**Description:** Localize the user’s position based on a query image.

**Request Format:**

```json
{
    "query_image": "data:image/png;base64,<base64_encoded_image>"
}
```

**Response Format:**


* ***Success:***
    ```json
    {
        "pose": [x, y, angle]
    }
    ```

* ***Failure:***
    ```json
    {
        "pose": null
    }
    ```

### 5. Get Floorplan and Destinations

**Endpoint:** `/get_floorplan_and_destinations`

**Method:** `GET`

**Description:** Retrieve the floorplan image and destination points.

**Response Format:**

```json
{
    "floorplan": "<base64_encoded_image>",
    "destinations": [
        {
            "name": "ADA Restroom",
            "id": "07993",
            "location": [x, y]
        },
        ...
    ],
    "anchors": [
        [x, y],
        ...
    ]
}
```

### 6. Select Destination

**Endpoint:** `/select_destination`

**Method:** `POST`

**Description:** Select a destination for navigation.

**Request Format:**

```json
{
    "destination_id": "07993"
}
```

**Response Format:**

```json
{
    "status": "success"
}
```

### 7. Navigate

**Endpoint:** `/planner`

**Method:** `GET`

**Description:** Get the navigation path from the current position to the selected destination.

**Response Format:**

* ***Success:***
    ```json
    {
        "paths": [
            [x, y],
            [x, y, "floor_name"],
            ...
        ],
        "floorplan": "<base64_encoded_image>",
        "actions": [rotation, distance, rotation, distance, ...]
    }
    ```

* ***Failure:***
    ```json
    {
        "error": "Pose or selected destination ID is not set"
    }
    ```

### 8. List Images

**Endpoint:** `/list_images`

**Method:** `GET`

**Description:** List available images for localization.

**Response Format:**

```json
{
    "id1": ["image1.png", "image2.png"],
    "id2": ["image1.png", "image2.png"],
    ...
}
```

### 9. Get Image

**Endpoint:** `/get_image/<id>/<image_name>`

**Method:** `GET`

**Description:** Retrieve a specific image by ID and image name.

**Response Format:**

```json
{
    "image": "<base64_encoded_image>"
}
```

## Example Workflow

### Step1: Update Settings
Send a **'POST'** request to **`/settings`** with the required settings.

### Step 2: Start Server

Send a **'POST'** request to **`/start`** to initialize the server.

### Step 3: Browse and Select Image

Send a **'GET'** request to **`/list_images`** to get available images and then a **'GET'** request to **`/get_image/{id}/{image_name}`** to retrieve the selected image.

**Note:** This step is optional, just for testing your system in case you don't have a query image. You can use the image you captured.

### Step 4: Localize Position

Send a **'POST'** request to **`/localize`** with the selected query image in base64 format.

### Step 5: Select Destination

Send a **'GET'** request to **`/get_floorplan_and_destinations`** to get the floorplan and destinations. Then send a **'POST'** request to **`/select_destination`** with the selected destination ID.

### Step 6: Navigate

Send a **'GET'** request to **`/planner`** to get the navigation path.

### Step 7: Terminate Server

Send a **'POST'** request to **`/terminate`** to shut down the server.

## Modal Volume set up 
- make the following changes in the file src/modal_functions/volume/modalvolumedata_setup.py 
- update the items_to_rearrange = [] in the fucntion : rearrange_files_and_folders(), if rearrangment is needed else please pass the empty list ,
- add s3 bucket credentials to your .env file refer .env.example file 
- execute the file 
```
python src/modal_functions/volume/modalvolumedata_setup.py
```