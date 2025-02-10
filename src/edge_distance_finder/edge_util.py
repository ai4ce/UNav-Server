from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import requests
import os
import subprocess
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
from mlsd.utils import pred_lines, pred_squares
from urllib.request import urlretrieve
import torch
import matplotlib.pyplot as plt
import math
from sklearn.cluster import DBSCAN

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
prompts = ["the floor"]
# Load MLSD 512 Large FP32 tflite
model_name = 'mlsd/tflite_models/M-LSD_512_large_fp32.tflite'
interpreter = tf.lite.Interpreter(model_path=model_name)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to calculate the slope of a line
def calculate_slope(line):
    x1, y1, x2, y2 = line
    if x2 - x1 == 0:
        return float('inf')  # Handle vertical line (infinite slope)
    return (y2 - y1) / (x2 - x1)

# Function to calculate the distance between a line and a point (vanishing point)
def distance_to_point(line, point):
    x1, y1, x2, y2 = line
    px, py = point
    # Line equation coefficients: Ax + By + C = 0
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    # Distance from point to line formula
    distance = abs(A * px + B * py + C) / math.sqrt(A**2 + B**2)
    return distance

# Function to count how many edge pixels are close to the line
def count_nearby_edge_pixels(line, edges, threshold=10):
    line_image = np.zeros_like(edges, dtype=np.uint8)
    cv2.line(line_image, (line[0], line[1]), (line[2], line[3]), 255, 1)
    dilated_line_image = cv2.dilate(line_image, np.ones((threshold * 2 + 1, threshold * 2 + 1), np.uint8))
    overlap = np.logical_and(dilated_line_image > 0, edges > 0)
    return np.sum(overlap)

# Function to calculate the line length
def line_length(line):
    x1, y1, x2, y2 = line[0]
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Function to compute the angle of a line
def compute_line_angle(line):
    x1, y1, x2, y2 = line[0]
    angle_rad = np.arctan2(y2 - y1, x2 - x1)
    angle_deg = np.degrees(angle_rad) % 180  # Normalize angle to [0°, 180°)
    return angle_deg

# Function to check if a line is almost horizontal
def is_almost_horizontal(line, threshold_angle):
    angle = compute_line_angle(line)
    return angle <= threshold_angle or angle >= (180 - threshold_angle)

# Function to extend a line segment
def extend_line(line, length=1500):
    x1, y1, x2, y2 = line[0]
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return line  # Can't extend a point
    norm = np.sqrt(dx ** 2 + dy ** 2)
    scale = length / norm
    x1_ext = int(x1 - dx * scale)
    y1_ext = int(y1 - dy * scale)
    x2_ext = int(x2 + dx * scale)
    y2_ext = int(y2 + dy * scale)
    return np.array([[x1_ext, y1_ext, x2_ext, y2_ext]])

# Function to find the intersection point between two lines
def line_intersection(line1, line2, img_shape, tolerance=1e-5):
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # Check if lines are parallel or nearly parallel
    if abs(denominator) < tolerance:
        return None  # Lines are parallel or overlapping

    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denominator
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denominator

    # Check if the intersection point is within the image frame
    height, width = img_shape
    if 0 <= px <= width and 0 <= py <= height:
        return (int(px), int(py))
    else:
        return None

def find_vanishing_point(image, score_thr=0.01, dist_thr=2.0):
    line_candidate = []
    # Ensure the image is in color
    if len(image.shape) == 2:
        # Image is grayscale
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        # Image has one channel but in 3D
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_color = image.copy()

    # Convert to grayscale for processing
    gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    drawn_image = image_color.copy()
    selected_lines_image = np.zeros_like(image_color)  # Create an empty canvas for the selected lines

    # Get the height and width of the image
    frame_height, frame_width = gray.shape

    # Lines detection using your custom function
    lines = pred_lines(image_color, interpreter, input_details, output_details, input_shape=[512, 512], score_thr=score_thr, dist_thr=dist_thr)

    # Set minimum length threshold for filtering
    min_length = 8  # Adjust based on your requirements

    # Threshold angles to try
    threshold_angles = [30, 25, 20, 15, 10, 5, 2]

    centroid = None  # Initialize centroid to None

    for threshold_angle in threshold_angles:
        # Reset filtered and extended lines for each threshold
        filtered_lines = []
        extended_lines = []

        if lines is not None:
            for line in lines:
                line = [line]
                if (
                    line_length(line) > min_length
                    and not is_almost_horizontal(line, threshold_angle=threshold_angle)
                ):
                    filtered_lines.append(line)
                    extended_lines.append(extend_line(line))

        # Find intersection points between pairs of extended lines
        intersection_points = []
        for i in range(len(extended_lines)):
            for j in range(i + 1, len(extended_lines)):
                intersection = line_intersection(extended_lines[i], extended_lines[j], img_shape=gray.shape)
                if intersection is not None:
                    intersection_points.append(intersection)

        # Proceed only if there are intersection points within the image
        if intersection_points:
            intersection_points = np.array(intersection_points)
            # Cluster intersection points to find the most common convergence point
            dbscan = DBSCAN(eps=20, min_samples=3).fit(intersection_points)
            labels = dbscan.labels_

            # Find the largest cluster whose centroid is within the image frame
            unique_labels = set(labels)
            max_points = 0
            for label in unique_labels:
                if label == -1:
                    continue  # Ignore noise points
                class_member_mask = (labels == label)
                cluster_points = intersection_points[class_member_mask]
                # Calculate centroid
                centroid_candidate = np.mean(cluster_points, axis=0)
                # Check if centroid is within the image frame
                if (0 <= centroid_candidate[0] <= frame_width) and (0 <= centroid_candidate[1] <= frame_height):
                    num_points = cluster_points.shape[0]
                    if num_points > max_points:
                        max_points = num_points
                        centroid = centroid_candidate.astype(int)
                        best_cluster_points = cluster_points

            if centroid is not None:
                # Plot the centroid on the image
                cv2.circle(drawn_image, tuple(centroid), 10, (255, 255, 0), -1)

                # Draw the extended lines
                for line in extended_lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(drawn_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.line(selected_lines_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 1)
                    line_candidate.append(line[0])

                # Vanishing point found within image frame, exit the loop
                print(f"Vanishing point found with threshold angle: {threshold_angle}")
                break
        else:
            # If no intersection points found, continue to the next threshold
            continue

    # If no centroid found after all thresholds, optionally process the last set of lines for visualization
    if centroid is None:
        # Draw the extended lines for the last threshold tried
        for line in extended_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(drawn_image, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # Return the selected_lines_image, the vanishing point coordinates, and line candidates
    return selected_lines_image, centroid, line_candidate

def find_curb_line(image, reference_slope, plot=False):
    """return the curb line

    Args:
        image (Numpy Arry): image in numpy arrary form
        reference_slope (_type_): _description_
        plot (bool, optional): _description_. Defaults to False.
    """
    # Process inputs
    inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
    preds = outputs.logits.unsqueeze(1)

    # Convert to binary tensor
    binary_tensor = (preds[0][0] > 0).int().numpy()

    # Get image dimensions
    height, width = image.shape[:2]

    # Rescale the binary tensor to the original image size
    resized_binary_tensor = cv2.resize(binary_tensor, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

    # Create a blank mask
    new_mask = np.zeros_like(resized_binary_tensor)

    # Find contours and fill them
    contours, _ = cv2.findContours(resized_binary_tensor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    epsilon_factor = 0.001
    for contour in contours:
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)
        cv2.fillPoly(new_mask, [approx_contour], 1)
    resized_binary_tensor = new_mask

    # Call the function to find the vanishing point
    selected_lines_image, vanishing_point, line_candidate = find_vanishing_point(image)
    print("Vanishing Point:", vanishing_point)
    print("Line Candidate:", line_candidate)

    # Create final mask
    final_mask = (resized_binary_tensor * 255).astype(np.uint8)

    # Detect edges and dilate
    edges = cv2.Canny(final_mask, 100, 200)
    buffer_size = 15
    dilated_edges = cv2.dilate(edges, np.ones((2 * buffer_size + 1, 2 * buffer_size + 1), np.uint8))

    # Create an inner mask by shrinking the original mask by 20%
    scale_factor = 0.4
    inner_mask = cv2.resize(final_mask, (0, 0), fx=scale_factor, fy=scale_factor)
    inner_mask_resized = np.zeros_like(final_mask)
    inner_height, inner_width = inner_mask.shape[:2]
    inner_mask_resized[
        (height - inner_height) // 2:(height + inner_height) // 2,
        (width - inner_width) // 2:(width + inner_width) // 2
    ] = inner_mask

    # Updated function to check if a line is near edges but does not pass through the inner part
    def is_line_near_edges_and_outside_inner(line, edges_mask, inner_mask):
        line_image = np.zeros_like(edges_mask, dtype=np.uint8)
        cv2.line(line_image, (line[0], line[1]), (line[2], line[3]), 255, 1)
        near_edges = np.any(np.logical_and(line_image > 0, edges_mask > 0))
        outside_inner = not np.any(np.logical_and(line_image > 0, inner_mask > 0))
        return near_edges and outside_inner

    # Filter lines near edges and outside the inner part
    filtered_lines = [line for line in line_candidate if is_line_near_edges_and_outside_inner(line, dilated_edges, inner_mask_resized)]

    # Find the best line
    selected_line = None
    min_score = float('inf')
    for line in filtered_lines:
        slope = calculate_slope(line)
        if (reference_slope < 0 and slope < 0) or (reference_slope > 0 and slope > 0):
            slope_difference = abs(slope - reference_slope)
        else:
            continue
        distance_from_vanishing_point = distance_to_point(line, vanishing_point)
        edge_pixel_count = count_nearby_edge_pixels(line, edges)
        score = 100 * slope_difference + distance_from_vanishing_point - 0.5 * edge_pixel_count

        if is_line_near_edges_and_outside_inner(line, dilated_edges, inner_mask_resized) and score < min_score:
            min_score = score
            selected_line = line

    # Plot and display if required
    if plot:
        # Show the intermediate steps and results
        # cv2_imshow(resized_binary_tensor)

        plt.imshow(binary_tensor, cmap='gray', aspect='auto')
        plt.title('Binary Tensor Heatmap')
        plt.show()

        plt.figure(figsize=(10, 10))
        plt.imshow(resized_binary_tensor, cmap='gray')
        plt.title('Binary Mask with Contour Area Filled')
        plt.axis('off')
        plt.show()

        cv2.circle(selected_lines_image, tuple(vanishing_point), 10, (255, 255, 0), -1)
        # cv2_imshow(selected_lines_image)

        # Visualization with initial lines
        visualization_image = np.zeros((height, width, 3), dtype=np.uint8)
        visualization_image[:, :, 0] = final_mask
        for line in line_candidate:
            cv2.line(visualization_image, (line[0], line[1]), (line[2], line[3]), (255, 255, 0), 2)
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(visualization_image, cv2.COLOR_BGR2RGB))
        plt.scatter(vanishing_point[0], vanishing_point[1], color='yellow', label='Vanishing Point', s=100)
        plt.title('Lines Close to Reference Slope and Floor Mask')
        plt.legend()
        plt.show()

        # Visualization with filtered lines
        visualization_image_filtered = visualization_image.copy()
        visualization_image_filtered[:, :, :] = 0
        visualization_image_filtered[:, :, 0] = final_mask
        for line in filtered_lines:
            cv2.line(visualization_image_filtered, (line[0], line[1]), (line[2], line[3]), (255, 255, 0), 2)
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(visualization_image_filtered, cv2.COLOR_BGR2RGB))
        plt.scatter(vanishing_point[0], vanishing_point[1], color='yellow', label='Vanishing Point', s=100)
        plt.title('Lines Close to Edges of the Mask')
        plt.legend()
        plt.show()

        # Visualization with selected line
        if selected_line is not None:
            visualization_image_selected = visualization_image.copy()
            cv2.line(visualization_image_selected, (selected_line[0], selected_line[1]), (selected_line[2], selected_line[3]), (155, 255, 225), 2)
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(visualization_image_selected, cv2.COLOR_BGR2RGB))
            plt.scatter(vanishing_point[0], vanishing_point[1], color='yellow', label='Vanishing Point', s=100)
            plt.title('Selected Line')
            plt.legend()
            plt.show()

    # Return the slope and intercept
    if selected_line is not None:
        x1, y1, x2, y2 = selected_line
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        return slope, intercept
    else:
        print("No line selected, cannot draw on the frame.")
        return None, None
