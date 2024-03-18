from pathlib import Path

import cv2
import numpy as np
from util import saw_wave, triangle_wave
from video import Video, make_effect_decorator

datapath = Path(__file__).parent.parent / "data"

# My Example
# ----------
# Frame width: 1920
# Frame height: 1080
# Frame count: 1967
# Frame rate: 59.72692215560781
# ----------

second = 59.73

video = Video(int(1920 / 2), int(1080 / 2), second)

effect = make_effect_decorator(video)


# video.append_clip(datapath / "example1.mp4")
# video.append_clip(datapath / "example2.mp4")
video.append_clip(datapath / "example3.mp4")

##########################################################################################
##########################################################################################
# 1: (0 - 20s) Basic image processing

##########################################################################################
# 1.1: (4s) Switch between grayscale and color


@effect(0, 4)
def effect_grayscale(frame: np.ndarray, now: float, frame_idx: int) -> np.ndarray:
    enabled = int(frame_idx / 10) % 2 == 0
    new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if enabled else frame
    return (
        new_frame,
        f"""Grayscale
        Frame: {frame_idx}
        Enabled: {enabled}
        """,
    )


##########################################################################################
# 1.2: (8s) Blur the video


# Gaussian blur
@effect(4, 8)
def effect_gaussian_blur(frame: np.ndarray, now: float, frame_idx: int) -> np.ndarray:
    kernel_size = 51
    sigma = 0
    if now < 6:
        kernel_size = 2 * int(triangle_wave(now, 2) * 50) + 1
    else:
        sigma = triangle_wave(now, 2) * 20
    return (
        cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma),
        f"""Gaussian blur
        Frame: {frame_idx}
        Kernel size: {kernel_size}x{kernel_size}
        Sigma: {sigma:.2f}
        """,
    )


# Bilateral filter
@effect(8, 12)
def effect_bilateral_filter(
    frame: np.ndarray, now: float, frame_idx: int
) -> np.ndarray:
    d = 15
    sigma_color = 75
    sigma_space = 75
    if now < 10:
        d = int(triangle_wave(now, 2) * 29) + 1
    elif now < 11:
        sigma_color = saw_wave(now, 1) * 200
    else:
        sigma_space = saw_wave(now, 1) * 200
    return (
        cv2.bilateralFilter(frame, d, sigma_color, sigma_space),
        f"""Bilateral filter
        Frame: {frame_idx}
        d: {d}
        Sigma color: {sigma_color:.2f}
        Sigma space: {sigma_space:.2f}
        """,
    )


##########################################################################################
# 1.3: (8s) Grab object in color space


@effect(12, 20)
def effect_grab_object(frame: np.ndarray, now: float, frame_idx: int) -> np.ndarray:
    # Set RGB range
    lower_bgr = np.array([0, 0, 30])
    upper_bgr = np.array([50, 50, 255])
    mask_bgr = cv2.inRange(frame, lower_bgr, upper_bgr)

    # Set HSV range
    lower_hsv = np.array([150, 20, 20])
    upper_hsv = np.array([180, 255, 255])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Combine masks
    result = cv2.bitwise_and(frame, frame, mask=mask_bgr)
    result = cv2.bitwise_and(result, result, mask=mask_hsv)

    # Convert to binary
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, result = cv2.threshold(result, 1, 255, cv2.THRESH_BINARY)

    if now >= 16:
        # Apply morphological transformations
        kernel = np.ones((5, 5), np.uint8)
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

        # Compare to only BGR and HSV mask and color improvements red
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        result[mask_bgr > 0] = [0, 0, 255]
        result[mask_hsv > 0] = [0, 255, 0]
        # TODO: Revise this part
        # TODO: Print what the colors mean

    return (
        result,
        f"""Grab object
        Frame: {frame_idx}
        BGR range: {lower_bgr} - {upper_bgr}
        HSV range: {lower_hsv} - {upper_hsv}
        {f"Morphological transformations applied" if now >= 16 else ""}
        """,
    )


##########################################################################################
##########################################################################################

video.preview(from_seconds=12)
# video.describe()
