from pathlib import Path

import cv2
import numpy as np
from util import saw_wave, triangle_wave
from video import Video, make_effect_decorator

datapath = Path(__file__).parent.parent / "openshot-project"

second = 30
video = Video(int(1920 / 2), int(1080 / 2), second)
effect = make_effect_decorator(video)


# video.append_clip(datapath / "example1.mp4")
# video.append_clip(datapath / "example2.mp4")
# video.append_clip(datapath / "example3.mp4")
video.append_clip(datapath / "Orange on Adventures.mp4")

##########################################################################################
##########################################################################################
# 1: (0 - 20s) Basic image processing

##########################################################################################
# 1.1: (4s) Switch between grayscale and color


@effect(0, 4)
def effect_grayscale(frame: np.ndarray, now: float, frame_idx: int) -> np.ndarray:
    enabled = int(frame_idx / 10) % 2 == 0
    if enabled:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return (
        frame,
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
    lower_bgr = np.array([0, 12, 30])
    upper_bgr = np.array([176, 211, 255])
    mask_bgr = cv2.inRange(frame, lower_bgr, upper_bgr)

    # Set HSV range
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([4, 70, 131])
    upper_hsv = np.array([22, 255, 255])
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Combine masks
    mask = cv2.bitwise_and(mask_bgr, mask_hsv)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert to binary
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, result = cv2.threshold(result, 1, 255, cv2.THRESH_BINARY)

    if now >= 16:
        # Apply morphological transformations
        kernel = np.ones((5, 5), np.uint8)
        mask_morph = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        mask_morph = cv2.morphologyEx(mask_morph, cv2.MORPH_CLOSE, kernel)

        # Color added pixels red and removed pixels green
        result = cv2.cvtColor(mask_morph, cv2.COLOR_GRAY2BGR)
        result[(mask_morph > 0) & (mask == 0)] = [0, 255, 0]
        result[(mask_morph == 0) & (mask > 0)] = [0, 0, 255]

    return (
        result,
        f"""Grab object
        Frame: {frame_idx}
        BGR range: {lower_bgr} - {upper_bgr}
        HSV range: {lower_hsv} - {upper_hsv}
        {f"Morphological transformations applied" if now >= 12 else ""}
        Red: Added pixels
        Green: Removed pixels
        """,
    )


##########################################################################################
##########################################################################################
# 2: (20 - 40s) Object detection

##########################################################################################
# 2.1: (5s) Sobel edge detection


@effect(20, 25)
def effect_sobel(frame: np.ndarray, now: float, frame_idx: int) -> np.ndarray:
    kernel_size = 2 * int(triangle_wave(now, 5) * 2.99) + 1
    img_blur = cv2.GaussianBlur(frame, (3, 3), 0)
    sobelx = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=kernel_size)
    convx = cv2.convertScaleAbs(sobelx)
    convy = cv2.convertScaleAbs(sobely)
    # Overlay x edges in red and y edges in green
    convx[:, :, 0] = 0
    convx[:, :, 1] = 0
    convy[:, :, 0] = 0
    convy[:, :, 2] = 0
    result = cv2.addWeighted(convx, 0.5, convy, 0.5, 0)
    result = cv2.addWeighted(frame, 0.5, result, 0.5, 0)

    return (
        result,
        f"""Sobel edge detection
        Frame: {frame_idx}
        Kernel size: {kernel_size}
        """,
    )


##########################################################################################
# 2.2: (5s) Circle detection


@effect(25, 35)
def effect_circle_detection(
    frame: np.ndarray, now: float, frame_idx: int
) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    absx = cv2.convertScaleAbs(sobelx)
    absy = cv2.convertScaleAbs(sobely)
    gray = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Set parameters based on frame index
    if frame_idx <= 140:
        dp = 1.3
        param1 = 200
        param2 = int(25 * saw_wave(now, 5) + 25)
        min_radius = int(10 * saw_wave(now, 5)) + 35
        max_radius = int(20 * saw_wave(now, 5)) + 50
    elif frame_idx <= 235:
        dp = 1.1
        param1 = 200
        param2 = 50
        min_radius = 50
        max_radius = 100
    else:
        dp = 1.4
        param1 = 200
        param2 = 45
        min_radius = 40
        max_radius = 60

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=20,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
    return (
        frame,
        f"""Circle detection
        Frame: {frame_idx}
        Circles found: {len(circles[0]) if circles is not None else 0}
        Parameters:
        - dp: {dp}
        - minDist: 20
        - param1: {param1}
        - param2: {param2}
        - minRadius: {min_radius}
        - maxRadius: {max_radius}
        """,
    )


##########################################################################################
# 2.3: (5s) Gray scale map


@effect(35, 40)
def effect_main_object(frame: np.ndarray, now: float, frame_idx: int) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    absx = cv2.convertScaleAbs(sobelx)
    absy = cv2.convertScaleAbs(sobely)
    gray = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.4,
        minDist=20,
        param1=200,
        param2=50,
        minRadius=30,
        maxRadius=50,
    )

    # Set RGB range
    lower_bgr = np.array([25, 80, 210])
    upper_bgr = np.array([120, 150, 255])
    mask_bgr = cv2.inRange(frame, lower_bgr, upper_bgr)

    # Set HSV range
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([3, 150, 150])
    upper_hsv = np.array([14, 240, 255])
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Combine masks
    mask = cv2.bitwise_and(mask_bgr, mask_hsv)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert to binary
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, result = cv2.threshold(result, 1, 255, cv2.THRESH_BINARY)

    default_return = (
        frame,
        f"""Gray scale map
        Frame: {frame_idx}
        """,
    )

    best_circle = None
    if circles is None:
        return default_return

    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Bounding box
        x = i[0] - i[2] if i[0] >= i[2] else 0
        y = i[1] - i[2] if i[1] >= i[2] else 0
        d = 2 * i[2]

        # Get ratio of pixels in bounding box that are part of the object
        ratio = np.sum(result[y : y + d, x : x + d]) / (d * d) / 255
        # Store best circle
        if best_circle is None or ratio > best_circle[1]:
            best_circle = (i, ratio)

    if best_circle is None:
        return default_return

    i, ratio = best_circle
    x = i[0] - i[2] if i[0] >= i[2] else 0
    y = i[1] - i[2] if i[1] >= i[2] else 0
    d = 2 * i[2]

    if now >= 37:
        # Get mean values for RGB
        target_window = frame[y : y + d, x : x + d]
        target_bgr = np.mean(target_window, axis=(0, 1))

        # Calculate mean color for the neighborhood of each pixel
        frame_blurred = cv2.blur(frame, (d + 1, d + 1))

        # Calculate mean squared error
        mse = np.mean((frame_blurred - target_bgr) ** 2, axis=2)
        # Normalize to [0, 255]
        mse = (mse - np.min(mse)) / (np.max(mse) - np.min(mse)) * 255
        # Convert to uint8 and invert
        mse = 255 - mse.astype(np.uint8)
        # Create 3-channel image
        frame = cv2.merge([mse] * 3)
    else:
        # Draw bounding box and center
        cv2.rectangle(frame, (x, y), (x + d, y + d), (0, 255, 0), 2)
        cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)

    return (
        frame,
        f"""Gray scale map
        Frame: {frame_idx}
        """,
    )


##########################################################################################
##########################################################################################
# 3: (40 - 60s) Freestyle

##########################################################################################
# 3.1: Erode


@effect(40, 44)
def effect_erode(frame: np.ndarray, now: float, frame_idx: int) -> np.ndarray:
    kernel = np.ones((5, 5), np.uint8)
    iterations = int(triangle_wave(now, 5) * 10)
    frame = cv2.dilate(frame, kernel, iterations=iterations)
    return (
        frame,
        f"""Erode
        Frame: {frame_idx}
        Iterations: {iterations}
        """,
    )


##########################################################################################
# 3.2: Dilate


@effect(44, 48)
def effect_dilate(frame: np.ndarray, now: float, frame_idx: int) -> np.ndarray:
    kernel = np.ones((5, 5), np.uint8)
    iterations = int(triangle_wave(now, 5) * 10)
    frame = cv2.erode(frame, kernel, iterations=iterations)
    return (
        frame,
        f"""Dilate
        Frame: {frame_idx}
        Iterations: {iterations}
        """,
    )


##########################################################################################
# 3.3: Gradient


@effect(48, 52)
def effect_gradient(frame: np.ndarray, now: float, frame_idx: int) -> np.ndarray:
    kernel_size = 2 * int(triangle_wave(now, 5) * 3.99) + 3
    frame = cv2.morphologyEx(frame, cv2.MORPH_GRADIENT, (kernel_size, kernel_size))
    return (
        frame,
        f"""Gradient
        Frame: {frame_idx}
        Kernel size: {kernel_size}
        """,
    )


##########################################################################################
# 3.4: Image Segmentation with Watershed Algorithm


@effect(52, 56)
def effect_watershed(frame: np.ndarray, now: float, frame_idx: int) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(frame, markers)
    frame[markers == -1] = [0, 0, 255]
    return (
        frame,
        f"""Watershed
        Frame: {frame_idx}
        """,
    )


##########################################################################################
# 3.5: Contour


@effect(56, 59.5)
def effect_contour(frame: np.ndarray, now: float, frame_idx: int) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    return (
        frame,
        f"""Contour
        Frame: {frame_idx}
        Contours found: {len(contours)}
        """,
    )


##########################################################################################
##########################################################################################

video.preview(from_seconds=0)
# video.save(datapath / "output.mp4")
# video.describe()
