from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np


##########################################################################################


@dataclass
class Effect:
    from_frame: int  # Inclusive
    to_frame: int  # Exclusive

    """Function to apply to the frame.
    Input:
        frame (np.ndarray),
        current time in seconds (float),
        frame number relative to from_frame (int)
    Output: frame (np.ndarray)
    """
    func: Callable[[np.ndarray, float, int], Tuple[np.ndarray, str | None]]


##########################################################################################


class Video:
    """Class to handle video files. A video can consist of multiple clips.

    NOTE: The frame rate is assumed to be the same for all clips.
    """

    def __init__(self, width: int, height: int, fps: float) -> None:
        self.resolution = (width, height)
        self.fps = fps
        self.clips: List[cv2.VideoCapture] = []
        self.effects: List[Effect] = []

    def append_clip(self, filepath: Path) -> None:
        cap = cv2.VideoCapture(filepath.as_posix())
        if not cap.isOpened():
            raise Exception("Error: Could not open video.")
        self.clips.append(cap)

    def add_effect(
        self,
        from_seconds: float,
        to_seconds: float,
        func: Callable,
    ) -> None:
        """Add an effect to the video.

        Args:
            from_seconds (float): Start time of the effect.
            to_seconds (float): End time of the effect.
            func (Callable): Function to apply to the frame.
        """
        # Convert seconds to frames
        from_frame = int(from_seconds * self.fps)
        to_frame = int(to_seconds * self.fps)
        # Add effect
        self.effects.append(Effect(from_frame, to_frame, func))

    def preview(self, from_seconds=0) -> None:
        cur_frame = 0
        for cap in self.clips:
            while cap.isOpened():
                # Read frame and check if it should be displayed
                ret, frame = cap.read()
                if not ret:
                    break
                cur_frame += 1
                if cur_frame < from_seconds * self.fps:
                    continue

                # Apply effects
                frame = cv2.resize(frame, self.resolution)
                subtitle = None
                for effect in self.effects:
                    if not (effect.from_frame <= cur_frame < effect.to_frame):
                        continue
                    results = effect.func(
                        frame, cur_frame / self.fps, cur_frame - effect.from_frame
                    )
                    # Allow subtitle to be an optional return value
                    try:
                        frame, subtitle = results
                    except ValueError:
                        frame = results

                # Add subtitle
                if subtitle is not None:
                    y0, dy = 30, 20
                    lines = [l.strip() for l in subtitle.splitlines()]
                    lines = [l for l in lines if l]
                    for i, line in enumerate(lines):
                        y = y0 + i * dy
                        cv2.putText(
                            frame,
                            line,
                            (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                            False,
                        )

                # Display frame and check for exit
                cv2.imshow("Frame", frame)
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break

            cap.release()
        cv2.destroyAllWindows()

    def describe(self) -> None:
        total_frames = 0
        for i, cap in enumerate(self.clips):
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames += num_frames
            print(f"Clip {i + 1}")
            print("Frame width:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            print("Frame height:", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            print("Frame count:", num_frames)
            print("Frame rate:", cap.get(cv2.CAP_PROP_FPS))
            print()
        print("Total frames:", total_frames)


##########################################################################################


def make_effect_decorator(video: Video):
    # Decorator to add effects to the video
    def effect(from_seconds: float, to_seconds: float):
        def decorator(func):
            video.add_effect(from_seconds, to_seconds, func)
            return func

        return decorator

    return effect
