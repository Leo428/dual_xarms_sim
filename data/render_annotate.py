import cv2
import os
from dataclasses import dataclass
from typing import List
import re
from datetime import datetime
import imageio

def wrap_text(text: str, max_width: int = 80) -> List[str]:
    import textwrap
    return textwrap.wrap(text, width=max_width)

@dataclass
class Annotation:
    start_time: float  # in seconds
    end_time: float
    description: str
    type: str  # "Action" or "Reasoning"

def parse_annotation_file(txt_path: str) -> List[Annotation]:
    from datetime import datetime

    def time_str_to_sec(t: str) -> float:
        dt = datetime.strptime(t.strip(), "%H:%M:%S.%f")
        return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6

    annotations = []
    with open(txt_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            # Split by tab, filter out empty columns
            raw_parts = line.strip().split('\t')
            parts = [p.strip() for p in raw_parts if p.strip()]
            if len(parts) < 7:
                print(f"Skipping malformed line (len={len(parts)}): {line.strip()}")
                continue
            try:
                ann_type = parts[0]
                start_sec = time_str_to_sec(parts[1])
                end_sec = time_str_to_sec(parts[3])
                description = " ".join(parts[7:]).strip()
                annotations.append(Annotation(start_sec, end_sec, description, ann_type))
            except Exception as e:
                print(f"Skipping line due to error: {e}\nLine: {line}")
    return annotations

def render_annotations(
    video_path: str,
    annotations: List[Annotation],
    output_path: str,
    font_scale: float = 0.6,
    font_thickness: int = 1
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    font = cv2.FONT_HERSHEY_SIMPLEX

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_sec = frame_idx / fps
        lines_to_draw = []

        for ann in annotations:
            if ann.start_time <= timestamp_sec <= ann.end_time:
                lines_to_draw.append((ann.type, ann.description))

        y = 50
        for ann_type, text in lines_to_draw:
            color = (0, 255, 0) if ann_type == "Action" else (0, 255, 255)
            wrapped = wrap_text(text, max_width=80)
            for line in wrapped:
                cv2.putText(frame, line, (20, y), font, font_scale, (0, 0, 0), font_thickness + 2, cv2.LINE_AA)
                cv2.putText(frame, line, (20, y), font, font_scale, color, font_thickness, cv2.LINE_AA)
                y += 25

        # Convert BGR to RGB for imageio
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_idx += 1

    cap.release()

    # Write using imageio
    imageio.mimsave(output_path, frames, fps=fps)

if __name__ == "__main__":
    ep_id = 11
    video_path = f"/home/huzheyuan/dual_xarms/dual_xarms_sim/data/real_hang_riya_0327/crop_real_hang_riya_0327_ep{ep_id}.mp4"
    txt_path = f"/home/huzheyuan/Downloads/crop_real_hang_riya_0327_ep{ep_id}.txt"
    output_path = video_path.replace(".mp4", "_annotated.mp4")

    annotations = parse_annotation_file(txt_path)
    for a in annotations:
        print(f"{a.type}: {a.start_time:.2f}s - {a.end_time:.2f}s -> {a.description}")

    render_annotations(video_path, annotations, output_path)
    print(f"Annotated video saved to: {output_path}")
