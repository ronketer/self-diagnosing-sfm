"""Extract sharp, evenly-spaced frames from a capture video for SfM.

Splits the video into n_frames equal windows and keeps the frame with the
highest Laplacian variance (sharpness) in each window. Writes downscaled
JPEGs plus an f35_equiv.txt with the 35mm-equivalent focal length so the
phone-capture loader can build K without EXIF.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out_dir")
    ap.add_argument("--n-frames", type=int, default=70)
    ap.add_argument("--max-dim", type=int, default=1024)
    ap.add_argument("--f35", type=float, default=26.0,
                    help="35mm-equivalent focal length of the video mode")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    bounds = np.linspace(0, total, args.n_frames + 1, dtype=int)
    win = 0
    best_score, best_frame = -1.0, None
    saved = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # score sharpness on a small grayscale version (fast)
        small = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if score > best_score:
            best_score, best_frame = score, frame
        idx += 1
        if idx >= bounds[win + 1]:
            if best_frame is not None:
                h, w = best_frame.shape[:2]
                s = args.max_dim / max(h, w)
                resized = cv2.resize(
                    best_frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA
                )
                cv2.imwrite(str(out / f"frame_{saved:04d}.jpg"), resized,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1
            win += 1
            best_score, best_frame = -1.0, None
            if win >= args.n_frames:
                break

    (out / "f35_equiv.txt").write_text(str(args.f35))
    print(f"saved {saved} frames to {out}")


if __name__ == "__main__":
    main()
