"""Run Phase 1 + 2: ingest video, detect people, log metrics.

Examples (from the project root):
  python scripts/run_detection.py --source data/videos/entrance.mp4
  python scripts/run_detection.py --source 0                  # webcam
  python scripts/run_detection.py --source rtsp://... --device cuda:0
  python scripts/run_detection.py --source data/videos/x.mp4 --no-show --save
"""
import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from vision_engine.detector import DetectionAnnotator, PersonDetector  # noqa: E402
from vision_engine.video_processor import VideoSource  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0")
    p.add_argument("--weights", default="yolo11n.pt")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu")
    p.add_argument("--fps", type=float, default=10.0, help="processing FPS")
    p.add_argument("--max-width", type=int, default=960)
    p.add_argument("--save", action="store_true", help="save annotated video")
    p.add_argument("--no-show", action="store_true")
    a = p.parse_args()

    out_dir = ROOT / "data" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = PersonDetector(a.weights, a.conf, imgsz=a.imgsz, device=a.device)
    annotator = DetectionAnnotator()
    writer = None
    infer_ms, counts = [], []

    csv_path = out_dir / "detections_log.csv"
    t_start = time.perf_counter()

    with VideoSource(a.source, a.fps, a.max_width) as src, \
            open(csv_path, "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["frame_index", "timestamp_s", "person_count",
                      "mean_conf", "inference_ms"])
        print(f"Source FPS {src.src_fps:.1f} -> processing ~{src.effective_fps:.1f} FPS")

        for fi in src.frames():
            t0 = time.perf_counter()
            det = detector.detect(fi.frame)
            ms = (time.perf_counter() - t0) * 1000

            n = len(det)
            mean_conf = float(np.mean(det.confidence)) if n else 0.0
            infer_ms.append(ms)
            counts.append(n)
            log.writerow([fi.index, f"{fi.timestamp_s:.2f}", n,
                          f"{mean_conf:.3f}", f"{ms:.1f}"])

            vis = annotator.annotate(fi.frame, det)
            cv2.putText(vis, f"People in frame: {n}  ({ms:.0f} ms)", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if a.save:
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(
                        str(out_dir / "detections.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"), src.effective_fps, (w, h))
                writer.write(vis)

            if not a.no_show:
                cv2.imshow("CrowdSense - detection (q to quit)", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    if writer:
        writer.release()
    cv2.destroyAllWindows()

    if infer_ms:
        total = time.perf_counter() - t_start
        print("\n=== Detection summary ===")
        print(f"Frames processed : {len(infer_ms)}")
        print(f"Avg inference    : {np.mean(infer_ms):.1f} ms "
              f"(p95 {np.percentile(infer_ms, 95):.1f} ms)")
        print(f"Pipeline FPS     : {len(infer_ms) / total:.1f}")
        print(f"People per frame : mean {np.mean(counts):.1f}, max {max(counts)}")
        print(f"Log saved to     : {csv_path}")


if __name__ == "__main__":
    main()
