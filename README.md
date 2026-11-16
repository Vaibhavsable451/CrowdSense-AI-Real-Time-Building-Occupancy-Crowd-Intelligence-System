# CrowdSense-AI-Real-Time-Building-Occupancy-Crowd-Intelligence-System
<div align="center">

# 👥 CrowdSense AI

### Real-Time Building Occupancy & Space Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![YOLO11](https://img.shields.io/badge/Detection-YOLO11-00A3E0)
![ByteTrack](https://img.shields.io/badge/Tracking-ByteTrack-8A2BE2)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Phase%201--2%20Complete-success)
![Privacy](https://img.shields.io/badge/Privacy-No%20Face%20Recognition-blue)

**Detect → Track → Count → Forecast — without facial recognition.**

</div>

---

## 📑 Table of Contents

- [Highlights](#-highlights)
- [System Architecture](#-system-architecture)
- [Processing Pipeline](#-processing-pipeline-sequence)
- [Tech Stack](#-tech-stack)
- [Quick Start](#-quick-start-windows-powershell)
- [CLI Reference](#-cli-reference)
- [Roadmap](#-roadmap)
- [Evaluation Metrics](#-evaluation-metrics)
- [Privacy & Ethics](#-privacy--ethics)
- [Project Structure](#-project-structure)
- [Datasets](#-datasets)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Highlights

| Feature | Description | Status |
|---|---|---|
| 🎯 Person detection | YOLO11 on CPU or NVIDIA GPU | ✅ Done |
| 🎞️ Video ingestion | MP4, webcam, RTSP with FPS throttling | ✅ Done |
| 🧭 Multi-object tracking | ByteTrack temporary anonymous IDs | ⏳ Phase 3 |
| 🚪 Entry/exit counting | Virtual line crossing logic | ⏳ Phase 4 |
| 🗺️ Zones, density, heatmaps | Per-area occupancy intelligence | ⏳ Phase 5 |
| 📈 Peak hours & forecasting | Time-series analytics | ⏳ Phase 6 |
| ⚡ REST API | FastAPI + SQLAlchemy | ⏳ Phase 7 |
| 📊 Live dashboard | Streamlit + Plotly | ⏳ Phase 8 |
| 🔒 Privacy-first | No face recognition, anonymous IDs only | ✅ By design |

---

## 🏗️ System Architecture

```mermaid
flowchart TB
    subgraph SRC["📹 1. Video Sources"]
        direction LR
        S1[CCTV / RTSP]
        S2[Webcam]
        S3[MP4 File]
    end

    subgraph VIS["🧠 2. Vision Engine"]
        direction TB
        V1[OpenCV Ingestion<br/>resize + FPS throttle]
        V2[YOLO11 Detection<br/>person class only]
        V3[ByteTrack<br/>temporary track IDs]
        V1 ==> V2 ==> V3
    end

    subgraph ANA["⚙️ 3. Analytics Engine"]
        direction TB
        A1[Entry / Exit Counter<br/>virtual lines]
        A2[Zone Manager<br/>polygons + density]
        A3[Heatmap Generator]
        A4[Occupancy Reconciler<br/>start count + drift check]
        A5[Forecasting<br/>peak hours]
    end

    subgraph DATA["💾 4. Storage"]
        direction TB
        D1[(SQL Database)]
        D2[/CSV Logs & Annotated Video/]
    end

    subgraph SRV["🌐 5. Serving Layer"]
        direction TB
        B1[FastAPI Backend<br/>REST + WebSocket]
        B2[Streamlit Dashboard]
        B1 ==> B2
    end

    U((👤 Facility Manager))

    S1 --> V1
    S2 --> V1
    S3 --> V1

    V3 --> A1
    V3 --> A2
    V3 --> A3
    A1 --> A4
    A2 --> A5
    A4 --> A5

    A4 ==> D1
    A2 ==> D1
    A3 ==> D1
    A5 ==> D1
    V2 -.-> D2

    D1 ==> B1
    B2 ==> U
    U -. config: lines, zones .-> B1

    classDef done fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#000
    classDef todo fill:#fff3cd,stroke:#ffc107,stroke-width:2px,color:#000
    class V1,V2,S1,S2,S3,D2 done
    class V3,A1,A2,A3,A4,A5,D1,B1,B2 todo
```

> 🟩 Green = implemented  ·  🟨 Yellow = planned  ·  `==>` main data flow  ·  `-.->` side channel / config

---

## 🔄 Processing Pipeline (Sequence)

```mermaid
sequenceDiagram
    autonumber
    participant Src as Video Source
    participant VP as VideoSource (OpenCV)
    participant Det as YOLO11 Detector
    participant Trk as ByteTrack
    participant Cnt as Entry/Exit Counter
    participant DB as Database
    participant API as FastAPI
    participant UI as Streamlit

    loop every processed frame
        Src->>VP: raw frame
        VP->>VP: resize + throttle to target FPS
        VP->>Det: frame
        Det->>Trk: person boxes + confidence
        Trk->>Cnt: boxes with temporary IDs
        Cnt->>DB: IN / OUT events + occupancy
    end

    UI->>API: GET /occupancy/live
    API->>DB: query latest state
    DB-->>API: occupancy, flow, zones
    API-->>UI: JSON response
    UI-->>UI: render charts and heatmap
```

---

## 🧱 Tech Stack

| Layer | Technology |
|---|---|
| Detection | YOLO11 (Ultralytics) |
| Tracking | ByteTrack |
| Video | OpenCV |
| Utilities | Supervision |
| Backend | FastAPI + SQLAlchemy |
| Dashboard | Streamlit + Plotly |
| Analytics | NumPy, Pandas, scikit-learn |

---

## 🚀 Quick Start (Windows PowerShell)

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# put a video in data/videos/, then:
python scripts/run_detection.py --source data/videos/car.mp4 --save
```

> 💡 If the filename has spaces, wrap it in quotes:
> `--source "data/videos/Untitled design.mp4"`

### Other source types

```powershell
# Webcam
python scripts/run_detection.py --source 0

# RTSP camera
python scripts/run_detection.py --source "rtsp://user:pass@192.168.1.10:554/stream"

# NVIDIA GPU, headless, saved output
python scripts/run_detection.py --source data/videos/car.mp4 --device cuda:0 --no-show --save
```

---

## 🎛️ CLI Reference

| Flag | Description | Default |
|---|---|---|
| `--source` | MP4 path, webcam index, or RTSP URL | `0` |
| `--weights` | YOLO weights | `yolo11n.pt` |
| `--conf` | Confidence threshold | `0.35` |
| `--device` | `cpu` or `cuda:0` | `cpu` |
| `--fps` | Processing FPS | `10` |
| `--max-width` | Resize frames to this width | — |
| `--save` | Save annotated video | off |
| `--no-show` | Headless mode | off |

**Outputs** go to `data/output/`:

| File | Content |
|---|---|
| `detections.mp4` | Annotated video |
| `detections_log.csv` | Per-frame detection log |

---

## 🗺️ Roadmap

```mermaid
flowchart LR
    P1([Phase 1<br/>Ingestion]) --> P2([Phase 2<br/>Detection])
    P2 --> P3([Phase 3<br/>Tracking])
    P3 --> P4([Phase 4<br/>Counting])
    P4 --> P5([Phase 5<br/>Zones + Heatmaps])
    P5 --> P6([Phase 6<br/>Forecasting])
    P6 --> P7([Phase 7<br/>FastAPI])
    P7 --> P8([Phase 8<br/>Dashboard])
    P8 --> P9([Phase 9<br/>Testing + Optimization])

    style P1 fill:#28a745,color:#fff
    style P2 fill:#28a745,color:#fff
    style P3 fill:#ffc107,color:#000
```

- [x] Phase 1 — Video ingestion
- [x] Phase 2 — YOLO person detection
- [ ] Phase 3 — ByteTrack tracking
- [ ] Phase 4 — Entry/exit counting
- [ ] Phase 5 — Zones, density, heatmaps
- [ ] Phase 6 — Business analytics & forecasting
- [ ] Phase 7 — FastAPI backend
- [ ] Phase 8 — Streamlit dashboard
- [ ] Phase 9 — Testing & optimization

### Planned API endpoints (Phase 7)

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/occupancy/live` | Current building occupancy |
| `GET` | `/flow?from=&to=` | Entries/exits over time |
| `GET` | `/zones/{id}/density` | Zone density |
| `GET` | `/forecast/peak-hours` | Predicted busy periods |
| `POST` | `/config/lines` | Define counting lines |
| `WS` | `/ws/stream` | Live updates |

---

## 📏 Evaluation Metrics

| Area | Metric |
|---|---|
| Counting | MAE (mean absolute error) |
| Detection | Precision / Recall |
| Tracking | IDF1 / HOTA |
| Performance | FPS, end-to-end latency |

---

## 🔐 Privacy & Ethics

- Use **consented footage only** and restrict access.
- **Never** perform facial identification.
- Track IDs are **temporary and anonymous**.
- A frame's person count is **not** building occupancy. True occupancy needs tracking, calibrated lines, a known starting count and reconciliation checks.

---

## 📁 Project Structure

```
CrowdSense-AI/
├── backend/app/{api,core,models,database}
├── vision_engine/        # video_processor, detector (+ tracker, counter, zones…)
├── analytics/
├── dashboard/
├── data/
│   ├── videos/           # input footage (git-ignored)
│   ├── processed/
│   └── output/           # detections.mp4, detections_log.csv
├── scripts/run_detection.py
├── tests/
├── requirements.txt
└── README.md
```

---

## 📚 Datasets

MOTChallenge · CrowdHuman · ShanghaiTech Crowd Counting · your own staged entrance/exit videos.

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot open video source` | Check the exact filename with `dir data\videos`; quote paths with spaces |
| Video still won't open | Re-export as H.264 MP4 |
| Slow on CPU | Lower `--fps`, use `--max-width 640`, or switch to `--device cuda:0` |
| `Activate.ps1` blocked | Run `Set-ExecutionPolicy -Scope Process RemoteSigned` |
