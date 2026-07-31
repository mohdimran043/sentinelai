# SENTINELAI — MASTER DEVELOPMENT PROMPT

You are a Senior AI Architect, Senior Python AI Engineer, Senior Go Backend Engineer, Senior React Engineer, Senior DevOps Engineer, Senior ML Engineer, and Senior UX Designer.

Your objective is to build an enterprise-grade AI Behavior Intelligence Platform called **SentinelAI**.

This system must be production-ready, modular, scalable, GPU-optimized, and designed using microservices. Every service must be independently deployable.

---

## 1. PROJECT OBJECTIVE

SentinelAI is an AI-powered behavior intelligence platform capable of monitoring multiple cameras simultaneously, understanding activities using AI models and Vision LLMs, learning normal behavior over time, detecting anomalies in real time, generating intelligent summaries, storing searchable behavioral history, and notifying users instantly with evidence clips.

The system should support:

- Unlimited cameras
- Live RTSP streams
- Uploaded videos
- Multi-GPU deployment
- Multi-machine deployment
- Cloud deployment
- Edge deployment

The goal is NOT to simply detect motion.

The goal is to **understand behavior**.

The goal is also NOT to run as many AI models as possible.

The goal is to achieve the **highest possible intelligence using the fewest AI models and the least GPU resources**.

---

## 2. VERY IMPORTANT ARCHITECTURE REQUIREMENT

The system MUST be divided into TWO completely independent systems.

**System 1 — AI Engine**

**System 2 — Web Platform**

Neither system should depend on the other internally.

Communication must happen ONLY through:

- REST APIs
- gRPC
- RabbitMQ
- Kafka (optional)
- WebSockets

No AI model should ever execute inside the web application.

The web application should only visualize information.

The AI Engine must expose exactly ONE entry point to the Web Platform: the **AI Orchestrator**.

The Web Platform must never call YOLO, Qwen, OCR, Whisper, or any model directly, and must never know which model produced a given result.

---

## 3. AI MODEL EXECUTION STRATEGY

The platform MUST be designed to use the minimum number of AI models required to achieve high accuracy.

The objective is NOT to run every available model simultaneously.

The objective is to intelligently orchestrate AI models to minimize GPU usage, inference time, latency, VRAM consumption, and operational cost.

### 3.1 Development Mode

During development, the system must run on a single RTX 4090 GPU.

Development Mode should load ONLY ONE primary AI pipeline.

Recommended default pipeline:

- YOLO11 (Object Detection)
- ByteTrack (Tracking)
- Qwen2.5-VL (Vision Reasoning)
- Whisper (only if audio is enabled)

No additional models should be loaded unless explicitly enabled.

The AI Engine must expose a configuration option:

- Development Mode
- Production Mode

The application must default to **Development Mode**.

### 3.2 Production Mode

In Production Mode, each AI capability should be an independent microservice, and each model should be deployable on a different machine.

Example:

| Server | Capability |
|---|---|
| Server A | YOLO Detection |
| Server B | Tracking |
| Server C | Vision LLM |
| Server D | Action Recognition |
| Server E | OCR |
| Server F | Face Recognition |
| Server G | Embeddings |

The AI Orchestrator decides which model to invoke.

Models must never communicate directly with each other.

Only the Orchestrator coordinates inference.

---

## 4. AI ORCHESTRATOR

Implement an AI Orchestrator responsible for:

- Model discovery
- Health monitoring
- Load balancing
- GPU allocation
- Model selection
- Queue management
- Retry logic
- Failover
- Result aggregation

The Orchestrator exposes a single API to the Go Backend.

This is a core architectural contribution of the project: the platform decides **which model to run, when to run it, and whether existing metadata is already sufficient**, rather than exposing individual models to the application layer.

---

## 5. MODEL LIFECYCLE AND LOADING

Every model must support the following states:

- Loaded
- Unloaded
- Sleeping
- Downloading
- Updating
- Offline
- Healthy
- Unhealthy

Models must be loaded lazily.

If a model has not been used recently, it must be unloaded automatically to free GPU memory.

Model switching must be possible without restarting the system.

---

## 6. MODEL EXECUTION POLICY

Never execute expensive Vision LLMs on every frame.

Instead implement a staged inference pipeline:

```
Stage 1  Video Decode
   ↓
Stage 2  Object Detection
   ↓
Stage 3  Tracking
   ↓
Stage 4  Motion Analysis
   ↓
Stage 5  Rule Engine
   ↓
If nothing interesting is detected  →  Store metadata only
   ↓
If unusual behavior is detected
   ↓
Invoke Vision LLM
   ↓
Generate explanation
   ↓
Generate event
   ↓
Send notification
```

The Vision LLM should execute ONLY when:

- A significant scene change occurs
- The user requests a live description
- A potential anomaly is detected
- A periodic scene summary is due (configurable, e.g. every 30–60 seconds)

---

## 7. MODEL REUSE

The system must maximize reuse of inference results.

Example — YOLO detects Person, Bag, Car, Door.

These detections must be reused by:

- Tracking
- Behavior Engine
- Vision LLM
- Reports
- Timeline
- Search

Do NOT rerun detection for each downstream task.

---

## 8. METADATA FIRST, MODEL SECOND

Always attempt to answer using existing metadata before invoking another AI model.

Example:

> "How many people entered today?"

Use stored detections and event metadata. Do NOT call the Vision LLM.

> "What was the suspicious behavior?"

Only then invoke the Vision LLM, using the stored keyframes and event context.

---

## 9. CONFIGURABLE MODEL PIPELINES

Each camera must support selecting a predefined pipeline:

- Pipeline 1 — Detection + Tracking
- Pipeline 2 — Detection + Tracking + Vision LLM
- Pipeline 3 — Detection + Tracking + OCR
- Pipeline 4 — Detection + Tracking + Face Recognition
- Pipeline 5 — Full Analysis

Users must also be able to create custom pipelines.

---

## 10. FUTURE MODEL SUPPORT

The architecture must allow adding new AI models without modifying existing services.

Every model must implement a common interface:

```
Initialize()
Health()
Predict()
Warmup()
Shutdown()
Version()
Capabilities()
```

This enables plug-and-play integration of future AI models.

---

## 11. DEPLOYMENT

Support deployment like this:

```
Machine 1
  React Frontend
     ↓
  Go Backend API
     ↓
  PostgreSQL / Redis / RabbitMQ / ElasticSearch / MinIO
     ↓
Machine 2
  Python AI Engine
     AI Orchestrator  (single entry point)
     YOLO
     Tracking
     Behavior Engine
     Vision LLM
     Whisper
     Embedding Service
     ↓
Machine 3
  Additional GPU Workers
     ↓
Machine N
  Additional AI Workers
```

The AI Engine must scale horizontally.

The Web Platform must scale independently.

---

## 12. WEB FRONTEND

Frontend MUST be developed using:

- React 19
- TypeScript
- Vite
- TailwindCSS
- ShadCN UI
- React Query
- React Router
- React Flow
- Recharts
- Framer Motion
- Socket.IO Client
- React Hook Form
- Zustand

No Angular.

---

## 13. BACKEND

- Go
- Echo Framework
- JWT Authentication
- REST API
- WebSocket
- Redis
- RabbitMQ
- ElasticSearch
- PostgreSQL
- MinIO

---

## 14. AI ENGINE

- Python
- FastAPI
- PyTorch
- ONNX Runtime
- TensorRT
- CUDA
- OpenCV
- FFmpeg

---

## 15. SUPPORTED AI MODELS

These are models the architecture must be *capable* of hosting. They must NOT all be loaded at once — loading is governed by Sections 3–9.

| Capability | Models |
|---|---|
| Object Detection | YOLO11, YOLO12, RT-DETR, YOLO-NAS |
| Tracking | ByteTrack, BoT-SORT, DeepSORT |
| Pose Estimation | RTMPose, YOLO Pose |
| Segmentation | SAM2 |
| Scene Understanding | Florence2 |
| Action Recognition | VideoMAE, InternVideo2, SlowFast |
| Vision LLM | Qwen2.5-VL, LLaVA, Gemma Vision, GPT API, Gemini API |
| Face Recognition | InsightFace |
| OCR | PaddleOCR |
| Audio | Whisper Large V3 |
| Embeddings | BGE, Jina Embeddings, Qdrant |

---

## 16. CAMERA MANAGEMENT

Support unlimited cameras.

Each camera should contain:

- Camera Name
- Location
- Zone
- Description
- RTSP URL
- ONVIF
- Status
- Retention
- Recording Schedule
- Alert Sensitivity
- Pipeline (see Section 9)
- Behavior Model
- Detection Model
- Vision LLM
- Embedding Collection
- Privacy Mask
- GPU Assignment
- Storage Location

Users must be able to assign different AI models and different pipelines to different cameras.

### 16.1 Per-Camera Model Selection

- Detection Model — YOLO11, YOLO12, RT-DETR, Custom
- Behavior Model — VideoMAE, InternVideo, Default
- Vision Model — Qwen2.5-VL, Gemini, GPT, Gemma Vision, LLaVA
- Scene Model — Florence2, SAM2

Every camera can use different models. The Orchestrator is responsible for resolving these selections to the models actually resident on the available GPUs.

---

## 17. VIDEO INPUT

Support:

- RTSP
- ONVIF
- USB Camera
- IP Camera
- Uploaded Videos
- Recorded Videos
- Video Files

---

## 18. DASHBOARD

The dashboard should display:

- Camera Grid
- Online Cameras
- Offline Cameras
- GPU Usage
- Storage Usage
- Today's Alerts
- Today's Visitors
- Today's Events
- Today's AI Summary
- Camera Status
- Live Notifications
- System Health

Clicking a camera opens a dedicated page containing:

- Live Video
- Live AI Description
- Detected Objects
- Current Behavior
- Threat Score
- Behavior Timeline
- Replay Timeline
- Past Events
- Daily Journal
- Heatmap
- Statistics
- Model Status
- GPU Assignment

The live AI description should continuously update. Example:

> **Current Scene** — Two people are talking near the entrance. One person is holding a backpack. No abnormal activity detected.

---

## 19. EVENT TIMELINE

Every event should appear on an interactive timeline:

```
08:12  Person entered      [Replay]
08:15  Delivery arrived    [Replay]
08:20  Vehicle left        [Replay]
08:45  Unknown visitor     [Replay]
```

Clicking Replay opens the video starting 3 seconds before the event.

---

## 20. BEHAVIOR ENGINE

Each camera must learn independently.

Learn:

- Normal visitors
- Schedules
- Object positions
- Employee routines
- Animal activity
- Lighting conditions
- Vehicle routines
- Behavior frequency
- Duration
- Movement paths

After learning:

- Compare live activity
- Generate anomaly score
- Generate explanation

---

## 21. ANOMALY DETECTION

Detect:

- Unknown visitors
- Loitering
- Running
- Falls
- Fight
- Fire
- Smoke
- Crowding
- Restricted area entry
- Vehicle anomalies
- Objects left behind
- Objects removed
- Suspicious behavior
- Tailgating
- Camera obstruction
- Camera movement
- Abnormal silence
- Abnormal noise

---

## 22. LIVE AI DESCRIPTION

When invoked (per Section 6), the Vision LLM receives:

- Current frame
- Tracked objects
- Behavior history
- Previous summaries
- Camera profile
- Current detections

And generates:

- Natural language description
- Behavior explanation
- Threat score
- Suggested action

---

## 23. NOTIFICATIONS

Real-time notifications through:

- Push
- Email
- Telegram
- WhatsApp
- Slack
- Teams
- Webhook
- SMS

Every notification must contain:

- Camera
- Time
- Severity
- Confidence
- AI Summary
- Threat Score
- Snapshot
- Replay Clip

The replay clip MUST begin 3 seconds BEFORE the anomaly and continue until the anomaly ends.

---

## 24. SEARCH

Natural language search. Examples:

- Show everyone wearing red.
- Show all visitors yesterday.
- Show suspicious activity.
- Show people carrying boxes.
- Show blue vehicles.
- Show unknown visitors.
- Show delivery persons.
- Show dogs.
- Show falls.
- Show children.
- Show vehicles parked over 2 hours.

Search must be answered from stored metadata, detections, and embeddings wherever possible (Section 8).

---

## 25. CHARTS

Interactive charts:

- Events per hour
- Events per day
- Threat trend
- Camera health
- GPU utilization
- Storage
- Alerts
- Behavior trend
- Anomaly frequency

Clicking a chart point opens the replay.

---

## 26. REPLAY

Replay must contain:

- Timeline
- AI explanation
- Detected objects
- Bounding boxes
- Threat score
- Behavior summary

Replay starts 3 seconds before the anomaly.

---

## 27. DAILY JOURNAL

Generated automatically. Includes:

- Summary
- Visitors
- Vehicles
- Interesting events
- Behavior changes
- Threat analysis
- Recommendations

---

## 28. REPORTS

Generate PDF, Excel, and CSV reports on Daily, Weekly, Monthly, and Custom ranges.

---

## 29. AI CHAT

The user can ask:

- What happened yesterday?
- Who entered after midnight?
- Show suspicious activity.
- Show all visitors.
- What changed this week?
- Generate incident report.

The chat layer must follow the metadata-first policy in Section 8 and only escalate to a Vision LLM when stored metadata is insufficient.

---

## 30. MODEL MANAGEMENT DASHBOARD

The dashboard must display, for every model:

- Installed Models
- Downloading
- Loaded
- Unloaded
- GPU Usage
- VRAM Usage
- Inference Speed
- Health
- Online / Offline

It must expose the lifecycle states from Section 5 and allow switching models without restarting the system.

---

## 31. ACTIVE LEARNING

Every event must allow user feedback:

- Correct
- Incorrect
- False Positive
- Missed Detection
- Need Review

Store all feedback.

---

## 32. RETRAINING PIPELINE

The system must continuously improve.

```
Collect difficult samples
   ↓
Collect false positives
   ↓
Collect false negatives
   ↓
User feedback
   ↓
Auto labeling
   ↓
Human validation
   ↓
Dataset versioning
   ↓
Retrain selected models
   ↓
Evaluate
   ↓
Compare with previous version
   ↓
Deploy
   ↓
Rollback if necessary
```

Support model versioning, A/B testing, scheduled retraining, manual retraining, and automatic retraining.

---

## 33. DATABASE

- PostgreSQL
- ElasticSearch
- Qdrant
- Redis
- MinIO

---

## 34. GPU OPTIMIZATION AND PERFORMANCE

Optimize primarily for a single NVIDIA RTX 4090 in development, and for multi-GPU / multi-machine deployment in production.

The system must continuously optimize GPU utilization by:

- Keeping lightweight models resident in memory
- Loading large models only on demand
- Sharing inference results between services
- Batching requests when possible
- Using TensorRT / ONNX optimization where supported
- Automatically unloading idle models
- Supporting multiple GPUs in production
- Supporting distributed inference across multiple AI servers

Run lightweight models continuously. Run Vision LLMs only under the conditions listed in Section 6.

Never waste GPU resources.

---

## 35. CODE QUALITY

- Use Clean Architecture
- Use Domain Driven Design
- Use SOLID principles
- Every microservice must be independently deployable
- Every service must expose Swagger documentation
- Every service must include a Dockerfile
- Provide Docker Compose
- Support Kubernetes deployment
- Support horizontal scaling
- Write production-quality code
- No placeholder implementations
- No mock logic unless explicitly requested

Every component must be modular, testable, documented, and suitable for an MSc BITS Pilani final project with enterprise-level architecture and future commercial scalability.
