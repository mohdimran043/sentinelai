"""One real, end-to-end pass with real models over the downloaded Avenue
clip — no mediamtx, no RTSP, no RabbitMQ, no MinIO, so it only needs the GPU
extra and the dataset script's output, not the full docker-compose stack.
This is the closest thing to an automated version of the manual demo in
ai-engine/README.md.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from sentinel_ai.adapters.detectors.yolo11 import Yolo11Detector, select_device
from sentinel_ai.adapters.sources.file import FileSource
from sentinel_ai.adapters.sources.preroll import PreRollBuffer
from sentinel_ai.adapters.trackers.bytetrack import ByteTrackTracker
from sentinel_ai.adapters.vision.qwen25vl import Qwen25VLDescriber
from sentinel_ai.domain.camera_profile import CameraProfile
from sentinel_ai.orchestrator.admission import AdmissionGate
from sentinel_ai.orchestrator.scheduler import VlmScheduler
from sentinel_ai.pipeline.runner import CameraRunner
from sentinel_ai.pipeline.stages.motion import MotionAnalyzer
from tests.fakes.io import FakePublisher

AVENUE_CLIP = Path(__file__).resolve().parents[3] / "datasets" / "avenue" / "avenue_01.mp4"


@pytest.mark.gpu
@pytest.mark.integration
async def test_full_pipeline_produces_a_real_event_with_real_models() -> None:
    if not AVENUE_CLIP.exists():
        pytest.skip(f"run datasets/download_sample.sh first — {AVENUE_CLIP} not found")

    device = select_device()
    detector = Yolo11Detector(model_id="yolo11s.pt", conf=0.35, iou=0.45, imgsz=640, device=device)
    vlm = Qwen25VLDescriber(
        model_id="Qwen/Qwen2.5-VL-3B-Instruct-AWQ", max_new_tokens=256, device=device
    )
    await detector.initialize()
    await detector.warmup()
    await vlm.initialize()
    await vlm.warmup()

    publisher = FakePublisher()
    admission = AdmissionGate(concurrency=1, min_interval_seconds=0.0)
    scheduler = VlmScheduler(
        vlm,
        publisher,
        admission,
        maxsize=4,
        timeout_seconds=30.0,
        clock=time.monotonic,
    )
    worker = asyncio.create_task(scheduler.run())

    source = FileSource(str(AVENUE_CLIP), camera_id="avenue_01", realtime=False)
    runner = CameraRunner(
        camera_id="avenue_01",
        camera_label="Avenue (demo)",
        source=source,
        detector=detector,
        tracker=ByteTrackTracker(frame_rate=25),
        motion=MotionAnalyzer(),
        profile=CameraProfile(camera_id="avenue_01"),
        scheduler=scheduler,
        clip_writer=None,
        preroll=PreRollBuffer(preroll_seconds=3.0),
        clock=time.monotonic,
    )
    try:
        await runner.run()
        await scheduler.drain()
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        await detector.shutdown()
        await vlm.shutdown()

    assert publisher.events, "no event was produced over the whole clip"
    event = publisher.events[0]
    assert event.description
    assert 0.0 <= event.threat.value <= 1.0
