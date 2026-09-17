"""Runtime configuration. This is the ONLY place the Development/Production seam is chosen."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from sentinel_ai.domain.entities import Severity


class Mode(StrEnum):
    """Spec §3.1/§3.2. Development binds in-process transport; Production binds gRPC."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"


class NotifierKind(StrEnum):
    """Which `Notifier` the composition root builds (spec §5, T10).

    An enum rather than "a webhook URL is set, so use a webhook": those are two
    different facts, and conflating them makes turning notifications off require
    deleting the URL — so an operator silencing a site for an afternoon has to
    keep the credential somewhere else and paste it back. It also gives the
    unreachable-webhook case a name: `webhook` with no URL is a configuration
    error a deployment can be told about, where the implicit form would silently
    fall back to logging and look like it worked.
    """

    LOGGING = "logging"
    WEBHOOK = "webhook"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SENTINEL_", env_file=".env", extra="ignore")

    mode: Mode = Mode.DEVELOPMENT

    cameras_file: str = "./cameras.json"
    """Where `sentinel_ai.main` reads its camera list from (see that module's
    docstring for the file's shape). A file rather than environment variables:
    a camera is a nested record with a per-camera `CameraProfile`, and flattening
    a list of those into `SENTINEL_*` names is a worse interface than one small
    JSON document that can be diffed, reviewed and mounted into a container."""

    enable_camera_writes: bool = False
    """Whether `PATCH /cameras/{id}` may change anything. **Off by default.**

    This engine has no authentication — the console's login screen is a shell and
    JWT is Phase 1C — so every endpoint is open to whatever can reach the port. For
    the read endpoints that is a disclosure question; for a write endpoint it is a
    control question, and the two are not the same size. A camera's label and zone
    are what an operator navigates by and what a console groups by, so an anonymous
    caller who can rewrite them can make a camera look like a different camera, and
    the change persists to `cameras.json` and survives the restart that would
    otherwise undo it.

    Default-off means a deployment that has not thought about this is not writable
    by anyone who finds it, and the operator who turns it on is the one who decided
    the port is reachable only by people who should be able to do this. The route
    still exists in the OpenAPI document either way — a contract that changes shape
    with a runtime flag is worse than a documented 403 — and answers 403 while this
    is false. See `docs/operations.md`.
    """

    device: str | None = None
    """Torch device for both models. `None` auto-detects via
    `yolo11.select_device()` (cuda when visible, else cpu). Set it explicitly to
    pin a device or to force CPU on a box that has a GPU."""

    vram_total_mib: int = Field(default=8192, gt=0)
    vram_reserved_mib: int = Field(default=2048, ge=0)

    detector_model_id: str = "yolo11s.pt"
    detector_vram_mib: int = Field(default=432, gt=0)
    """Startup estimate for `ModelSpec.vram_mib`, replaced by the measured
    `capabilities().vram_mib` once `warmup()` has run (see
    `main.refresh_specs_from_capabilities`). A configured seed is unavoidable:
    `plan_residency()` has to decide whether a model fits *before* it is loaded,
    and nothing can measure it before then. The default is the figure Task 11
    measured on an RTX 4060 (`memory_reserved()` + CUDA-context overhead)."""

    vlm_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
    vlm_vram_mib: int = Field(default=2766, gt=0)
    """The same startup estimate for the VLM; Task 13's measured figure."""

    vlm_idle_unload_seconds: float = Field(default=600.0, gt=0)

    pose_model_id: str = "yolo11n-pose.pt"
    """The keypoint model behind `Capability.FALL_DETECTION`.

    `n` rather than the detector's `s`, deliberately — see
    `adapters/pose/yolo11_pose.py` for the argument: pose runs on people the detector
    already found and has a working geometry fallback, so its misses are recoverable
    where the detector's are not."""

    pose_vram_mib: int = Field(default=420, gt=0)
    """Startup estimate for `ModelSpec.vram_mib`, replaced by the measured
    `capabilities().vram_mib` once `warmup()` has run — the same seed-then-measure
    arrangement `detector_vram_mib` documents. Seeded near the detector's own figure
    because YOLO11n-pose is the same architecture family at a smaller width, and
    because most of that number is the shared CUDA context either way."""

    pose_idle_unload_seconds: float | None = None
    """Whether the pose model may be idle-evicted, and after how long. `None` means
    never.

    `None` by default, unlike the VLM's 600s, because the two are used on opposite
    schedules. A VLM fires on escalations — rare, bursty, and tolerant of a
    multi-second reload because one escalation waits rather than a camera. Pose runs
    on *every sampled frame of every fall-detection camera*, so it is idle only when
    those cameras are empty, and evicting it then means the reload lands exactly when
    someone walks into an empty room — the moment a fall becomes possible. Set a value
    here only on a box where the VRAM is genuinely contended."""

    pose_conf_threshold: float = Field(default=0.4, gt=0.0, lt=1.0)
    """Person-detection confidence inside the pose model's own single-stage pass.
    Distinct from `detector_conf_threshold`: this gates whether a *skeleton* is
    offered for attribution, not whether a person is tracked at all."""

    pose_imgsz: int = Field(default=640, gt=0)

    face_model_name: str = "buffalo_l"
    """The InsightFace bundle behind `Capability.PERSON_AUTHORIZATION` (SCRFD + ArcFace)."""

    face_vram_mib: int = Field(default=704, gt=0)
    """What `plan_residency()` reserves for the face pipeline before it has loaded.

    Measured on an RTX 4090, and the two figures differ by more than rounding:

    * **608 MiB** standalone — 312 placing the weights, 480 after warmup, and the rest
      allocated lazily by the first frame that actually contains a face.
    * **654 MiB** in the live engine, warming up alongside the detector.

    704 covers both. Over-reserving makes the planner cautious; under-reserving makes it
    admit a set that does not fit and discover it at the first busy frame, which is the
    failure `adapters/vram.py` exists to prevent.

    It is a planning input, not a reading — `/health` reports what the adapter measured,
    which is taken at the end of warmup and is therefore a floor.

    **0 on a box with no CUDA provider for ONNX Runtime**, where the pipeline falls back
    to CPU and costs no VRAM. That fallback keeps the capability working rather than
    fast: 125 ms per 1080p frame holding six faces, against 15 ms on the GPU. Install
    `onnxruntime-gpu` if face cameras matter — `version()` reports which provider is
    actually in use."""

    face_store_path: str = "./var/faces/faces.json"
    """Where enrolled people and their encrypted embeddings live.

    **Treat this file as a credential store.** Every embedding in it is AES-256-GCM
    ciphertext and useless without the key, but it is still the record of who is
    enrolled at this site. It holds no images."""

    face_encryption_key: str | None = None
    """Base64 AES key for the biometric store. **Required when any camera enables
    `person_authorization`, and the engine refuses to start without it** rather than
    writing biometric data in the clear.

    No default, deliberately: a default key is no key. Generate one with
    `python -m sentinel_ai.adapters.face.encrypted_store` and supply it the way the
    deployment supplies its other secrets. Rotating it makes every enrolled face
    undecryptable, so everybody must be re-enrolled — which is the honest consequence
    of the data being genuinely encrypted rather than obfuscated."""

    alert_register_capacity: int = Field(default=500, ge=1)
    """How many alerts the in-process register holds before evicting the oldest closed
    one (spec §17). Bounds memory on a camera with a stuck detector; it is not a
    curation limit. See `orchestrator/alerts.py` — the register is **volatile**, so an
    acknowledgement does not survive a restart."""

    alert_merge_window_seconds: float = Field(default=120.0, gt=0)
    """How long after its last sighting an alert still absorbs a recurrence.

    The knob that decides whether an operator sees an *incident* or a *category*: long
    enough that a person loitering in and out of a restricted area is one row, short
    enough that the same person tomorrow is a new one. Nothing auto-resolves, so
    without a window the first alert of a kind would absorb every later one forever."""

    alert_store_path: str | None = "./var/alerts/alerts.json"
    """Where operator triage state is written, so an acknowledgement survives a restart.

    Set to `null` to run the register purely in memory, which is what it did before this
    setting existed: alerts still work, and a restart shows every one of them as unseen
    again.

    **This is not the record of what happened.** That is the anomaly event published to
    RabbitMQ. This file records what a human did about it — which alerts were seen and
    which were closed — and that exists nowhere else."""

    alert_flush_interval_seconds: float = Field(default=5.0, gt=0)
    """How long a machine-driven change to the register may sit unwritten.

    Applies to alerts opening and occurrence counts rising, which happen as fast as the
    site is busy. **It does not apply to acknowledging or resolving**, which are flushed
    before the API answers, because an operator being told "done" about something still
    only in memory is the failure the store exists to prevent.

    Five seconds bounds what a `kill -9` costs to five seconds of counts, against twelve
    whole-set writes a minute on a busy site."""

    clip_retention_days: int = Field(default=1, ge=0)
    """How long an evidence clip is kept, in days, enforced by the object store.

    A bucket lifecycle rule rather than a sweeper in this process, because a sweeper
    deletes nothing while the engine is down — and an engine that was down for a week
    comes back to a week of clips it should have expired.

    **0 keeps clips forever** and removes the rule, which is how a deployment says
    "retention is handled elsewhere". Plenty of sites have a bucket policy of their own,
    and layering a second one on top of it is how evidence disappears a week before
    anybody expected it to.

    One day is a demo default, not a recommendation. A site with a retention obligation
    should set this to what that obligation says."""

    notify_clip_seconds: float = Field(default=3.0, ge=0)
    """How much footage a **notification** carries, in seconds. 0 disables the short clip.

    Distinct from the evidence clip's pre-roll and post-roll, and the two want opposite
    things. Evidence wants context and is watched later by somebody who chose to sit down
    with it. A notification is read on a phone at 3am by somebody deciding whether to walk
    down a corridor, and the useful length is however long it takes to see what happened.

    Cut from the start of the finished clip, so it is the pre-roll — the seconds *before*
    the event. A clip that opened on the fall would show a person already on the floor.

    Only used at or above `notify_clip_min_severity`; below it the notification carries
    the full clip, because there is no hurry and trimming would lose context for nothing.
    """

    notify_clip_min_severity: Severity = Severity.HIGH
    """The severity at which a notification gets the short clip instead of the full one.
    `high` by default, which includes `critical`."""

    source_max_fps: float = Field(default=0.0, ge=0)
    """Cap how many frames a second a live source hands to the pipeline. 0 = every frame.

    The codec forces the engine to *decode* every frame — an inter-coded frame is
    meaningless without its references — but nothing forces it to hand every one on. A
    30 fps camera feeding a pipeline that processes five frames a second spends the other
    twenty-five waking the event loop to overwrite a mailbox slot, and that loop is the
    resource this engine is short of
    ([ADR 17](../../docs/decisions.md#17-scale-by-processes-not-by-threads)).

    Measured on three 1080p cameras: 90 slot writes a second, ~85% discarded before
    anything looked at them. Setting this near what the pipeline actually achieves costs
    nothing real and gives the loop back to the frames that *are* processed.

    It does not reduce decode cost, and it is not a substitute for sharding across
    processes — that remains the fix when one process cannot keep up."""

    decode_hwaccel: str | None = None
    """Hardware decode device type - `cuda`, `qsv`, `drm`, `amf` - or unset for software.

    **Off by default, and measurement says leave it off unless CPU is what you are
    short of.** On an RTX 4090 at 1080p, CUDA decode alone runs at 801 fps and 0.26
    cores against software's 580 fps and 1.00 core. But a decoded frame still has to
    reach a numpy array in host memory, and pulling an NV12 surface off the GPU to
    convert it there costs more than decoding straight to YUV420P ever did: end to end,
    260 fps and 1.84 cores against 336 fps and 1.60. It buys camera count on a
    core-starved box and costs throughput everywhere else.

    A device type this FFmpeg build cannot open is a **startup error**, not a fallback.
    A deployment that asked for hardware decode and quietly got software is one whose
    capacity planning is wrong and whose logs do not say so. See
    `adapters/sources/hwaccel.py` and `docs/performance.md`."""

    rabbitmq_url: str = "amqp://sentinel:sentinel@localhost:5672/"
    rabbitmq_exchange: str = "sentinel.events"
    event_spool_dir: str = "./var/spool/events"
    broker_replay_interval_seconds: float = Field(default=30.0, gt=0)
    """How often `main.BrokerLink` re-drains the spool while the broker is up.
    Spec §9's disk buffer is only half a guarantee without something that replays
    it; this is the period of the component that does."""

    dead_letter_dir: str = "./var/spool/dead-letter"
    """Last resort for an event the publisher itself would not take (spec §9).
    Distinct from `event_spool_dir`, which holds validated payloads waiting for a
    broker that is merely down; this one holds events that failed for a reason
    replay cannot fix, so mixing them would strand the healthy spool."""

    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "sentinel"
    minio_secret_key: str = "sentinel123"
    minio_bucket: str = "sentinel-clips"
    minio_secure: bool = False

    clip_preroll_seconds: float = Field(default=3.0, ge=0)
    clip_postroll_seconds: float = Field(default=5.0, gt=0)

    detector_conf_threshold: float = Field(default=0.35, gt=0.0, lt=1.0)
    detector_iou_threshold: float = Field(default=0.45, gt=0.0, lt=1.0)
    detector_imgsz: int = Field(default=640, gt=0)

    vlm_queue_maxsize: int = Field(default=4, ge=1)
    vlm_timeout_seconds: float = Field(default=30.0, gt=0)
    vlm_quantization: Literal["nf4", "none"] = "nf4"
    """How the vision-language model's weights are stored. `nf4` (4-bit) or `none` (fp16).

    `nf4` is [ADR 1](../../docs/decisions.md)'s configuration and stays the default,
    because it is what makes this model fit beside a detector on an 8 GiB card.

    **Set `none` where VRAM is not the binding constraint.** NF4 stores weights in four
    bits and dequantises them on *every forward pass*, which is a good trade when memory
    is scarce and a poor one when it is not. Measured on an RTX 4090: **1.42 s per
    describe against 1.97 s**, for 7224 MiB instead of 2820 MiB. That 28% matters more
    than it looks — a describe holds the GIL in its per-token loop, so its duration is
    time every camera's frame loop is not running, and it is the largest single cause of
    dropped frames on a busy site."""

    vlm_max_new_tokens: int = Field(default=256, gt=0)
    vlm_global_concurrency: int = Field(default=1, ge=1)
    vlm_global_min_interval_seconds: float = Field(default=2.0, ge=0)

    detect_every_n_frames: int = Field(default=1, ge=1)
    source_realtime: bool = True
    rtsp_reconnect_initial_seconds: float = Field(default=1.0, gt=0)
    rtsp_reconnect_max_seconds: float = Field(default=30.0, gt=0)

    clip_temp_dir: str = "./var/clips"

    notifier_kind: NotifierKind = NotifierKind.LOGGING
    """Which welfare notifier the engine delivers through. **Logging by default.**

    Not `None`: there is no "no notifier" state. `LoggingNotifier` touches no
    network and needs no configuration, so a deployment that has thought about
    nothing still leaves a trail an operator can tail — and every routing decision
    the engine makes is observable somewhere rather than only in the absence of an
    alert nobody was expecting."""

    notifier_webhook_url: str | None = None
    """Where `notifier_kind=webhook` POSTs. Required by that kind and ignored by
    every other.

    **Treat this as a credential.** ntfy and Slack both put a per-recipient token
    in the URL path, which is why `WebhookNotifier` never logs it, never follows a
    redirect that could re-send it elsewhere, and why the composition root installs
    httpx's log redaction around it (see `main.build_notifier`)."""

    notifier_timeout_seconds: float = Field(default=20.0, gt=0)
    """Wall-clock ceiling on **one whole notification**, retries included — not the
    per-request HTTP timeout.

    The distinction matters, because the two numbers pull opposite ways.
    `WebhookNotifier` bounds each individual attempt at its own 5s and retries a
    transient failure up to three times with backoff, a documented worst case of
    18.0 seconds; this is the outer deadline `NotificationDispatcher` enforces
    around all of that, so it must sit *above* the adapter's worst case or it
    cancels the retries midway and turns every transient 429 into a lost note —
    lost with no dead-letter record, because the cancellation does not reach the
    adapter's own last-resort spool. Hence 20.0 rather than the 5.0 an operator
    reading "timeout" as "HTTP timeout" would reach for, and hence
    `main.build_notifier` **refusing to start** a webhook deployment whose value
    here does not clear `WebhookNotifier.retry_worst_case_seconds`.

    It exists at all because `Notifier` the port promises no bound of its own: the
    webhook adapter happens to bound itself, a future adapter need not, and neither
    may park the single notification worker forever."""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
