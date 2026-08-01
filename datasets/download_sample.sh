#!/usr/bin/env bash
# Downloads one CUHK Avenue test clip for the Phase 1B GPU demo.
#
# Source: CUHK Avenue Dataset — Lu, C., Shi, J., & Jia, J. (2013). "Abnormal
# Event Detection at 150 FPS in Matlab." ICCV 2013.
# http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html
#
# Licence: see datasets/README.md for the full verification trail. Summary:
# the host page publishes no separate licence file, but the Anomalib
# datamodule documentation states verbatim "The CUHK Avenue dataset is
# released for academic research only." This script's use — local dev/demo
# footage, never redistributed — fits inside that grant. Cite the ICCV 2013
# paper above if this footage or a description of it appears in any
# publication.
#
# Ethical constraint (spec §5): deliberately-public, licensed research
# footage only — never a feed that indexes an unsecured private camera.
# Do not point this script at a different dataset without repeating this
# verification in datasets/README.md.

set -euo pipefail

DATASET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${DATASET_DIR}/avenue"
ZIP_URL="http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/Avenue_Dataset.zip"
ZIP_PATH="${WORK_DIR}/Avenue_Dataset.zip"
OUTPUT_MP4="${WORK_DIR}/avenue_01.mp4"

if [[ -f "${OUTPUT_MP4}" ]]; then
  echo "Already present: ${OUTPUT_MP4}"
  exit 0
fi

mkdir -p "${WORK_DIR}"

echo "Downloading Avenue Dataset (~780 MiB; the full 16-training/21-testing"
echo "archive — only one testing clip is kept) from CUHK..."
curl -fL --retry 3 -o "${ZIP_PATH}" "${ZIP_URL}"

echo "Extracting the first testing clip..."
UNZIP_DIR="${WORK_DIR}/extracted"
mkdir -p "${UNZIP_DIR}"
unzip -o "${ZIP_PATH}" -d "${UNZIP_DIR}" >/dev/null

SOURCE_AVI="$(find "${UNZIP_DIR}" -ipath "*testing_videos*" -iname "01.avi" | head -n1)"
if [[ -z "${SOURCE_AVI}" ]]; then
  echo "Could not locate testing video 01 inside the archive." >&2
  echo "Inspect ${UNZIP_DIR} and adjust the find pattern above." >&2
  exit 1
fi

echo "Re-encoding to H.264/yuv420p for the RTSP/remux pipeline (the archive"
echo "ships Xvid-encoded AVI, which mediamtx/RTSP publishing does not want)..."
# -g/-keyint_min/-sc_threshold force a strict ~2s keyframe interval (25fps source ->
# -g 50). Without this, libx264's default keyframe spacing (~250 frames, ~10s here)
# is longer than a typical clip_preroll_seconds + clip_postroll_seconds window
# (spec §5.5's default 3-5s pre-roll + 5s post-roll): `MinioClipWriter` can only start
# muxing a clip from a keyframe (`PreRollBuffer.flush()`'s own contract), so an
# escalation's whole recording window can land entirely between two keyframes and
# produce no clip at all — verified against this exact script's own output during
# Task 14's manual demo, not a hypothetical. Real IP cameras almost always use a
# short (1-2s) keyframe interval, so this matches production, not just working
# around the demo.
ffmpeg -y -i "${SOURCE_AVI}" -c:v libx264 -profile:v baseline -pix_fmt yuv420p \
  -g 50 -keyint_min 50 -sc_threshold 0 \
  -movflags +faststart "${OUTPUT_MP4}"

echo "Cleaning up the archive (kept: ${OUTPUT_MP4})..."
rm -rf "${ZIP_PATH}" "${UNZIP_DIR}"

echo "Done: ${OUTPUT_MP4}"
