# Sample footage

This directory is gitignored except for this file, `download_sample.sh`, and
`.gitkeep` — the footage itself is never committed (spec §8 / Phase 1B §8).

## Source

**CUHK Avenue Dataset** — Lu, C., Shi, J., & Jia, J. (2013). *Abnormal Event
Detection at 150 FPS in Matlab.* ICCV 2013.
http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html

## Licence — verified before the URL was hardcoded

The dataset's own page was fetched and read directly: it publishes **no
separate licence file**, and states no explicit terms for commercial use,
research use, or redistribution on the page itself.

What corroborates an academic-research-only grant, checked independently:

- The [Anomalib datamodule documentation](https://anomalib.readthedocs.io/en/v2.1.0/markdown/guides/reference/data/datamodules/video/avenue.html)
  — a third-party academic redistribution/loader that itself cites the
  dataset — states verbatim: *"The CUHK Avenue dataset is released for
  academic research only."* No commercial-use grant is published anywhere
  that was found.
- The paper accompanying the dataset (cited above) is the expected citation
  if this footage, or a description of it, appears in any publication.
- The download URL used by `download_sample.sh`
  (`http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/Avenue_Dataset.zip`)
  was confirmed to resolve directly against the CUHK host — not a
  third-party mirror — via `curl -I`, returning `HTTP/1.1 200 OK`,
  `Content-Type: application/zip`, `Content-Length: 813227845`.

SentinelAI's use here is non-commercial: one clip, used as local
development/demo footage for a research project, looped through a private
mediamtx instance and never redistributed or re-published. That fits inside
the academic-research grant described above.

**Do not point this script at a different dataset without repeating this
verification.** A dataset without a published academic-research grant, or
one sourced from an indexed feed of unsecured private cameras, is out of
scope regardless of convenience.

## Ethical constraint (binding, from spec §5)

Only deliberately-public, licensed research footage is used for
verification. Nothing that indexes unsecured private cameras (e.g. via
Shodan-style scanners) is ever a source, for testing or otherwise.

## What the script does

`download_sample.sh` downloads the full Avenue archive, extracts a single
testing clip (`testing_videos/01.avi`), re-encodes it to H.264/yuv420p MP4,
and deletes the archive and everything else extracted from it. Output:
`datasets/avenue/avenue_01.mp4`.

Run once, from the repo root or `ai-engine/`:

    bash datasets/download_sample.sh

Requires `curl`, `unzip`, and `ffmpeg` on `PATH`.
