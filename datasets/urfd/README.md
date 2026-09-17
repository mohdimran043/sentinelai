# URFall Detection Dataset (fetched, not committed)

`fetch.sh` downloads 70 clips (~90 MB) from the University of Rzeszow. The `.mp4` files
are deliberately **not** in this repository — they are somebody else's data, and git is
not where a video dataset belongs.

```bash
cd datasets/urfd && ./fetch.sh
cd ../../ai-engine && python -m sentinel_ai.validation.falls
```

**What it is.** 30 clips each containing one fall, and 40 of activities of daily living
containing none — sitting, bending, lying down deliberately, picking things up. The ADL
half is the important one: a detector that fires on somebody sitting heavily is a
detector an operator switches off.

`cam0` is the camera parallel to the floor, which is the view a wall-mounted security
camera has, and is the only one this harness scores. `cam1` is ceiling-mounted; nothing
in this system is designed for an overhead view.

**Credit.** Bogdan Kwolek and Michal Kepski, *Human fall detection on embedded platform
using depth maps and wireless accelerometer*, Computer Methods and Programs in
Biomedicine, 2014. <http://fenix.ur.edu.pl/~mkepski/ds/uf.html>

**What this engine scores on it** is in
[docs/performance.md](../../docs/performance.md#fall-detection-urfall-70-clips), and it
does not flatter the detector.
