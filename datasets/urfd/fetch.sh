#!/usr/bin/env bash
# URFall Detection Dataset (Kwolek & Kepski, University of Rzeszow) — cam0 is the
# camera parallel to the floor, i.e. the view a wall-mounted security camera has.
# 30 fall clips (one fall each) and 40 activity-of-daily-living clips (no falls).
set -u
base="http://fenix.ur.edu.pl/~mkepski/ds/data"
for i in $(seq -w 1 30); do
  [ -s "fall-$i-cam0.mp4" ] || curl -sfL -o "fall-$i-cam0.mp4" "$base/fall-$i-cam0.mp4" || echo "MISS fall-$i"
done
for i in $(seq -w 1 40); do
  [ -s "adl-$i-cam0.mp4" ] || curl -sfL -o "adl-$i-cam0.mp4" "$base/adl-$i-cam0.mp4" || echo "MISS adl-$i"
done
echo "done: $(ls fall-*.mp4 2>/dev/null | wc -l) falls, $(ls adl-*.mp4 2>/dev/null | wc -l) adls"
