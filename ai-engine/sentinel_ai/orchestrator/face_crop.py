"""Keeping the enrolment photograph, at a size a roster can load (spec §9, §12).

This module used to crop to the detector's face box and keep only that. It now keeps the
photograph as uploaded, because the crop answered the wrong question: an operator
enrolling somebody needs to see *the picture they chose* — is it the right person, is it
a usable photo, is it the one they meant — and a 112-pixel face shows none of that.

What is kept is still bounded and still protected. The image is re-encoded and downscaled
if it is large, because a roster of 4000-pixel phone photographs is a roster nobody can
load; it is sealed with the same AES-256-GCM key as the embedding; it is deleted with the
person; and it is never logged.

**This is a deliberate reading of §12, not an oversight of it.** The strictest reading —
keep no biometric images at all — is what this file used to implement, and the cost was
a face-recognition roster nobody could eyeball, in which a wrong enrolment goes unnoticed
until it accuses somebody. Storing the subject's own photograph, encrypted and deletable,
buys the ability to check that. It does not license storing anything else.
"""

from __future__ import annotations

import io
import logging

__all__ = ["JPEG_QUALITY", "MAX_EDGE", "reference_image"]

logger = logging.getLogger(__name__)

MAX_EDGE = 1024
"""Longest side of the stored photograph, in pixels.

Large enough to recognise a face at a glance and to judge whether the photo was worth
enrolling; small enough that a hundred of them are a few tens of megabytes rather than
a few hundred. A modern phone photo is four times this on its long edge and carries
nothing extra that a roster needs.
"""

JPEG_QUALITY = 82


def reference_image(payload: bytes) -> bytes | None:
    """The uploaded photograph, re-encoded as a bounded JPEG. `None` if unreadable.

    Re-encoded rather than stored verbatim, for two reasons worth separating:

    * **Size.** Bounding the long edge is what keeps a roster loadable.
    * **Metadata.** A camera JPEG carries EXIF, and EXIF routinely carries GPS
      coordinates and a capture time. Enrolling somebody should not quietly file where
      and when their photograph was taken. Re-encoding through a decoded pixel buffer
      drops all of it.

    `None` rather than raising: an enrolment whose embedding succeeded has enrolled the
    person, and failing the whole request because a thumbnail could not be made would
    trade recognition for a picture.
    """
    try:
        from PIL import Image

        with Image.open(io.BytesIO(payload)) as handle:
            picture = handle.convert("RGB")
        picture.thumbnail((MAX_EDGE, MAX_EDGE))
        buffer = io.BytesIO()
        picture.save(buffer, format="JPEG", quality=JPEG_QUALITY)
        return buffer.getvalue()
    except Exception:
        logger.debug("could not re-encode an enrolment photograph", exc_info=True)
        return None
