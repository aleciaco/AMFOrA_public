# `amfora.core.detection`

Image preprocessing, sherd masking, and the two feature detectors (blob and contour). This module also contains the paste-anchored MAD-scaled pop gate, the watershed cluster-recovery and multigrain-split passes, and the edge-band rejection helpers.

The full source lives in [`src/amfora/core/detection.py`](https://github.com/aleciaco/AMFOrA_public/blob/main/src/amfora/core/detection.py). Function docstrings are pulled directly from there into the listing below.

```{eval-rst}
.. autofunction:: amfora.core.detection.sherd_mask
.. autofunction:: amfora.core.detection.full_image_mask
.. autofunction:: amfora.core.detection.apply_mask
.. autofunction:: amfora.core.detection.clahe_enhance
.. autofunction:: amfora.core.detection.setup_robust_blob_params
.. autofunction:: amfora.core.detection.sherd_blobs
.. autofunction:: amfora.core.detection.contour_detection
.. autofunction:: amfora.core.detection.detect_multiple_sherds
.. autofunction:: amfora.core.detection.split_multi_sherd_scan
.. autofunction:: amfora.core.detection.prepare_multi_sherd_directory
.. autofunction:: amfora.core.detection.super_zorro_cv
```
