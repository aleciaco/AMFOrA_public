# `amfora.core.analysis`

Per-sherd and batch analysis. `analyze_single_sherd` is the main entry point; `full_analysis` is the batch wrapper that produces a CSV / DataFrame from a folder of images. The other functions are component helpers and are exposed mostly so you can pull them out of the pipeline when you need finer control.

Source: [`src/amfora/core/analysis.py`](https://github.com/aleciaco/AMFOrA_public/blob/main/src/amfora/core/analysis.py).

## Top-level pipeline

```{eval-rst}
.. autofunction:: amfora.core.analysis.analyze_single_sherd
.. autofunction:: amfora.core.analysis.full_analysis
```

## Per-stage helpers

```{eval-rst}
.. autofunction:: amfora.core.analysis.size_count_summary_single
.. autofunction:: amfora.core.analysis.size_count_summary
.. autofunction:: amfora.core.analysis.void_counter
.. autofunction:: amfora.core.analysis.contour_counter
.. autofunction:: amfora.core.analysis.sacredsquare
.. autofunction:: amfora.core.analysis.inclusion_colors
.. autofunction:: amfora.core.analysis.inclusion_colors_from_contours
.. autofunction:: amfora.core.analysis.inclusion_orientation
.. autofunction:: amfora.core.analysis.inclusion_orientation2
.. autofunction:: amfora.core.analysis.sherd_color_analysis
.. autofunction:: amfora.core.analysis.sherd_color_summary
.. autofunction:: amfora.core.analysis.extract_core_periphery_colors
.. autofunction:: amfora.core.analysis.analyze_inclusion_angularity
.. autofunction:: amfora.core.analysis.analyze_orientation_for_pca
.. autofunction:: amfora.core.analysis.analyze_manufacturing_technique
```
