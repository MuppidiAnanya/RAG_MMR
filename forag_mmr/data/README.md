# Data

The dataset is downloaded dynamically through `datasets.load_dataset("forag/webglm_oe")`; no dataset files are committed here. The `doc` text is retained verbatim, including numbered evidence labels. With `PARSE_NUMBERED_DOCS=true`, labels such as `[1]` remain in their individual retrieval units.
