# Obtain the data

This repository contains **no EEG recordings or subject-level tabular data**.
1. Open [OpenNeuro ds007169, version 1.0.5](https://openneuro.org/datasets/ds007169/versions/1.0.5).
2. Use its Download options to retrieve the original `sourcedata/xdf/` file
   and the matching `sub-XXX/eeg/` directory. A full dataset download also
   works; select the recorded version when reproducing the example.
3. Preserve the directory layout below under `data/ds007169/`, then check
   that `001_nback.xdf` and the relevant TSVs exist and have downloaded
   contents. If using a DataLad or Git annex checkout, retrieve the actual
   large-file content before running the code.

Make sure large-file content,
including the original XDF under `sourcedata/xdf/`, is actually present rather
than just a Git annex pointer or symlink. The analysis needs the original XDF,
not solely the converted BrainVision EEG.

Place the dataset at `data/ds007169/` (ignored by Git), keeping the public layout:

```text
data/ds007169/
├── sourcedata/xdf/001_nback.xdf
└── sub-001/eeg/
    ├── sub-001_task-nback_events.tsv
    ├── sub-001_task-nback_channels.tsv
    ├── sub-001_task-nback_eeg.vhdr  # optional for scale comparison
    └── sub-001_task-nback_eeg.eeg   # optional for scale comparison
```

For another released participant, substitute its numeric ID. The runner also
accepts the original notebook's flattened `data/sub_001/` layout with these
files directly in that directory. Point `--data-root` to the **parent** of
`sub_001`, or to the root of the OpenNeuro dataset. Do not add the downloaded
dataset to this code repository. OpenNeuro's dataset and its provenance/terms
are separate from the MIT license for this analysis code.
