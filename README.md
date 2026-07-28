# Bruker MRI Viewer

A napari-based browser for Bruker ParaVision MRI study folders.

## Features

- **Study browser** — left-panel tree showing Patient → Study → Experiment → Proc hierarchy. Detects Layout A (patient folders), Layout B (study folders directly in root), and Layout C (single study) automatically.
- **Direct napari loading** — double-click any proc row to load it as a napari Image layer. The same proc double-clicked again replaces the layer in-place.
- **Frame Group selector** — right panel shows one dropdown per selectable frame-group axis (echoes, T2/ISA parametric maps, diffusion directions). Switching updates the displayed slice without re-reading from disk.
- **Window / Level controls** — manual width/level spinboxes, standard presets (Brain, Soft Tissue, Lung, Bone), and Auto Contrast.
- **Correct voxel scale** — reads `dataset.resolution` from brukerapi; slice thickness and in-plane voxel size are passed to napari so measurements and colocalisation are physically accurate.
- **Display upsampling** — small acquisition matrices (e.g. 48×48 MSME) are upsampled to 256 px using cubic spline interpolation, matching ParaVision's default display.

## Running

```bash
uv run python main.py
```

Then click **Open Patient Folder…** in the left panel and select the directory containing your Bruker study folders.

## Data layout supported

```
patients/
  RA_2022_07_20.f21/       ← one folder per animal/timepoint
    subject
    9/
      acqp
      pdata/
        1/  2dseq          ← raw reconstruction
        2/  2dseq          ← derived map (T2, ADC, …)
```

## For analysis / scripting

Load data without the GUI using `load_2dseq` directly:

```python
from bruker_browser.loader import load_2dseq

result = load_2dseq(
    "patients/real_data/RA_2022_07_20.f21",
    exp_id="9",
    proc_id="1",
    zoom_to=0,          # disable interpolation for quantitative analysis
)

# result.data  — numpy array (n_slices, n_echoes, H, W) in napari axis order
# result.scale — voxel size in mm per axis
# result.frame_groups — FrameGroupInfo list (echo times, ISA param names, …)
```

> **Important:** always pass `zoom_to=0` when loading data for curve fitting or
> statistics.  The default `zoom_to=256` is for display only — fitting on
> interpolated voxels produces incorrect parameter estimates.

## Dependencies

- [napari](https://napari.org/)
- [brukerapi](https://github.com/isi-nmr/brukerapi)
- [scipy](https://scipy.org/) (for display upsampling)
- [qtpy](https://github.com/spyder-ide/qtpy)
