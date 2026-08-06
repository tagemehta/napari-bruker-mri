# Bruker MRI Viewer

A napari-based browser for Bruker ParaVision MRI study folders.

## Features

- **Study browser** — left-panel tree showing Patient → Study → Experiment → Proc hierarchy. Detects Layout A (patient folders), Layout B (study folders directly in root), and Layout C (single study) automatically.
- **Direct napari loading** — double-click any proc row to load it as a napari Image layer. The same proc double-clicked again replaces the layer in-place.
- **Frame Group selector** — right panel shows one dropdown per selectable frame-group axis (echoes, T2/ISA parametric maps, diffusion directions). Switching updates the displayed slice without re-reading from disk.
- **Window / Level controls** — manual width/level spinboxes, standard presets (Brain, Soft Tissue, Lung, Bone), and Auto Contrast.
- **Correct voxel scale** — reads `dataset.resolution` from brukerapi; slice thickness and in-plane voxel size are passed to napari so measurements and colocalisation are physically accurate.
- **Display upsampling** — small acquisition matrices (e.g. 48×48 MSME) are upsampled to 256 px using cubic spline interpolation, matching ParaVision's default display.
- **Parameter map browser** — right-panel widget lists any T2/ADC maps (and ROI sets) already generated for the loaded study and loads them into the viewer on click. See [Per-disc T2/ADC + ROI analysis](#per-disc-t2adc--roi-analysis) below.

## Running

```bash
uv run python main.py
```

Then click **Open Patient Folder…** in the left panel and select the directory containing your Bruker study folders.

## Which file do I open? (task → script)

Every script in this repo can be run standalone with `uv run python <name>.py`.
Filenames aren't always self-explanatory, so here's a map from "what am I
trying to do" to the file that does it:

| I want to...                                                          | Open this file                     |
|------------------------------------------------------------------------|-------------------------------------|
| Browse raw scans / load a study into a viewer                          | `main.py`                           |
| Generate T2 + ADC maps for one disc, draw ROIs, save stats              | `analyze_disc.py`                   |
| See a study's already-saved maps/ROIs without regenerating anything     | *(automatic)* — `main.py`'s "Parameter Maps" dock, backed by `parameter_browser.py` |
| Sanity-check a T2 map fit against ParaVision's own map                  | `check_t2_map.py`                   |
| Sanity-check an ADC map fit against ParaVision's own map                 | `check_adc_map.py`                  |
| Check per-direction diffusion signal / ADC quality for one scan         | `check_diffusion_directions.py`     |
| Visually inspect curve-fit quality for ROIs you've already drawn       | `inspect_roi_fit.py`                |
| Figure out which raw study folder = which rat ID                        | `index_bruker_studies.py` (writes `patients/real_data/INDEX.md`) |
| Figure out which scans are repeats vs. genuinely different disc locations (Top vs Bottom) | `inspect_scan_positions.py` |
| Visually confirm Top vs Bottom by projecting a scan onto the sagittal localizer | `plot_disc_positions_on_sagittal.py` |
| Attach each ROI's irradiation Group / timepoint to its stats            | `join_roi_stats_grouping.py`        |
| Filter out noisy/poorly-fit pixels and compare irradiated vs. control trends, with t-tests | `snr_filter_and_compare.py` |
| Export raw Bruker scans to NIfTI (e.g. for 3D Slicer)                    | `export_nifti_studies.py`           |
| Figure out what an exported NIfTI file actually is                      | `label_nifti_exports.py`            |

All of these (except `main.py`) follow the same pattern: open the file, edit
the constants at the top (study path, experiment ID, etc.), then
`uv run python <file>.py`. None of them take command-line arguments.

### Typical order of operations for a new disc/animal

1. `index_bruker_studies.py` — confirm which raw folder corresponds to which rat.
2. `inspect_scan_positions.py` + `plot_disc_positions_on_sagittal.py` — confirm Top vs. Bottom disc and which experiment IDs are repeats.
3. `analyze_disc.py` — fit T2/ADC maps and draw ROIs for that disc.
4. `join_roi_stats_grouping.py` — attach cohort/timepoint metadata to the new ROI stats.
5. `snr_filter_and_compare.py` — re-run once new discs are added, to refresh the filtered stats and trend/t-test tables across the whole cohort.

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

## Per-disc T2/ADC + ROI analysis

For each disc (e.g. two per animal), generate T2 and ADC parameter maps,
draw ROIs (NP, AF, and as many named variants as you like — e.g. "NP sample
1", "AF sample 2"), and extract mean/min/max/SD per ROI. This is two
separate, deliberately simple pieces:

### 1. Generate + annotate — `analyze_disc.py`

T2 and ADC come from two different acquisitions/exp_ids even for "the same
disc" — edit the constants at the top of the script:

```python
STUDY_PATH = "patients/real_data/RA_2022_07_20.f22"
STUDY_NAME = "Rat Spine 690731"
EXP_ID_T2 = "6"
EXP_ID_ADC = "7"
```

Then run:

```bash
uv run python analyze_disc.py
```

This will:
1. **Always fit a fresh** T2 and ADC map from the raw proc 1 data — at
   native resolution, never upsampled — and save them to `parameter_maps/`.
   This never uses ParaVision's own proc-2 map as the analysis source, so
   results don't depend on whether PV happened to already compute one for
   this experiment.
2. Print a **smoke check** against PV's own map, if one exists (a quick
   agreement stat, e.g. median absolute difference) — reference only, never
   used for ROI stats.
3. Open napari with each map's own signal-intensity image (proc 1's first
   echo for T2, lowest-b ~b0 DWI frame for ADC — always matching the freshly
   fitted map's own grid) plus a **ROI Annotator** dock widget for drawing
   ROIs (New -> Polygon/Circle -> draw -> auto-named, no fixed slots). If
   PV's map was found, it's also added as a hidden `"... — PV (smoke check)"`
   layer you can toggle on to compare visually.
4. Wait for you to draw ROIs (polygon tool) on **`T2 ROIs`** and
   **`ADC ROIs`** — draw the same conceptual ROI (e.g. "NP") on both if you
   want stats from both maps. There's no fixed ROI count.
5. When you **close the napari window**, prompt you in the terminal for a
   name (e.g. `NP`, `AF sample 1`) and tissue tag (`NP`/`AF`) for each newly
   drawn shape.
6. Compute mean/min/max/SD per ROI against its own **fitted** map and
   write/update rows in `parameter_maps/roi_stats.csv`.

Re-running the script for the same disc reloads any previously saved ROIs
so you can review or add more, and updates that disc's CSV rows in place
(no duplicate rows).

> **Why ROIs are drawn on the anatomy image, not the map, and never shared
> between T2 and ADC:** T2 and ADC acquisitions for the same disc have
> different FOV, matrix size, and in-plane center — verified against real
> data. Each anatomy image shares its own map's exact native pixel grid
> (both come from the same acquisition), so ROIs always rasterize straight
> onto the array they were drawn against — no coordinate-registration code
> needed anywhere. See `analysis/rois.py` for details.

### 2. Browse saved results in the live app — `parameter_browser.py`

Already wired into `main.py`. Load a study as usual in the tree browser;
the **Parameter Maps** dock (right panel) lists everything `analyze_disc.py`
has generated for that study (e.g. `exp 6 — T2 (+ ROIs)`). Select an entry
and click **Load Selected** to add that map — and its ROIs, if saved — as
layers. This widget is read-only: it never generates or draws anything
itself, so opening a study never triggers a slow fit.

### Storage layout

```
parameter_maps/                                  ← gitignored, reproducible
  {study_name}_exp{exp_id}_t2.npy / .json
  {study_name}_exp{exp_id}_adc.npy / .json
  rois/{study_name}_exp{exp_id}_{t2,adc}.json
  roi_stats.csv                                   ← one row per disc/ROI/map
```

### Other inspection scripts

- `check_t2_map.py` / `check_adc_map.py` — compare PV's own map against a
  freshly fitted one (difference, R², bias) for one study/experiment.
- `check_diffusion_directions.py` — per-direction ADC + SNR diagnostics for
  one diffusion acquisition (useful for checking whether an ADC finding is
  real biology or a fitting/noise artifact).

All three follow the same pattern: edit `STUDY_PATH`/`EXP_ID`/`STUDY_NAME`
constants at the top, then `uv run python <script>.py`.

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

### How frame groups are represented in raw data

A single Bruker `2dseq` file can hold more than one 2-D image per slice —
e.g. one image per echo time (multi-echo T2), one per b-value/direction
(diffusion), or one per fitted parameter (a T2 map's `T2`, `A`, `bias`
images all live in the same file). Bruker calls each of these extra axes a
**Frame Group (FG)**; ParaVision's own UI is what lets you scroll through
them one at a time.

**On disk**, brukerapi reports every axis as one flat list — spatial axes and
frame-group axes mixed together — with a parallel `dim_type` list telling
you which is which. Concrete example, a real 16-echo T2 acquisition with
8 slices (`patients/real_data/RA_2022_07_18.f01`, exp 5, proc 1):

```
raw shape (brukerapi):  (150,      150,      16,        8)
raw dim_type:           ('spatial', 'spatial', 'FG_ECHO', 'FG_SLICE')
                              X         Y         echo      slice
```

`'spatial'` is always exactly 2 axes (X, Y), always first in brukerapi's raw
layout. `FG_SLICE` (and equivalents `FG_SLICE_ORIENT`/`SLICE`/`FRAME`) is the
physical slice/Z axis; everything else (`FG_ECHO`, `FG_ISA`, `FG_DIFFUSION`,
`FG_MOVIE`, …) is a genuinely selectable axis — different echo times,
different fitted parameters, different diffusion directions/b-values.

`load_2dseq` transposes this into napari's `(..., Z, Y, X)` convention.
For the shape above, the result is:

```
napari shape:  (8,          16,       150, 150)
napari axes:   (FG_SLICE,   FG_ECHO,  Y,   X)
```

**Why slice ends up outermost and echo second — not the other way round:**
`_reorder_for_napari` (`bruker_browser/loader.py`) does one purely mechanical
thing: it reverses brukerapi's *entire* raw axis order end-to-end.

```python
new_order = fg_idx[::-1] + spatial_idx[::-1]   # fg_idx=[2,3], spatial_idx=[0,1]
reordered = np.transpose(data, axes=new_order)  # -> [3, 2, 1, 0]
```

Raw axis order was `[X, Y, FG_ECHO, FG_SLICE]` (indices `0,1,2,3`); reversed,
that's `[3,2,1,0]` = `[FG_SLICE, FG_ECHO, Y, X]`. There's no special-casing
of slice vs. echo in this step — it's whatever order brukerapi happened to
put the raw axes in, reversed. Bruker's own convention puts `FG_SLICE` last
among the non-spatial axes in its raw layout, so it becomes first (outermost)
after reversal. An acquisition where brukerapi reported `FG_SLICE` *before*
`FG_ECHO` in the raw order would come out the other way round in napari —
the code only guarantees spatial axes end up last, not any particular order
among the FG axes.

Separately — and deliberately, unlike the reversal above — `FG_SLICE` is
excluded from `result.frame_groups` (only `FG_ECHO` appears there) by
`_extract_frame_groups`, so the UI always presents slice as a plain
scrollable image stack, not a labeled dropdown, regardless of where the
reversal happened to place it in the array.

So to pull out, say, the 3rd echo of a multi-echo stack, index
`result.data` using the `.axis` position reported on the matching
`FrameGroupInfo` in `result.frame_groups` rather than assuming a fixed axis
number — the position (and even the presence) of each FG axis varies by
acquisition type (see `AGENTS.md`'s "Two T2-map layouts in this dataset" for
a second real example where the axis order differs). `analysis/maps.py` and
`analysis/fitting.py` show the canonical pattern for consuming
`frame_groups` when fitting T2/ADC maps from raw multi-echo/multi-b-value
data.

## Dependencies

- [napari](https://napari.org/)
- [brukerapi](https://github.com/isi-nmr/brukerapi)
- [scipy](https://scipy.org/) (for display upsampling and T2/ADC fitting)
- [qtpy](https://github.com/spyder-ide/qtpy)
- [pandas](https://pandas.pydata.org/) (for the ROI stats CSV)
