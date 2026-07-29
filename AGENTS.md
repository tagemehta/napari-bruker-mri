# AGENTS.md — project notes for AI agents

## Running the app

```
uv run python main.py
```

## Verification commands

```bash
# Syntax-check all Python sources
uv run python -c "
import ast, pathlib
for f in pathlib.Path('.').rglob('*.py'):
    if '.venv' not in str(f) and '__pycache__' not in str(f):
        ast.parse(f.read_text(), filename=str(f))
        print('OK', f)
"

# Headless data-loading smoke test (no display required)
uv run python -c "
from bruker_browser.loader import load_2dseq
from bruker_browser.tile_viewer import _initial_display
from pathlib import Path

cases = [
    ('patients/real_data/RA_2022_07_18.f01', '5', '1'),   # MSME echo stack, Layout A
    ('patients/real_data/RA_2022_07_18.f01', '5', '2'),   # T2 map, Layout A multi-slice
    ('patients/real_data/RA_2022_07_20.f21', '9', '1'),   # MSME echo stack, Layout B
    ('patients/real_data/RA_2022_07_20.f21', '9', '2'),   # T2 map, Layout B single-slice
]
for study_path, exp_id, proc_id in cases:
    r = load_2dseq(Path(study_path), exp_id, proc_id, study_name='test')
    d, s = _initial_display(r.data, r.scale, r.frame_groups)
    print(f'{Path(study_path).name} exp={exp_id} proc={proc_id}: '
          f'full={r.data.shape} display={d.shape} scale={tuple(round(x,3) for x in s)} '
          f'fg={[(fg.name, fg.labels[:2]) for fg in r.frame_groups]}')
"
```

## Architecture

```
main.py                      Entry point.  Creates viewer, docks tree + FG selector.
windowing.py                 Right dock: FrameGroupWidget — one dropdown per FG axis.
bruker_browser/
  __init__.py
  models.py                  Pure data model: PatientNode → StudyNode → ExpNode → ProcNode.
                             No I/O beyond reading JCAMP-DX parameter files.
  loader.py                  load_2dseq() — all brukerapi interaction lives here.
                             Returns LoadResult (array, scale, layer name, FG descriptors).
  tile_viewer.py             load_into_viewer() — bridges LoadResult → napari layer.
                             _initial_display() extracts the first FG frame for display.
  tree_widget.py             BrukerTreeWidget — Qt tree populated from models.py.
                             Emits experiment_activated(study_path, exp_id, proc_id, study_name).
```

## Key design decisions

### Axis ordering
brukerapi returns `(X, Y[, FGs...])`.  `_reorder_for_napari` transposes to
napari's `([FGs...,] Z, Y, X)` convention.  `FrameGroupInfo.axis` records
the position of each selectable FG in the **napari-ordered** array.

### FG_SLICE vs selectable FGs
`FG_SLICE` is treated as the Z/slice axis — it is never listed in
`frame_groups` and always remains stacked in the displayed array.  Only echoes
(`FG_ECHO`), parametric-map types (`FG_ISA`), diffusion directions
(`FG_MOVIE`/`FG_DIFFUSION`), etc. appear in `frame_groups`.

### Two T2-map layouts in this dataset
- **Layout A** (early studies, e.g. `RA_2022_07_18`): brukerapi gives
  `(H, W, FG_ISA, FG_SLICE)` → napari `(n_slices, n_params, H, W)`.
  `FG_ISA` is at axis 1; `FG_SLICE` (axis 0) stays stacked.
- **Layout B** (later studies, e.g. `RA_2022_07_20` onward): brukerapi gives
  `(H, W, frame, FG_ISA, FG_SLICE)` with size-1 `frame` and size-1 `FG_SLICE`.
  After squeeze: `(n_params, H, W)` — single-slice parametric map.

### Display upsampling
`_zoom_spatial(zoom_to=256)` upsamples any spatial axis below 256 px using
cubic spline interpolation (`scipy.ndimage.zoom(order=3)`), matching
ParaVision's default display.  The `scale` is adjusted proportionally so
physical mm coordinates remain correct.  **Always pass `zoom_to=0` when
loading data for fitting** — never fit on interpolated voxels.

### Full array stored in layer metadata
`load_into_viewer` stores the complete pre-display array in
`layer.metadata["_full_data"]` alongside `_full_scale` and `_frame_groups`.
`WindowLevelWidget._apply_fg_selection` re-indexes this array in-place when
the user changes a Frame Group dropdown — no disk re-read required.

### Scale extraction
`dataset.resolution` (brukerapi property) gives `[dy, dx[, dz]]` in mm.
`dataset.get_value()` does **not** exist in this version of brukerapi — do not
use it.

### JCAMP-DX parsing
`_read_jcampdx_value` in `models.py` handles both inline and two-line array
forms.  `_parse_fg_elem_comment_from_file` uses regex to extract `<...>`
values across line-wrapped multi-line strings (PV wraps at 65 chars).

## Data layout on disk

```
patients/real_data/
  RA_2022_07_20.f21/          ← Study folder (Layout B: one per animal/timepoint)
    subject                   ← JCAMP-DX: subject name, study name, date
    9/                        ← Experiment folder (numeric)
      acqp                    ← Acquisition parameters (PULPROG, ACQ_scan_name, …)
      method                  ← Method parameters (PVM_Matrix, PVM_Fov, …)
      pdata/
        1/                    ← Proc 1: raw reconstruction
          2dseq               ← Binary image data
          visu_pars           ← Display parameters (VisuCoreSize, VisuFGOrderDesc, …)
          reco                ← Reconstruction parameters
        2/                    ← Proc 2: derived map (T2, ADC, …)
          2dseq
          visu_pars
```

## T2 map fitting (`analysis/`)

`analysis/maps.py` has two separate entry points, neither falling back to
the other: `get_pv_t2_map` extracts PV's own proc-2 map (`source="pv"`,
returns `None` if proc 2 has no T2 frame), and `fit_t2_map` always fits a
fresh map from the proc-1 echo stack (`source="fitted"`) regardless of
whether proc 2 exists — see the "Per-disc ROI analysis" section below for
why these are kept strictly separate. The from-scratch fit
(`analysis/fitting.py::fit_t2_monoexp`) was built to reproduce PV's own
fitted values as closely as possible, since PV's fit is the ground truth
users compare against.

### PV's fitting macros (reverse-engineered from `pdata/*/isa` files)

Studies in this dataset use one of two different ISA macros for the T2 map,
found by inspecting `##$ISA_func_name` in `pdata/2/isa`:

- **`t2blmodes_RA`** (later studies, e.g. `RA_2022_07_20` onward) — custom
  3-parameter macro: `S(TE) = A + C·exp(-TE/T2)`, with `A` (noise-floor
  bias) free. Critically, **the first echo's nominal TE is not used in the
  fit** — PV anchors it to effectively `TE≈0` (stimulated-echo
  contamination in MSME's first echo makes its true decay value
  unreliable, so PV pins it near the y-intercept instead of fitting it at
  its real TE). Reproduced in `fitting.py` via `_FIRST_ECHO_ANCHOR_MS = 1e-6`
  substituted for `te[argmin(te)]` before fitting.
- **`t2vtr`** (early studies, e.g. `RA_2022_07_18`) — PV's built-in 2-parameter
  macro: `S(TE) = C·exp(-TE/T2)`, bias fixed at 0, true echo times used
  throughout (no first-echo anchoring).

`fit_t2_monoexp` always fits the 3-parameter anchored model (a superset —
bias converges near 0 for `t2vtr`-style data). Initial guesses come from a
log-linear regression on the signal (skipping the anchored first echo), not
fixed constants.

### Validation

Compared against PV's own proc-2 T2 maps across 3 independent studies:
median |fitted − PV| ≈ 0.02 ms, >98% of plausible-range pixels within 1 ms.
`check_t2_map.py` prints these stats for any single study on demand.

**Known open issue**: `RA_2022_07_18.f01` exp 5 (an early `t2vtr`-macro
study) diverges much more than the others (~72% of pixels outside 1 ms
agreement). Root cause not confirmed — suspected to be a scaling or
data-specific quirk of this particular early acquisition rather than a
general modeling bug. Deprioritized; revisit if more `t2vtr`-style studies
show the same pattern.

### Slice alignment between proc 1 and proc 2

PV's proc-2 map is not always the same shape as proc 1's echo stack — it
may cover only one slice, or the same slices in a different order (see
"Two T2-map layouts" above). `analysis/maps.py::match_slices_to_proc1`
matches each proc-2 slice to its proc-1 index by nearest `VisuCorePosition`
(read directly from `visu_pars`), so fitted/PV/raw layers can be compared
voxel-for-voxel.

### Performance

`fit_t2_monoexp` fits per-pixel with `scipy.optimize.least_squares`
(`method="trf"`) using an **analytic Jacobian** (`_jacobian` in
`fitting.py`) rather than finite differences — ~4x faster per pixel than
the original `lmfit`-based solver, with byte-identical results (lmfit was
removed from `pyproject.toml` once this was verified).

Above `_PARALLEL_THRESHOLD = 2000` fit pixels, the column-wise fit is
automatically distributed across CPU cores via
`concurrent.futures.ProcessPoolExecutor` (`n_jobs` param, default
`os.cpu_count()`; pass `n_jobs=1` to force serial). Verified bit-identical
output between serial and parallel paths. ~4x additional speedup on 8
cores; a full 150×150×8 volume fits in ~64s (previously several minutes
with `lmfit`).

**Multiprocessing spawn hazard**: on macOS/Windows, worker processes
re-import the calling script (`multiprocessing`'s "spawn" method). Any
script that calls `fit_t2_monoexp` with parallelism enabled (the default)
**must** guard its top-level executable code with
`if __name__ == "__main__":`, or workers will re-run the whole script
(re-open napari, etc). `main.py` and `check_t2_map.py` already do this.

### Visual inspection

`check_t2_map.py` opens napari with PV's T2 map, the freshly fitted map,
their difference, R², and bias — all pixel-aligned via
`match_slices_to_proc1` — for one study/experiment (edit `STUDY_PATH`,
`EXP_ID`, `STUDY_NAME` constants at the top of the file). Also prints
median/1ms/5ms agreement stats to the console. Useful checks: disc tissue
(NP/AF) should agree with PV within ~1–2 ms; R² < 0.9 in disc tissue flags
partial-volume or B1 issues; background pixels should be NaN (transparent)
in the fitted map even where PV shows spurious high T2 values.

## Per-disc ROI analysis (`analyze_disc.py`, `analysis/rois.py`, `parameter_browser.py`)

For each disc, generate T2 + ADC maps, draw ROIs (NP/AF and any number of
named variants, e.g. "NP sample 1"), and extract mean/min/max/SD per ROI.
Two separate pieces, kept deliberately simple and decoupled:

- **`analyze_disc.py`** — standalone script (same style as `check_t2_map.py`):
  edit `STUDY_PATH`/`STUDY_NAME`/`EXP_ID_T2`/`EXP_ID_ADC` constants, run it.
  Always fits fresh T2 + ADC via `fit_t2_map`/`fit_adc_map` (native res,
  saved to `parameter_maps/`) — never uses ParaVision's own proc-2 map as
  the analysis source, so results don't depend on whether PV happened to
  already compute one. PV's map, when available, is loaded separately
  (`get_pv_t2_map`/`get_pv_adc_map`) purely as a smoke check: a printed
  agreement stat plus a hidden `"... — PV (smoke check)"` layer, never used
  for ROI stats. Opens napari with each map's own signal-intensity anatomy
  image — via `analysis.maps.fit_anatomy_image` — plus one dynamic Shapes
  layer per map ("T2 ROIs"/"ADC ROIs", `analysis.rois.add_roi_shapes_layer`)
  and a `RoiAnnotatorWidget` dock (`roi_annotator_widget.py`) rebuilding
  ParaVision's own "New -> pick shape -> draw -> named object" workflow:
  click a layer to target its map, click New, pick Polygon/Circle, trace —
  the shape is auto-named "ROI 1", "ROI 2", ... and listed live; double-click
  a list entry to rename/tag it (NP/AF). No fixed slots, no terminal
  prompts. After you close the window, it rasterizes + computes stats +
  upserts into `parameter_maps/roi_stats.csv`.
- **`parameter_browser.py`** — small **read-only** dock widget wired into
  `main.py`, next to the tree browser. When you load a study, it lists
  whatever `analyze_disc.py` already generated for that study and loads a
  selected map (+ its ROIs) into the viewer on button click. It never
  generates or draws anything — that split keeps the live app free of any
  async/threading code for the ~10-60s fits.

**Two parallel families in `analysis/maps.py`, deliberately kept separate,
neither falling back to the other**: `get_pv_t2_map`/`get_pv_adc_map` are
PV-only (extract PV's proc-2 map, return `None` if it doesn't exist or has
no matching frame) — that's what `check_t2_map.py`/`check_adc_map.py` use,
since their whole purpose is comparing PV against a fresh fit, and what
`analyze_disc.py` uses for its smoke check. `fit_t2_map`/`fit_adc_map`
always fit from proc 1 and never touch proc 2 — that's what
`analyze_disc.py` uses for the maps it actually saves and computes ROI
stats against, since per-disc analysis should always be your own generated
result. `fit_anatomy_image` is the matching
anatomy function for the always-fit family (always proc 1, pairs with
`fit_t2_map`/`fit_adc_map`); there's no PV-preferred equivalent since
`check_t2_map.py`/`check_adc_map.py` build their own anatomy image directly
from the same proc 1 stack they already load for fitting. **Don't mix
families** — PV's proc 2 can cover a different/reordered subset of slices
than proc 1 (see "Two T2-map layouts" below), so an always-fit map
(spanning all of proc 1's slices) paired with anything PV-sourced (often a
single PV-map slice) can silently land on different slice counts.
`analyze_disc.py` asserts the shapes match as a backstop.

**What the anatomy image is**: PV's proc 2 for a T2/ADC map always includes
a `"signal intensity"` FG_ISA frame alongside the parameter frame itself
(e.g. T2 proc 2 labels: `['absolute bias', ..., 'signal intensity', ...,
'T2 relaxation time', ...]`) — this is the base image ParaVision itself
shows you, but `analyze_disc.py` never reads it. `fit_anatomy_image` always
reads proc 1's own raw stack instead (first echo for T2, lowest-b ~b0 DWI
frame for ADC — highest, most reliable SNR everywhere; the highest-b frame
was tried first but rejected since high-diffusivity tissue like NP can sit
near the Rician noise floor there, making it a worse image to trace ROI
boundaries on despite its stronger NP/AF contrast), matching
`fit_t2_map`/`fit_adc_map`'s own proc-1-only source exactly. `match_slices_to_proc1` is unrelated to this — it's for
comparing proc 2 against proc 1's own slice ordering (used by
`check_t2_map.py`/`check_adc_map.py`).

**Why ROIs are never shared between T2 and ADC**: verified against real
data (`RA_2022_07_20.f22` exp 6/7, same disc) that T2 (MSME) and ADC
(DtiEpi) acquisitions have different FOV, matrix size, *and* in-plane
center offset — e.g. FOV 10×12mm/48×60 vs 20.8×12mm/128×72, offsets that
don't match either. Projecting one ROI across both grids would need real
affine registration; instead each anatomy image shares its own map's exact
native grid (both come from the same acquisition), so ROIs always
rasterize straight onto the array they were drawn against — no coordinate
transform anywhere. See `analysis/rois.py` module docstring.

**Storage**: `parameter_maps/` (top-level, sibling to `patients/`, gitignored):
```
parameter_maps/
  {study_name}_exp{exp_id}_t2.npy / .json
  {study_name}_exp{exp_id}_adc.npy / .json
  rois/{study_name}_exp{exp_id}_{t2,adc}.json
  roi_stats.csv
```
`study_name` (from `SUBJECT_study_name`, e.g. `"Rat Spine 690731"`) is
sanitized for filenames via `analysis.maps.sanitize_stem`. Re-running
`analyze_disc.py` for the same disc updates its `roi_stats.csv` rows in
place (keyed by study/exp/roi_name/map_kind) instead of duplicating them.

## ADC map fitting (`analysis/`)

`analysis/maps.py` mirrors the T2 split: `get_pv_adc_map` extracts PV's own
proc-2 "diffusion constant" ISA frame (`dtraceb` macro, `source="pv"`,
`None` if absent), and `fit_adc_map` always fits a combined ("trace") ADC
from proc 1's DWI stack — all 3 diffusion directions pooled with their
trace-corrected effective b-values (`PVM_DwEffBval`) — regardless of
whether proc 2 exists (`source="fitted"`). The fit
(`analysis/fitting.py::fit_adc_monoexp`) uses the same
`scipy.optimize.least_squares` + analytic-Jacobian + `ProcessPoolExecutor`
architecture as the T2 fit.

### Model

Mono-exponential `S(b) = C * exp(-b*D)`, bias fixed at 0 by default —
reproduces PV's `dtraceb` macro exactly (validated: median |fitted − PV| ~
4e-8 mm^2/s, 100% of plausible-range pixels within 5e-5 mm^2/s, across a
representative study). An optional `bias_free=True` mode frees the
noise-floor term as a 3rd parameter — useful as a diagnostic for Rician
noise-floor bias, but **not stable enough to report as a number**: with
only 5 b-values per direction (or 16 pooled), ADC can shift by an amount
comparable to its own magnitude between fixed- and free-bias fits, for only
a modest R² gain (~0.86 → ~0.91 in one test). Trace-weighted (orientation-
averaged over 3 directions) rather than full DTI/IVIM/DKI, because this
acquisition only has 3 diffusion directions and 5 b-values per direction —
not enough angular or b-value sampling to support those richer models. See
`analysis/fitting.py` module docstring for the full literature discussion.

### Visual inspection

`check_adc_map.py` — same layout as `check_t2_map.py` (PV map, fitted map,
difference, R², bias), for the combined/trace ADC.

`check_diffusion_directions.py` — fits ADC *per diffusion direction*
independently (using the same `fit_adc_monoexp`, just on one direction's
5-point sub-stack at a time) instead of PV's pooled trace value, so you can
see whether an apparent ADC finding depends on which direction you look
from. Large direction-to-direction spread suggests anisotropy/orientation
is driving the result rather than a true isotropic diffusivity change.
Also includes an SNR-vs-ADC diagnostic (via a difference-image noise
estimate, since this cropped EPI acquisition has no reliable background
region) to check whether an elevated ADC is a Rician noise-floor fitting
artifact rather than genuine biology.
