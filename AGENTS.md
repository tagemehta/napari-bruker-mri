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

`analysis/maps.py::get_t2_map` is the single entry point: it uses PV's own
proc-2 map when present (`source="pv"`), and falls back to fitting a fresh
map from the proc-1 echo stack when proc 2 is absent (`source="fitted"`) —
not every study has a T2 map on disk. The from-scratch fit
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

## ADC map fitting (`analysis/`)

`analysis/maps.py::get_adc_map` mirrors `get_t2_map`: uses PV's own proc-2
"diffusion constant" ISA frame (`dtraceb` macro) when present
(`source="pv"`), falls back to fitting a combined ("trace") ADC from proc 1's
DWI stack — all 3 diffusion directions pooled with their trace-corrected
effective b-values (`PVM_DwEffBval`) — when proc 2 is absent
(`source="fitted"`). The fit (`analysis/fitting.py::fit_adc_monoexp`) uses
the same `scipy.optimize.least_squares` + analytic-Jacobian +
`ProcessPoolExecutor` architecture as the T2 fit.

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
