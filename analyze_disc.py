"""
Generate + annotate T2 and ADC maps for one disc — run with:
    uv run python analyze_disc.py

T2 and ADC come from two different acquisitions (different sequences,
different exp_ids) even for "the same disc" — set EXP_ID_T2/EXP_ID_ADC
below to whichever two experiments cover that disc.

What this does
---------------
1. For each of T2 and ADC:
   * If a previously-saved map exists in ``parameter_maps/`` and
     ``REFIT`` is False (the default), reuses it — no refit.
   * Otherwise fits fresh from the raw proc 1 data via
     ``analysis.maps.fit_t2_map`` / ``fit_adc_map`` (native resolution,
     never upsampled) and saves it to ``parameter_maps/``.
   Never uses ParaVision's own proc 2 map as the analysis source, so
   results don't depend on whether PV happened to already compute one.
   PV's map, when available, is loaded separately purely as a smoke
   check (printed agreement stats + a hidden comparison layer in
   napari) — never used for ROI stats.
2. Opens napari with:
   - T2 anatomy + T2 map + one "T2 ROIs" Shapes layer
   - ADC anatomy + ADC map + one "ADC ROIs" Shapes layer
   - A "ROI Annotator" dock widget (see ``roi_annotator_widget.py``)
     rebuilding ParaVision's own annotation workflow: click a T2 or ADC
     layer to target that map, click New, pick Polygon or Rectangle, trace
     it — it's immediately auto-named "ROI 1", "ROI 2", ... and listed in
     the widget.  Double-click a list entry (or "Rename Selected") to
     rename it. Draw as many ROIs per map as you want; previously saved
     ROIs are pre-populated.
3. When you close napari, stats (mean/min/max/SD) are computed per ROI
   against its own map and upserted into parameter_maps/roi_stats.csv.

Layer order in napari (top to bottom):
    ADC ROIs, T2 ROIs,
    T2 map, T2 anatomy (signal intensity),
    ADC map, ADC anatomy (signal intensity),
    T2 map — PV (smoke check, hidden),
    ADC map — PV (smoke check, hidden)

Guarded by ``if __name__ == "__main__":`` since fit_t2_monoexp/
fit_adc_monoexp may parallelize the fit across worker processes — see
AGENTS.md "Multiprocessing spawn hazard".
"""

import logging
from pathlib import Path

import napari
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from analysis.maps import (
    anatomy_path,
    fit_adc_map,
    fit_anatomy_image,
    fit_t2_map,
    get_pv_adc_map,
    get_pv_t2_map,
    load_map,
    map_path,
    save_map,
)
from analysis.rois import (
    RoiRecord,
    add_roi_shapes_layer,
    collect_rois_from_shapes_layer,
    compute_roi_stats,
    load_rois,
    rasterize_roi,
    rois_path,
    save_rois,
    upsert_roi_stats,
)
from roi_annotator_widget import RoiAnnotatorWidget

# ── change these to analyze a different disc ────────────────────────────────
STUDY_PATH = "patients/real_data/RA_2022_07_21.f32"
STUDY_NAME = "690068"
EXP_ID_T2 = "9"
EXP_ID_ADC = "10"
# Set to True to force a fresh fit even if a previously-saved map exists.
REFIT = False
# ─────────────────────────────────────────────────────────────────────────────

# Upsample the *displayed* anatomy image to at least this many pixels/axis
# (cubic spline, display only — ROI stats always use the native map grid).
# Set to 0 to disable and show anatomy at native resolution.
ANATOMY_ZOOM_TO = 256

_T2_CONTRAST = [5, 200]
_ADC_CONTRAST = [0.0002, 0.0025]
_FOOTPRINT_RTOL = 1e-3


def _assert_same_footprint(
    anatomy_shape: tuple[int, ...],
    anatomy_scale: tuple[float, ...],
    map_shape: tuple[int, ...],
    map_scale: tuple[float, ...],
    label: str,
) -> None:
    """
    Verify the anatomy image covers the exact same physical extent as its
    map, axis by axis (pixel count times scale) — not that their pixel
    counts match, since the anatomy image may be displayed upsampled
    (``ANATOMY_ZOOM_TO``) while the map never is.
    """
    for i, (a_n, a_s, m_n, m_s) in enumerate(
        zip(anatomy_shape, anatomy_scale, map_shape, map_scale)
    ):
        a_extent, m_extent = a_n * a_s, m_n * m_s
        if not np.isclose(a_extent, m_extent, rtol=_FOOTPRINT_RTOL):
            raise RuntimeError(
                f"{label} anatomy axis {i} covers {a_extent:.4g}mm but its map "
                f"covers {m_extent:.4g}mm (anatomy shape={anatomy_shape} "
                f"scale={anatomy_scale}, map shape={map_shape} scale={map_scale})"
            )


def _load_or_fit_map(kind: str, study_path: Path, exp_id: str):
    """
    Reuse a saved map for (STUDY_NAME, exp_id, kind) if one exists and
    ``REFIT`` is False; otherwise fit fresh and save.
    """
    saved = map_path(STUDY_NAME, exp_id, kind)
    if not REFIT and saved.exists():
        print(
            f"Loading saved {kind.upper()} map from {saved} (set REFIT=True to refit)..."
        )
        return load_map(STUDY_NAME, exp_id, kind)
    print(
        f"Fitting {kind.upper()} map (exp {exp_id}) — always from proc 1, never PV's proc 2..."
    )
    fitter = {"t2": fit_t2_map, "adc": fit_adc_map}[kind]
    result = fitter(study_path, exp_id, study_name=STUDY_NAME)
    save_map(result)
    print(f"  shape={result.data.shape}  -> saved to {saved}")
    return result


def _smoke_check_pv(fitted, pv, thresholds: tuple[float, float], unit: str) -> None:
    """Print PV-vs-fitted agreement — sanity check only, never used for ROI stats."""
    kind = fitted.kind.upper()
    if pv is None:
        print(f"  No PV {kind} map available — nothing to smoke-check against.")
        return
    if pv.data.shape != fitted.data.shape:
        print(
            f"  PV {kind} map found ({pv.meta.get('pv_label', '?')!r}) but its shape "
            f"{pv.data.shape} differs from fitted {fitted.data.shape} — "
            "skipping numeric comparison; still shown as a hidden layer."
        )
        return
    diff = (fitted.data - pv.data)[np.isfinite(fitted.data - pv.data)]
    if diff.size == 0:
        print(f"  PV {kind} map found but no overlapping finite pixels to compare.")
        return
    lo, hi = thresholds
    print(
        f"  Smoke check vs PV {kind} ({pv.meta.get('pv_label', '?')!r}): "
        f"median|diff|={np.median(np.abs(diff)):.4g} {unit}, "
        f"within {lo:g} {unit}={np.mean(np.abs(diff) < lo) * 100:.1f}%, "
        f"within {hi:g} {unit}={np.mean(np.abs(diff) < hi) * 100:.1f}%"
    )


def main() -> None:
    study_path = Path(STUDY_PATH)

    t2_result = _load_or_fit_map("t2", study_path, EXP_ID_T2)
    adc_result = _load_or_fit_map("adc", study_path, EXP_ID_ADC)

    print("Smoke-checking against PV's own map, if available...")
    pv_t2 = get_pv_t2_map(study_path, EXP_ID_T2, STUDY_NAME)
    _smoke_check_pv(t2_result, pv_t2, thresholds=(1.0, 5.0), unit="ms")
    pv_adc = get_pv_adc_map(study_path, EXP_ID_ADC, STUDY_NAME)
    _smoke_check_pv(adc_result, pv_adc, thresholds=(5e-5, 1e-4), unit="mm^2/s")

    # Anatomy images are cached to parameter_maps/ so subsequent runs skip
    # the full brukerapi load + scipy zoom.  REFIT=True clears the cache.
    for _kind, _exp_id in (("t2", EXP_ID_T2), ("adc", EXP_ID_ADC)):
        cached = anatomy_path(STUDY_NAME, _exp_id, _kind, ANATOMY_ZOOM_TO).exists()
        if REFIT and cached:
            anatomy_path(STUDY_NAME, _exp_id, _kind, ANATOMY_ZOOM_TO).unlink(
                missing_ok=True
            )
            anatomy_path(STUDY_NAME, _exp_id, _kind, ANATOMY_ZOOM_TO).with_suffix(
                ".json"
            ).unlink(missing_ok=True)
            cached = False
        status = "cached" if cached else "computing (will cache for next run)..."
        print(f"  Anatomy {_kind.upper()} exp {_exp_id}: {status}")

    t2_anatomy, t2_anatomy_scale = fit_anatomy_image(
        study_path, EXP_ID_T2, STUDY_NAME, kind="t2", zoom_to=ANATOMY_ZOOM_TO
    )
    adc_anatomy, adc_anatomy_scale = fit_anatomy_image(
        study_path, EXP_ID_ADC, STUDY_NAME, kind="adc", zoom_to=ANATOMY_ZOOM_TO
    )
    _assert_same_footprint(
        t2_anatomy.shape, t2_anatomy_scale, t2_result.data.shape, t2_result.scale, "T2"
    )
    _assert_same_footprint(
        adc_anatomy.shape,
        adc_anatomy_scale,
        adc_result.data.shape,
        adc_result.scale,
        "ADC",
    )

    t2_preloaded: list[RoiRecord] = []
    if rois_path(STUDY_NAME, EXP_ID_T2, "t2").exists():
        t2_preloaded = load_rois(STUDY_NAME, EXP_ID_T2, "t2")
        print(f"  Loaded {len(t2_preloaded)} previously saved T2 ROI(s).")

    adc_preloaded: list[RoiRecord] = []
    if rois_path(STUDY_NAME, EXP_ID_ADC, "adc").exists():
        adc_preloaded = load_rois(STUDY_NAME, EXP_ID_ADC, "adc")
        print(f"  Loaded {len(adc_preloaded)} previously saved ADC ROI(s).")

    viewer = napari.Viewer(
        title=f"Disc analysis — {STUDY_NAME}  T2 exp{EXP_ID_T2} / ADC exp{EXP_ID_ADC}"
    )

    # ── image layers (added first so they sit below the ROI layers) ─────────
    viewer.add_image(
        t2_anatomy,
        name="T2 anatomy (signal intensity)",
        scale=t2_anatomy_scale,
        colormap="gray",
    )
    viewer.add_image(
        t2_result.data,
        name="T2 map",
        scale=t2_result.scale,
        colormap="magma",
        contrast_limits=_T2_CONTRAST,
    )
    viewer.add_image(
        adc_anatomy,
        name="ADC anatomy (signal intensity)",
        scale=adc_anatomy_scale,
        colormap="gray",
    )
    viewer.add_image(
        adc_result.data,
        name="ADC map",
        scale=adc_result.scale,
        colormap="magma",
        contrast_limits=_ADC_CONTRAST,
    )

    # PV reference — hidden by default, purely a smoke-check visual.
    if pv_t2 is not None:
        viewer.add_image(
            pv_t2.data,
            name="T2 map — PV (smoke check)",
            scale=pv_t2.scale,
            colormap="magma",
            contrast_limits=_T2_CONTRAST,
            visible=False,
        )
    if pv_adc is not None:
        viewer.add_image(
            pv_adc.data,
            name="ADC map — PV (smoke check)",
            scale=pv_adc.scale,
            colormap="magma",
            contrast_limits=_ADC_CONTRAST,
            visible=False,
        )

    # ── ROI layer — one dynamic Shapes layer per map ────────────────────────
    # Shapes layers are scaled to each map's own native scale (not the
    # anatomy image's, which may be upsampled for display) — napari aligns
    # layers by physical scale, not pixel count, so shapes drawn against
    # the (possibly zoomed) anatomy image still land in native map pixel
    # coordinates.  See fit_anatomy_image's docstring.
    t2_shapes = add_roi_shapes_layer(
        viewer, "T2 ROIs", t2_result.data.ndim, t2_preloaded, t2_result.scale
    )
    adc_shapes = add_roi_shapes_layer(
        viewer, "ADC ROIs", adc_result.data.ndim, adc_preloaded, adc_result.scale
    )

    # ── ROI annotator dock widget ────────────────────────────────────────────
    # Click a T2 or ADC layer (anatomy, map, or its ROI layer) to target it,
    # then use the widget: New -> pick Polygon/Circle -> draw -> auto-named
    # "ROI N" and listed live.  Double-click a list entry to rename/tag it.
    annotator = RoiAnnotatorWidget(viewer)
    annotator.register_target(
        "T2", t2_shapes, ["T2 anatomy (signal intensity)", "T2 map"]
    )
    annotator.register_target(
        "ADC", adc_shapes, ["ADC anatomy (signal intensity)", "ADC map"]
    )
    viewer.window.add_dock_widget(annotator, name="ROI Annotator")

    print()
    print("Napari open.  To draw an ROI:")
    print("  1. Click a T2 or ADC layer to target that map.")
    print("  2. In the 'ROI Annotator' panel: pick Polygon/Circle, click New.")
    print("  3. Trace it (double-click/Enter to finish a polygon).")
    print("  4. Repeat for as many ROIs as you want.  Close when done.")
    napari.run()

    # ── collect, validate, save ─────────────────────────────────────────────
    t2_rois = collect_rois_from_shapes_layer(t2_shapes, STUDY_NAME, EXP_ID_T2, "t2")
    adc_rois = collect_rois_from_shapes_layer(adc_shapes, STUDY_NAME, EXP_ID_ADC, "adc")

    save_rois(t2_rois, STUDY_NAME, EXP_ID_T2, "t2")
    save_rois(adc_rois, STUDY_NAME, EXP_ID_ADC, "adc")

    rows = []
    for rois, map_result, exp_id in (
        (t2_rois, t2_result, EXP_ID_T2),
        (adc_rois, adc_result, EXP_ID_ADC),
    ):
        for roi in rois:
            mask = rasterize_roi(roi, map_result.data.shape)
            stats = compute_roi_stats(mask, map_result.data)
            rows.append({
                "study_name": STUDY_NAME,
                "exp_id": exp_id,
                "roi_name": roi.name,
                "tissue": roi.tissue,
                "map_kind": map_result.kind,
                "map_source": map_result.source,
                **stats,
            })

    if rows:
        csv_path = upsert_roi_stats(rows)
        print(f"\nSaved {len(t2_rois)} T2 ROI(s), {len(adc_rois)} ADC ROI(s).")
        print(f"Wrote {len(rows)} stat row(s) to {csv_path}")
    else:
        print("\nNo ROIs drawn — nothing to save.")


if __name__ == "__main__":
    main()
