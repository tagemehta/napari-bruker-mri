"""
Data model classes for the three-level Bruker hierarchy:
    PatientNode  ->  StudyNode  ->  ExpNode

Each class scans its folder on construction but defers heavy I/O
(loading image data) until explicitly requested.

JCAMP-DX value format notes
----------------------------
String/array parameters in Bruker files use a two-line format:

    ##$PULPROG=( 32 )
    <DtiEpi.ppg>

The first line carries the key and an array-length hint; the actual value
is on the next line, optionally enclosed in < >.  Single-line numeric
parameters look like:

    ##$VisuCoreDim=2

_read_jcampdx_value handles both forms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _read_jcampdx_value(path: Path, key: str) -> str:
    """
    Return the string value of a JCAMP-DX parameter or ''.

    Handles both inline values (##$KEY=value) and two-line string arrays
    (##$KEY=( N )\\n<value>).
    """
    try:
        with open(path, encoding="latin-1", errors="replace") as fh:
            lines = fh.readlines()
        for i, line in enumerate(lines):
            if line.startswith((f"##${key}=", f"##{key}=")):
                _, _, val = line.partition("=")
                val = val.strip()
                # Two-line form: value is "( N )" — actual content on next line
                if val.startswith("(") and val.endswith(")") and i + 1 < len(lines):
                    val = lines[i + 1].strip()
                return val.strip("<>")
    except OSError:
        pass
    return ""


@dataclass
class ExpNode:
    """
    A single experiment (scan) folder inside a Study.

    Metadata (sequence name, scan description) is loaded lazily on first
    access so scanning a large patient list stays fast.

    Attributes
    ----------
    study_path:
        Path to the parent Study folder (contains the ``subject`` file).
    exp_id:
        Numeric folder name, e.g. ``"3"``.
    study_name:
        Human-readable study label propagated from ``StudyNode.study_name``,
        used to build the napari layer name (e.g. ``"Rat Spine 690728 - 9"``).
    """

    study_path: Path
    exp_id: str  # numeric folder name, e.g. "3"
    study_name: str = (
        ""  # human-readable study name for display, e.g. "Rat Spine 690728"
    )

    # Populated lazily via _load_metadata()
    _sequence: str = field(default="", init=False, repr=False)
    _description: str = field(default="", init=False, repr=False)
    _metadata_loaded: bool = field(default=False, init=False, repr=False)

    @property
    def path(self) -> Path:
        """Absolute path to this experiment folder."""
        return self.study_path / self.exp_id

    def _load_metadata(self) -> None:
        """Read sequence and description from ``acqp`` / ``method`` on first call."""
        if self._metadata_loaded:
            return
        acqp = self.path / "acqp"
        method = self.path / "method"
        self._sequence = _read_jcampdx_value(acqp, "PULPROG")
        self._description = _read_jcampdx_value(acqp, "ACQ_scan_name")
        if not self._description:
            self._description = _read_jcampdx_value(acqp, "ACQ_protocol_name")
        if not self._description:
            self._description = _read_jcampdx_value(method, "PVM_ScanJobName")
        self._metadata_loaded = True

    @property
    def sequence(self) -> str:
        """Pulse programme name from ``acqp`` (e.g. ``"MSME.ppg"``)."""
        self._load_metadata()
        return self._sequence

    @property
    def description(self) -> str:
        """Short scan description from ``ACQ_scan_name`` / ``ACQ_protocol_name``."""
        self._load_metadata()
        return self._description

    @property
    def display_name(self) -> str:
        """``"{exp_id} - {sequence}"`` e.g. ``"9 - MSME.ppg"``."""
        parts = [self.exp_id]
        if self.sequence:
            parts.append(self.sequence)
        return " - ".join(parts)

    def has_2dseq(self, proc_id: str = "1") -> bool:
        """Return True if ``pdata/{proc_id}/2dseq`` exists."""
        return (self.path / "pdata" / proc_id / "2dseq").exists()

    def available_proc_ids(self) -> list[str]:
        """Sorted list of proc folder names that contain a ``2dseq`` file."""
        pdata = self.path / "pdata"
        if not pdata.is_dir():
            return []
        return sorted(
            (d.name for d in pdata.iterdir() if d.is_dir() and (d / "2dseq").exists()),
            key=lambda n: int(n) if n.isdigit() else 10_000_000,
        )

    def proc_nodes(self) -> list[ProcNode]:
        """Return a ``ProcNode`` for every available proc folder."""
        return [ProcNode(exp=self, proc_id=pid) for pid in self.available_proc_ids()]


@dataclass
class ProcNode:
    """
    One processed reconstruction (``pdata/N``) inside an experiment.

    Proc 1 is typically the raw scanner reconstruction.
    Higher-numbered procs are derived outputs generated in ParaVision:
    T2 maps, ADC maps, DTI tensors, etc.

    Attributes
    ----------
    exp:
        Parent experiment.
    proc_id:
        Proc folder name, e.g. ``"1"`` or ``"2"``.
    """

    exp: ExpNode
    proc_id: str  # e.g. "1", "2"

    _label: str = field(default="", init=False, repr=False)
    _comment: str = field(default="", init=False, repr=False)
    _meta_loaded: bool = field(default=False, init=False, repr=False)

    @property
    def pdata_path(self) -> Path:
        """Absolute path to this proc folder (``…/pdata/{proc_id}``)."""
        return self.exp.path / "pdata" / self.proc_id

    def _load_meta(self) -> None:
        """Read protocol label and series comment from ``visu_pars`` on first call."""
        if self._meta_loaded:
            return
        visu = self.pdata_path / "visu_pars"
        # Protocol name (meaningful for proc 1 = raw recon)
        self._label = _read_jcampdx_value(visu, "VisuAcquisitionProtocol")
        # Series comment (meaningful for derived procs: T2 map, DTI tensors, …)
        self._comment = _read_jcampdx_value(visu, "VisuSeriesComment")
        self._meta_loaded = True

    @property
    def protocol(self) -> str:
        """Acquisition protocol name from ``VisuAcquisitionProtocol`` (meaningful for proc 1)."""
        self._load_meta()
        return self._label

    @property
    def comment(self) -> str:
        """Series comment from ``VisuSeriesComment`` (describes derived procs: T2 map, ADC, …)."""
        self._load_meta()
        return self._comment

    @property
    def description(self) -> str:
        """Best single-line description: comment preferred for derived procs."""
        self._load_meta()
        return self._comment or self._label


@dataclass
class StudyNode:
    """
    A single Bruker Study folder containing the ``subject`` file and
    one numeric sub-folder per experiment (scan).

    Subject metadata (name, study name, date) is read lazily from the
    ``subject`` JCAMP-DX file on first access.
    """

    path: Path

    _name: str = field(default="", init=False, repr=False)
    _study_name: str = field(default="", init=False, repr=False)
    _date: str = field(default="", init=False, repr=False)
    _experiments: list[ExpNode] | None = field(default=None, init=False, repr=False)
    _meta_loaded: bool = field(default=False, init=False, repr=False)

    def _load_meta(self) -> None:
        """Read subject name, study name, and date from the ``subject`` file on first call."""
        if self._meta_loaded:
            return
        subject_file = self.path / "subject"
        # Prefer human-readable name; fall back through several key variants
        self._name = _read_jcampdx_value(
            subject_file, "SUBJECT_name_string"
        ) or _read_jcampdx_value(subject_file, "SUBJECT_id")
        self._study_name = (
            _read_jcampdx_value(subject_file, "SUBJECT_study_name") or self.path.name
        )
        self._date = _read_jcampdx_value(subject_file, "SUBJECT_date")
        self._meta_loaded = True

    @property
    def subject_name(self) -> str:
        """Animal/subject identifier from ``SUBJECT_name_string`` or ``SUBJECT_id``."""
        self._load_meta()
        return self._name

    @property
    def study_name(self) -> str:
        """Human-readable study label from ``SUBJECT_study_name`` (e.g. ``"Rat Spine 690728"``)."""
        self._load_meta()
        return self._study_name

    @property
    def date(self) -> str:
        """Acquisition date/time string from ``SUBJECT_date``."""
        self._load_meta()
        return self._date

    @property
    def display_name(self) -> str:
        """Returns e.g. 'TEST_IO  |  14:13:18 16 Jun 2020'."""
        parts = [self.study_name]
        if self.date:
            parts.append(self.date)
        return "  |  ".join(parts)

    @property
    def experiments(self) -> list[ExpNode]:
        if self._experiments is None:
            self._experiments = _scan_experiments(self.path, self.study_name)
        return self._experiments


def _scan_experiments(study_path: Path, study_name: str = "") -> list[ExpNode]:
    """Return sorted ExpNodes for all numeric sub-folders with acqp."""
    nodes: list[ExpNode] = []
    try:
        for entry in sorted(study_path.iterdir(), key=lambda e: _sort_key(e.name)):
            if entry.is_dir() and entry.name.isdigit() and (entry / "acqp").exists():
                nodes.append(
                    ExpNode(
                        study_path=study_path,
                        exp_id=entry.name,
                        study_name=study_name,
                    )
                )
    except OSError:
        pass
    return nodes


def _sort_key(name: str) -> tuple[int, str]:
    """Sort numeric names numerically; non-numeric names sort after them lexicographically."""
    if name.isdigit():
        return (int(name), "")
    return (10_000_000, name)


@dataclass
class PatientNode:
    """
    Represents one patient/animal entry in the tree.

    In Layout A (patient folders containing study folders) `path` is the
    patient folder and all its study sub-folders are returned.

    In Layout B (study folders sitting directly inside the root) each
    PatientNode wraps exactly one study folder passed via `_single_study`.
    `path` is still the parent root so the PatientNode can be constructed
    consistently, but `studies` returns only the pinned study.
    """

    path: Path
    _single_study: Path | None = field(default=None, repr=False)
    _studies: list[StudyNode] | None = field(default=None, init=False, repr=False)

    @property
    def display_name(self) -> str:
        """Folder name is the most reliable unique identifier per animal."""
        if self._single_study is not None:
            return self._single_study.name
        studies = self.studies
        if studies:
            name = studies[0].subject_name
            if name:
                return name
        return self.path.name

    @property
    def studies(self) -> list[StudyNode]:
        if self._studies is None:
            if self._single_study is not None:
                self._studies = [StudyNode(path=self._single_study)]
            else:
                self._studies = _scan_studies(self.path)
        return self._studies


def _scan_studies(patient_path: Path) -> list[StudyNode]:
    """Return StudyNodes for all sub-folders of *patient_path* that contain a ``subject`` file."""
    nodes: list[StudyNode] = []
    try:
        for entry in sorted(patient_path.iterdir(), key=lambda e: e.name):
            if entry.is_dir() and (entry / "subject").exists():
                nodes.append(StudyNode(path=entry))
    except OSError:
        pass
    return nodes


def scan_patient_list(root: str | Path) -> list[PatientNode]:
    """
    Scan a root directory for patient folders.

    Handles three layouts automatically:

    Layout A — root contains patient folders, each containing study folders:
        root/
          PatientA/
            Study1/subject
            Study2/subject
          PatientB/
            Study1/subject

    Layout B — root directly contains study folders (one patient per study folder,
    or one patient with multiple studies all in root):
        root/
          sample_data/subject
          another_study/subject

    Layout C — root itself is a single study:
        root/subject
    """
    root = Path(root)
    patients: list[PatientNode] = []

    # Layout C: root is a single study
    if (root / "subject").exists():
        patients.append(PatientNode(path=root.parent))
        return patients

    try:
        entries = [
            e for e in sorted(root.iterdir(), key=lambda e: e.name) if e.is_dir()
        ]
    except OSError:
        return patients

    # Layout B: root contains study folders directly (each has a 'subject' file).
    # Each study folder is its own animal/patient — give each its own PatientNode
    # so the tree shows one top-level entry per study rather than one giant group.
    direct_studies = [e for e in entries if (e / "subject").exists()]
    if direct_studies:
        for study_dir in direct_studies:
            patients.append(PatientNode(path=study_dir.parent, _single_study=study_dir))
        return patients

    # Layout A: root contains patient folders that in turn contain study folders
    for entry in entries:
        try:
            has_study = any(
                (child / "subject").exists()
                for child in entry.iterdir()
                if child.is_dir()
            )
        except OSError:
            continue
        if has_study:
            patients.append(PatientNode(path=entry))

    return patients
