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
    """A single experiment (scan) folder inside a Study."""

    study_path: Path
    exp_id: str  # numeric folder name, e.g. "3"

    # Populated lazily via _load_metadata()
    _sequence: str = field(default="", init=False, repr=False)
    _description: str = field(default="", init=False, repr=False)
    _metadata_loaded: bool = field(default=False, init=False, repr=False)

    @property
    def path(self) -> Path:
        return self.study_path / self.exp_id

    def _load_metadata(self) -> None:
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
        self._load_metadata()
        return self._sequence

    @property
    def description(self) -> str:
        self._load_metadata()
        return self._description

    @property
    def display_name(self) -> str:
        parts = [self.exp_id]
        if self.sequence:
            parts.append(self.sequence)
        return " - ".join(parts)

    def has_2dseq(self, proc_id: str = "1") -> bool:
        return (self.path / "pdata" / proc_id / "2dseq").exists()

    def available_proc_ids(self) -> list[str]:
        pdata = self.path / "pdata"
        if not pdata.is_dir():
            return []
        return sorted(
            (d.name for d in pdata.iterdir() if d.is_dir() and (d / "2dseq").exists()),
            key=lambda n: int(n) if n.isdigit() else 10_000_000,
        )

    def proc_nodes(self) -> list[ProcNode]:
        return [ProcNode(exp=self, proc_id=pid) for pid in self.available_proc_ids()]


@dataclass
class ProcNode:
    """One processed reconstruction (pdata/N) inside an experiment."""

    exp: ExpNode
    proc_id: str  # e.g. "1", "2"

    _label: str = field(default="", init=False, repr=False)
    _comment: str = field(default="", init=False, repr=False)
    _meta_loaded: bool = field(default=False, init=False, repr=False)

    @property
    def pdata_path(self) -> Path:
        return self.exp.path / "pdata" / self.proc_id

    def _load_meta(self) -> None:
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
        self._load_meta()
        return self._label

    @property
    def comment(self) -> str:
        self._load_meta()
        return self._comment

    @property
    def description(self) -> str:
        """Best single-line description: comment preferred for derived procs."""
        self._load_meta()
        return self._comment or self._label


@dataclass
class StudyNode:
    """A single Bruker Study folder (subject info + experiments)."""

    path: Path

    _name: str = field(default="", init=False, repr=False)
    _study_name: str = field(default="", init=False, repr=False)
    _date: str = field(default="", init=False, repr=False)
    _experiments: list[ExpNode] | None = field(default=None, init=False, repr=False)
    _meta_loaded: bool = field(default=False, init=False, repr=False)

    def _load_meta(self) -> None:
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
        self._load_meta()
        return self._name

    @property
    def study_name(self) -> str:
        self._load_meta()
        return self._study_name

    @property
    def date(self) -> str:
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
            self._experiments = _scan_experiments(self.path)
        return self._experiments


def _scan_experiments(study_path: Path) -> list[ExpNode]:
    """Return sorted ExpNodes for all numeric sub-folders with acqp."""
    nodes: list[ExpNode] = []
    try:
        for entry in sorted(study_path.iterdir(), key=lambda e: _sort_key(e.name)):
            if entry.is_dir() and entry.name.isdigit() and (entry / "acqp").exists():
                nodes.append(ExpNode(study_path=study_path, exp_id=entry.name))
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
    """Represents a patient directory containing one or more StudyNodes."""

    path: Path

    _studies: list[StudyNode] | None = field(default=None, init=False, repr=False)

    @property
    def display_name(self) -> str:
        """Use the subject name from the first study if available, else folder name."""
        studies = self.studies
        if studies:
            name = studies[0].subject_name
            if name:
                return name
        return self.path.name

    @property
    def studies(self) -> list[StudyNode]:
        if self._studies is None:
            self._studies = _scan_studies(self.path)
        return self._studies


def _scan_studies(patient_path: Path) -> list[StudyNode]:
    """Return StudyNodes for all sub-folders that look like Bruker studies."""
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

    # Layout B: root contains study folders directly (each has a 'subject' file)
    direct_studies = [e for e in entries if (e / "subject").exists()]
    if direct_studies:
        # Group all direct studies under one synthetic patient node whose path is root.
        # PatientNode.display_name will use the subject name from the first study.
        patients.append(PatientNode(path=root))
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
