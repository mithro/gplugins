# SPDX-License-Identifier: MIT
"""Running Charon: the interpreter, then ``charon_mp.exe`` as one process.

``charonInterpreter.py -i deck.inp`` writes ``deck.inp.xml``; ``charon_mp.exe --i=deck.inp.xml
--current`` solves it, single rank, without ``mpirun`` (the Charon image sets
``OMPI_MCA_ess_singleton_isolated=1`` so OpenMPI runs as a singleton). Both run in the current
working directory, which holds the mesh, the deck and every output. A run whose log lacks
``Charon run completed.`` raises :class:`CharonError` with the log.

Charon appends to ``currents-loca.dat``; :func:`run_deck` removes it first, so the file holds
only this run's rows.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

#: What NOX prints when it uses a linear solve that did not converge (Rescue Bad Newton Solve).
RESCUED = "using the step anyway"


class CharonError(RuntimeError):
    """A Charon run failed.

    ``log`` is its complete output, ``failed_at`` the bias (V) of the LOCA step that failed to
    converge, or None when the run failed otherwise, and ``deck`` the deck file that ran.
    """

    def __init__(
        self,
        message: str,
        log: str,
        failed_at: float | None = None,
        deck: str | None = None,
    ) -> None:
        """``message``, the run's complete output ``log``, the failed step's bias, the deck."""
        super().__init__(message)
        self.log = log
        self.failed_at = failed_at
        self.deck = deck


@dataclass(frozen=True)
class CharonExe:
    """Where the interpreter and the solver are: ``CHARON_EXECUTABLE_PATH`` by default."""

    bin_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("CHARON_EXECUTABLE_PATH", "/opt/charon/bin")
        )
    )
    python: str = "python3"
    timeout_s: float = 6 * 3600

    @property
    def interpreter(self) -> Path:
        return self.bin_dir / "charonInterpreter.py"

    @property
    def solver(self) -> Path:
        return self.bin_dir / os.environ.get("CHARON_EXECUTABLE", "charon_mp.exe")


@dataclass(frozen=True)
class RunLog:
    """The complete output of one deck's interpreter and solver runs."""

    deck: str
    interpreter: str
    solver: str


@dataclass(frozen=True)
class StepInfo:
    """A Newton solve as the NOX log reports it: iterations, final F-norm and WRMS norm."""

    iterations: int
    f_norm: float | None
    wrms: float | None


def _run(cmd: list[str], *, cwd: Path, timeout: float, what: str):
    try:
        return subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + (e.stderr or "")
        log = out.decode(errors="replace") if isinstance(out, bytes) else out
        raise CharonError(f"{what} timed out after {timeout} s", log) from e


def run_deck(deck_file: str, *, cwd: Path, exe: CharonExe = CharonExe()) -> RunLog:
    """Interpret and run ``deck_file`` in ``cwd``; raise :class:`CharonError` on failure.

    A run fails on a non-zero exit, a log without ``Charon run completed.``, a time-out, or a
    log in which NOX used an unconverged linear solve (:data:`RESCUED`).
    """
    cwd = Path(cwd)
    (cwd / "currents-loca.dat").unlink(missing_ok=True)
    r = _run([exe.python, "-W", "ignore", str(exe.interpreter), "-i", deck_file], cwd=cwd,
             timeout=exe.timeout_s, what=f"charonInterpreter.py on {deck_file}")  # fmt: skip
    interp = r.stdout + r.stderr
    if r.returncode != 0 or not (cwd / f"{deck_file}.xml").is_file():
        raise CharonError(
            f"charonInterpreter.py failed on {deck_file} ({r.returncode})", interp
        )
    # charon_mp.exe buffers stdout: an MPI_ABORT would lose the solver's last lines, so it runs
    # unbuffered under stdbuf (in the Charon image), which is required
    if shutil.which("stdbuf") is None:
        raise CharonError("stdbuf is required to run charon_mp.exe unbuffered", interp)
    r = _run(["stdbuf", "-o0", "-e0", str(exe.solver), f"--i={deck_file}.xml", "--current"], cwd=cwd,
             timeout=exe.timeout_s, what=f"charon_mp.exe on {deck_file}")  # fmt: skip
    log = r.stdout + r.stderr
    (cwd / f"{deck_file}.log").write_text(log, encoding="utf-8")
    if r.returncode != 0 or "Charon run completed." not in log:
        fails = re.findall(
            r"Value of continuation parameter at failed step = (\S+)", log
        )
        raise CharonError(
            f"charon_mp.exe failed on {deck_file} (exit {r.returncode}"
            + (f"; LOCA failed at {fails[0]} V" if fails else "")
            + ")",
            log,
            float(fails[0]) if fails else None,
            deck_file,
        )
    if RESCUED in log:
        raise CharonError(
            f"charon_mp.exe used an unconverged linear solve on {deck_file} ({RESCUED!r})",
            log,
        )
    return RunLog(deck_file, interp, log)


def newton_steps(log: str) -> list[StepInfo]:
    """Per converged solve of the log: its iterations and its final F-norm and WRMS norm.

    A solve is a steady state or one LOCA step.
    """
    out = []
    blocks = log.split("-- Final Status Test Results --")
    for i, block in enumerate(blocks[1:]):
        if not block.lstrip().startswith("Converged"):
            continue
        f = re.search(r"F-Norm = ([0-9.e+-]+)", block)
        w = re.search(r"WRMS-Norm = ([0-9.e+-]+)", block)
        steps = re.findall(r"-- Nonlinear Solver Step (\d+) --", blocks[i])
        out.append(StepInfo(int(steps[-1]) if steps else 0,
                            float(f.group(1)) if f else None,
                            float(w.group(1)) if w else None))  # fmt: skip
    return out
