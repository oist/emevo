from __future__ import annotations

import pathlib
import shutil
import subprocess

import nox

SOURCES = ["src/emevo", "tests", "smoke-tests", "experiments"]


def _sync(session: nox.Session, requirements: str) -> None:
    if (
        not session._runner.global_config.no_install
        or session._runner.global_config.install_only
    ):
        session.install("pip-tools")
        session.run("pip-sync", requirements)


@nox.session(reuse_venv=True)
def compile(session: nox.Session) -> None:
    session.install("pip-tools")
    requirements_dir = pathlib.Path("requirements")
    has_cuda = shutil.which("ptxas") is not None
    if has_cuda:
        nvcc_out = subprocess.run(["nvcc", "--version"], capture_output=True)
        is_cuda_12 = "cuda_12" in nvcc_out.stdout.decode("utf-8")
    else:
        nvcc_out = ""
        is_cuda_12 = False

    def _run_pip_compile(in_file: str, out_name: str) -> None:
        # If -k {out_name} is given, skip compiling
        if "-k" in session.posargs and out_name not in session.posargs:
            return

        out_file = f"requirements/{out_name}.txt"
        args = ["pip-compile"]
        if has_cuda and out_name not in ["format", "lint"]:
            if is_cuda_12:
                args.append("requirements/cuda12.in")
            else:
                args.append("requirements/cuda11.in")
        args += [
            in_file,
            "--output-file",
            out_file,
            "--resolver=backtracking",
        ]
        if "--upgrade" in session.posargs:
            args.append("--upgrade")
        session.run(*args)

    for path in requirements_dir.glob("*.in"):
        if "cuda" not in path.stem:
            _run_pip_compile(path.as_posix(), path.stem)


@nox.session(reuse_venv=True)
def bench(session: nox.Session) -> None:
    _sync(session, "requirements/bench.txt")
    session.run("pytest", "benches", *session.posargs)


@nox.session(reuse_venv=True)
def format(session: nox.Session) -> None:
    _sync(session, "requirements/format.txt")
    session.run("black", *SOURCES)
    session.run("isort", *SOURCES)


@nox.session(reuse_venv=True, python=["3.10", "3.11", "3.12"])
def lint(session: nox.Session) -> None:
    _sync(session, "requirements/lint.txt")
    session.run("ruff", "check", *SOURCES)
    session.run("black", *SOURCES, "--check")
    session.run("isort", *SOURCES, "--check")


@nox.session(reuse_venv=True)
def lab(session: nox.Session) -> None:
    _sync(session, "requirements/jupyter.txt")
    session.run("python", "-m", "ipykernel", "install", "--user", "--name", "emevo-lab")
    session.run("jupyter", "lab", *session.posargs)


@nox.session(reuse_venv=True)
def ipython(session: nox.Session) -> None:
    _sync(session, "requirements/jupyter.txt")
    session.run("python", "-m", "IPython")


@nox.session(reuse_venv=True)
def script(session: nox.Session) -> None:
    """Run scripts"""
    _sync(session, "requirements/scripts.txt")
    DEFAULT = "scripts/plot_bd_models.py"
    if 0 < len(session.posargs) and session.posargs[0].endswith(".py"):
        session.run("python", *session.posargs)
    else:
        session.run("python", DEFAULT, *session.posargs)


@nox.session(reuse_venv=True)
def smoke(session: nox.Session) -> None:
    """Run smoke tests"""
    _sync(session, "requirements/smoke.txt")
    DEFAULT = "smoke-tests/circle_loop.py"
    if 0 < len(session.posargs) and session.posargs[0].endswith(".py"):
        session.run("python", *session.posargs)
    else:
        session.run("python", DEFAULT, *session.posargs)


@nox.session(reuse_venv=True, python=["3.10", "3.11", "3.12"])
def tests(session: nox.Session) -> None:
    _sync(session, "requirements/tests.txt")
    session.run("pytest", "tests", *session.posargs)
