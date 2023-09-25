import pathlib

import nox

SOURCES = ["src/emevo", "tests", "smoke-tests"]


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

    def _run_pip_compile(in_file: str, out_name: str) -> None:
        # If -k {out_name} is given, skip compiling
        if "-k" in session.posargs and out_name not in session.posargs:
            return

        out_file = f"requirements/{out_name}.txt"
        args = [
            "pip-compile",
            in_file,
            "--output-file",
            out_file,
            "--resolver=backtracking",
        ]
        if "--upgrade" in session.posargs:
            args.append("--upgrade")
        session.run(*args)

    for path in requirements_dir.glob("*.in"):
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


@nox.session(reuse_venv=True, python=["3.9", "3.10", "3.11"])
def lint(session: nox.Session) -> None:
    _sync(session, "requirements/lint.txt")
    session.run("ruff", *SOURCES)
    session.run("black", *SOURCES, "--check")
    session.run("isort", *SOURCES, "--check")


@nox.session(reuse_venv=True)
def smoke(session: nox.Session) -> None:
    """Run a smoke test"""
    _sync(session, "requirements/smoke.txt")
    DEFAULT = "smoke-tests/circle_loop.py"
    if 0 < len(session.posargs) and session.posargs[0].endswith(".py"):
        session.run("python", *session.posargs)
    else:
        session.run("python", DEFAULT, *session.posargs)


@nox.session(reuse_venv=True, python=["3.9", "3.10", "3.11"])
def tests(session: nox.Session) -> None:
    _sync(session, "requirements/tests.txt")
    session.run("pytest", "tests", *session.posargs)
