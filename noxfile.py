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
    session.run(
        "pip-compile",
        "pyproject.toml",
        "--output-file",
        "requirements/self.txt",
    )
    for path in requirements_dir.glob("*.in"):
        txt_file = f"requirements/{path.stem}.txt"
        session.run("pip-compile", path.as_posix(), "--output-file", txt_file)


@nox.session(reuse_venv=True)
def format(session: nox.Session) -> None:
    _sync(session, "requirements/format.txt")
    session.run("black", *SOURCES)
    session.run("isort", *SOURCES)


@nox.session(reuse_venv=True, python=["3.8", "3.9", "3.10"])
def lint(session: nox.Session) -> None:
    _sync(session, "requirements/lint.txt")
    session.run("flake8", *SOURCES)
    session.run("black", *SOURCES, "--check")
    session.run("isort", *SOURCES, "--check")


@nox.session(reuse_venv=True)
def smoke(session: nox.Session) -> None:
    """Run a smoke test"""
    _sync(session, "requirements/smoke.txt")
    session.install("-e", ".")
    DEFAULT = "smoke-tests/circle_loop.py"
    if 0 < len(session.posargs) and session.posargs[0].endswith(".py"):
        session.run("python", *session.posargs)
    else:
        session.run("python", DEFAULT, *session.posargs)


@nox.session(reuse_venv=True, python=["3.8", "3.9", "3.10"])
def tests(session: nox.Session) -> None:
    _sync(session, "requirements/tests.txt")
    session.install("-e", ".")
    session.run("pytest", "tests", *session.posargs)
