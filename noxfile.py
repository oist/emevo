import pathlib

import nox

SOURCES = ["src/emevo", "tests", "examples"]


def _install_self(session: nox.Session) -> None:
    session.install("setuptools", "--upgrade")
    session.install(".")


@nox.session(reuse_venv=True)
def compile(session: nox.Session) -> None:
    session.install("pip-tools")
    requirements_dir = pathlib.Path("requirements")
    for path in requirements_dir.glob("*.in"):
        txt_file = f"requirements/{path.stem}.txt"
        session.run("pip-compile", path.as_posix(), "--output-file", txt_file)


@nox.session(reuse_venv=True)
def lab(session: nox.Session) -> None:
    _install_self(session)
    session.install("-r", "requirements/lab.txt")
    session.run("python", "-m", "ipykernel", "install", "--user", "--name", "emevo-lab")
    session.run("jupyter", "lab", *session.posargs)


@nox.session(reuse_venv=True)
def example(session: nox.Session) -> None:
    _install_self(session)
    DEFAULT_EXAMPLE = "examples/waterworld_evolution_loop.py"
    if 0 < len(session.posargs) and session.posargs[0].endswith(".py"):
        session.run("python", *session.posargs)
    else:
        session.run("python", DEFAULT_EXAMPLE, *session.posargs)


@nox.session(reuse_venv=True, python=["3.8", "3.9"])
def tests(session: nox.Session) -> None:
    _install_self(session)
    session.install("pytest")
    session.run("pytest", "tests")


@nox.session(reuse_venv=True, python=["3.8", "3.9"])
def lint(session: nox.Session) -> None:
    session.install("-r", "requirements/lint.txt")
    session.run("flake8", *SOURCES)
    session.run("black", *SOURCES, "--check")
    session.run("isort", *SOURCES, "--check")


@nox.session(reuse_venv=True)
def format(session: nox.Session) -> None:
    session.install("-r", "requirements/format.txt")
    session.run("black", *SOURCES)
    session.run("isort", *SOURCES)