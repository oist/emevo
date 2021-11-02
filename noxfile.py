import nox

SOURCES = ["src/emevo", "tests", "examples"]


@nox.session(reuse_venv=True, python=["3.8", "3.9"])
def tests(session: nox.Session) -> None:
    session.install("setuptools", "--upgrade")
    session.install(".")
    session.install("pytest")
    session.run("pytest", "tests")


@nox.session(reuse_venv=True)
def lint(session: nox.Session) -> None:
    session.install("black")
    session.install("flake8")
    session.install("flake8-bugbear")
    session.install("isort")
    session.run("flake8", *SOURCES)
    session.run("black", *SOURCES, "--check")
    session.run("isort", *SOURCES, "--check")


@nox.session(reuse_venv=True)
def format(session: nox.Session) -> None:
    session.install("black")
    session.install("isort")
    session.run("black", *SOURCES)
    session.run("isort", *SOURCES)
