# noxfile.py
import tempfile

import nox

locations = "src", "tests", "noxfile.py"


def install_with_constraints(session, *args, **kwargs):
    with tempfile.NamedTemporaryFile() as requirements:
        session.run(
            "poetry",
            "export",
            "--dev",
            "--format=requirements.txt",
            f"--output={requirements.name}",
            external=True,
        )
        session.install(f"--constraint={requirements.name}", *args, **kwargs)


@nox.session(python=["3.8", "3.7"])
def tests(session):
    args = session.posargs  # or ["--cov"]
    # session.run("poetry", "install", external=True)
    session.run("poetry", "install", "--no-dev", external=True)
    install_with_constraints(
        session, "coverage[toml]", "pytest", "pytest-cov", "pytest-mock"
    )
    session.run("poetry", "run", "pytest", *args, external=True)


@nox.session(python=["3.8"])
def lint(session):
    args = session.posargs or locations
    # session.install("flake8", "flake8-bugbear", "flake8-bandit", "flake8-import-order")
    install_with_constraints(
        session,
        "flake8",
        "flake8-bandit",
        # "flake8-black",
        "flake8-bugbear",
        # "flake8-import-order",
    )
    session.run("flake8", *args)


@nox.session(python="3.8")
def black(session):
    args = session.posargs or locations
    # session.install("black")
    install_with_constraints(session, "black")
    session.run("black", *args)


@nox.session(python="3.8")
def isort(session):
    args = session.posargs or locations
    # session.install("black")
    install_with_constraints(session, "isort")
    session.run("isort", *args)


"""
@nox.session(python="3.8")
def safety(session):
    with tempfile.NamedTemporaryFile() as requirements:
        session.run(
            "poetry",
            "export",
            "--dev",
            "--format=requirements.txt",
            "--without-hashes",
            f"--output={requirements.name}",
            external=True,
        )
        session.install("safety")
        session.run("safety", "check", f"--file={requirements.name}", "--full-report")
"""
