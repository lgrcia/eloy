# mypy: ignore-errors

import nox

ALL_PYTHON_VS = ["3.10", "3.11", "3.12"]


@nox.session(python=ALL_PYTHON_VS)
def test(session):
    session.install(".[test]")
    session.run("pytest", "-n", "auto", *session.posargs)


@nox.session
def docs(session):
    session.install(".[docs]")
    with session.chdir("docs"):
        session.run(
            "python",
            "-m",
            "sphinx",
            "-T",
            "-E",
            "-W",
            "--keep-going",
            "-b",
            "dirhtml",
            "-d",
            "_build/doctrees",
            "-D",
            "language=en",
            ".",
            "_build/dirhtml",
        )
