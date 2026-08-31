from scripts.clean_notebooks import clean_notebook


def test_clean_notebook_removes_execution_artifacts() -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 3,
                "outputs": [{"output_type": "stream", "text": ["result\n"]}],
                "metadata": {
                    "collapsed": True,
                    "scrolled": True,
                    "trusted": True,
                    "keep": "value",
                },
                "source": ["print('result')\n"],
            },
            {
                "cell_type": "markdown",
                "metadata": {"keep": "value"},
                "source": ["# Heading\n"],
            },
        ],
        "metadata": {"widgets": {"state": {}}, "kernelspec": {"name": "python3"}},
    }

    cleaned = clean_notebook(notebook)

    assert cleaned["cells"][0]["execution_count"] is None
    assert cleaned["cells"][0]["outputs"] == []
    assert cleaned["cells"][0]["metadata"] == {"keep": "value"}
    assert cleaned["cells"][1]["metadata"] == {"keep": "value"}
    assert cleaned["metadata"] == {"kernelspec": {"name": "python3"}}
