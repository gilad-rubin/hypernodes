import glob
import os

import pytest


def find_notebooks(directory):
    """Find all .ipynb files in the given directory and its subdirectories."""
    return glob.glob(os.path.join(directory, "**", "*.ipynb"), recursive=True)


# Find all notebook files in tests/nodes and its subdirectories
notebook_files = find_notebooks("tests/nodes")


@pytest.mark.parametrize("notebook", notebook_files)
def test_notebook(notebook):
    """Run the notebook as a test."""
    pytest.main(["--nbval", notebook])


if __name__ == "__main__":
    pytest.main([__file__])
