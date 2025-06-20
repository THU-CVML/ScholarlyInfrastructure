"""nbscholar (new-big scholar, 大牛学者) is a extension for better NoteBook development, extending fastai's `nbdev` libary. It is designed to assist you become a New-Big (a.k.a. 牛逼 or awesome) scholar one day."""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../notebooks/03_nbscholar (nbdev extensions).ipynb.

# %% auto 0
__all__ = [
    "read_settings_ini_none",
    "read_settings_ini",
    "nbscholar_export",
    "guess_notebooks_path",
    "split_import_and_code_cells",
    "operate_on_notebook_in",
    "process_notebooks_in_folder",
    "nbscholar_separate",
]

# %% ../notebooks/03_nbscholar (nbdev extensions).ipynb 4
from fastcore.script import call_parse

# %% ../notebooks/03_nbscholar (nbdev extensions).ipynb 9
import configparser
import os
from pathlib import Path


# %% ../notebooks/03_nbscholar (nbdev extensions).ipynb 10
def read_settings_ini(
    directory, item="nbs_path", track="DEFAULT", ini_name="settings.ini"
):
    config = configparser.ConfigParser()
    settings_path = os.path.join(directory, ini_name)
    assert os.path.exists(directory), f"Directory {directory} does not exist"
    assert os.path.exists(settings_path), f"Could not find {ini_name} in {directory}"
    config.read(settings_path)
    assert track in config, f"Could not find {track} in {settings_path}"
    assert item in config[track], f"Could not find {item} in {settings_path}"
    return config[track][item]


# %% ../notebooks/03_nbscholar (nbdev extensions).ipynb 11
import subprocess
import os
from .logging.nucleus import logger


@call_parse
def nbscholar_export(path: str = "."):
    res = os.system("nbdev_export")
    if res != 0:
        raise Exception("nbdev_export failed")
    # 读取 settings.ini 的 lib_name
    lib_name = read_settings_ini(path, item="lib_name")

    # MKINIT 生成 __init__.py
    # res = os.system(f"mkinit {lib_name} -w --lazy_loader --recursive --relative")
    res = os.system(f"mkinit {lib_name} -w --lazy_loader_typed --recursive --relative")
    if res != 0:
        # raise Exception("mkinit failed")
        logger.warning("mkinit failed")

    # RUFF 格式化
    res = os.system(f"ruff format {lib_name}")
    if res != 0:
        logger.warning("ruff format failed")


# %% ../notebooks/03_nbscholar (nbdev extensions).ipynb 15
import os
import nbformat
import re
from nbformat.notebooknode import NotebookNode, from_dict
from . import default_on_exception

# %% ../notebooks/03_nbscholar (nbdev extensions).ipynb 16
read_settings_ini_none = default_on_exception(read_settings_ini, default_value=None)


def guess_notebooks_path(directory="."):
    if isinstance(directory, Path):
        directory = directory.as_posix()
    # 读取 setting.ini 的 nbs_path， 如果有的话，没有就返回None
    return read_settings_ini_none(directory)


# %% ../notebooks/03_nbscholar (nbdev extensions).ipynb 19
from copy import deepcopy


# %% ../notebooks/03_nbscholar (nbdev extensions).ipynb 20
def split_import_and_code_cells(notebook, inplace=True):
    """
    Process a Jupyter Notebook file, splitting cells with both import and non-import lines into two cells.
    The first new cell will contain only import statements, and the second will contain the rest of the code.
    """
    notebook = notebook if inplace else deepcopy(notebook)

    new_cells = []

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            # Split the lines in the cell
            lines = cell["source"].splitlines()

            # Extract leading blank lines or lines starting with "#|"
            leading_lines = []
            while lines and (lines[0].strip() == "" or lines[0].startswith("#|")):
                leading_lines.append(lines.pop(0))

            # Separate import statements and other code lines
            import_lines = [
                line for line in lines if re.match(r"^\s*import\b|^\s*from\b", line)
            ]
            other_lines = [line for line in lines if line not in import_lines]

            if import_lines and other_lines:
                # Add the leading lines to the import cell

                new_cells.append(
                    from_dict(
                        cell
                        | {
                            "cell_type": "code",
                            "metadata": {},
                            "source": "\n".join(leading_lines + import_lines),
                            "outputs": [],
                        }
                    )
                )
                # Add the leading lines to the other code cell
                new_cells.append(
                    from_dict(
                        cell
                        | {
                            "cell_type": "code",
                            "metadata": {},
                            "source": "\n".join(leading_lines + other_lines),
                            "outputs": cell["outputs"],
                        }
                    )
                )
            else:
                # If no split is needed, retain the original cell
                new_cells.append(cell)
        else:
            # Retain non-code cells as is
            new_cells.append(cell)

    # Update the notebook with the modified cells
    notebook["cells"] = new_cells
    return notebook


# %% ../notebooks/03_nbscholar (nbdev extensions).ipynb 22
def operate_on_notebook_in(
    input_path, output_path=None, operation=split_import_and_code_cells
):
    if output_path is None:
        output_path = input_path
    with open(input_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)
    notebook = operation(notebook)
    # Save the modified notebook
    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(notebook, f)


# %% ../notebooks/03_nbscholar (nbdev extensions).ipynb 23
def process_notebooks_in_folder(folder_path, operation=split_import_and_code_cells):
    """
    Traverse all .ipynb files in a folder and apply the cell-splitting logic.
    """
    if isinstance(folder_path, Path):
        folder_path = folder_path.as_posix()
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".ipynb"):
                notebook_path = os.path.join(root, file)
                print(f"Processing {notebook_path}")
                operate_on_notebook_in(notebook_path, operation=operation)


# %% ../notebooks/03_nbscholar (nbdev extensions).ipynb 24
@call_parse
def nbscholar_separate(path: str = "."):
    if os.path.isfile(path):
        return operate_on_notebook_in(path, split_import_and_code_cells)

    notebook_path = guess_notebooks_path(path)
    if notebook_path is None:
        notebook_path = path
    process_notebooks_in_folder(notebook_path)
