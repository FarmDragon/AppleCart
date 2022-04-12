# AppleCart


## Installation
[Install drake](https://drake.mit.edu/installation.html)
[Install poetry](https://python-poetry.org/docs/master/#installing-with-the-official-installer)
[Download the underactuated robotics companion code](https://github.com/RussTedrake/underactuated)

```sh
poetry install
```

Add two files to the virtual env that poetry made to tell it about the path to the drake and underactuated python modules.

```
echo $DRAKE_INSTALL/lib/python3.9/site-packages $VENV/lib/python3.9/site-packages/drake.pth
echo $UNDERACTUATED_REPO $VENV/lib/python3.9/site-packages/underactuated.pth
```

Where `DRAKE_INSTALL` is the directory where drake is installed, `UNDERACTUATED_REPO` is the directory of the underactuated repo, and `VENV` is the location of the virtual environment poetry made.

## Running

### From the command line

```
poetry run python triple_cart_pole.py
```

or

```
poetry shell
python triple_cart_pole.py
```

### From VS Code

Open `triple_cart_pole.py` click run, and select the virtual environment poetry created for this project as the interpreter.