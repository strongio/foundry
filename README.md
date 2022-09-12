# Foundry

A helpful library of functions useful for predictive analytics, ML modelling, and ML evaluation. 

Foundry includes:
- a selection of torch-accelerated GLM models, in `foundry.glm`
- quality-of-life extensions to `scikit-learn`. Most notably in `foundry.preprocessing` 
- model evaluation (and visualizations) in `foundry.evaluation`
- and more to come!
 
## Installation
This is typically installed using `setuptools` as a dependency for other modelling efforts. For example, one might have the following `setup.py` 

``` python
from setuptools import setup

setup(
    name='waimea_bay',
    version="0.1.0",
    description='Wave height predictions for Waimea Bay, Hawaii',
    author='Strong Analytics',
    packages=setuptools.find_packages(include='waimea_bay.*'),
    zip_safe=False,
    install_requires=[
        'foundry @ git+https://github.com/strongio/foundry',
        'matplotlib',
        'pandas>=1.0',
        'scikit-learn>=1.0.0',
    ],
    extras_require={
        'tests': [
            'pytest',
        ]
    },
    python_requires='>=3.6'
)
```


### For development in `foundry`

- Create a venv. `conda` is a safe choice. 
  ```console
  conda create -n waimea_bay_venv python=3.8.13
  conda activate waimea_bay_venv
  ```
- Clone this repo, and `cd` into `foundry`
  ```console
  cd ~
  git clone git+ssh://git@github.com/strongio/foundry
  cd foundry
  ```
- Install `foundry` into the virtualenv
  ```console
  pip install -e ".[dev]"
  ```
- Run tests!
  ```console
  pytest .
  ```


