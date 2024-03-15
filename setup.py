#!/usr/bin/env python
from setuptools import setup, find_packages

print(find_packages)
if __name__ == "__main__":
    setup(
    name="diag-pred-from-sulci",
    version="0.0.1",
    packages=find_packages(exclude=['tests*', 'notebooks*']),
    license='CeCILL license version 2',
    description='Deep learning models '
                'to predict diagnosis from cortical sulci.',
    install_requires=["numpy",
                      "matplotlib",
                      "torch",
                      "torchvision",
                      "scikit-learn",
                      "scikit-multilearn",
                      "scikit-image",
                      "statsmodels",
                      "scipy",
                      "pandas",
                      "typing-extensions",
                      "tabulate",
                      "tqdm",
                      "iterative-stratification"],
    extras_require={"wandb": ["wandb", "plotly"],
                    "figures": ["notebook", "nilearn", "seaborn", "nibabel"]},
    author="Pierre Auriau and Benoit Dufumier"
    )