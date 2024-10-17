# Import required functions
from setuptools import find_packages, setup

# Call setup function
setup(
    author="Shubham Kumar",
    description="A complete package for implementing House Price Prediction using regression.",
    name="HousePricePrediction",
    version="0.1.0",
    package_dir={
        "": "src"
    },  # Tell setuptools to look for packages in the src directory
    packages=find_packages(where="src"),
)
