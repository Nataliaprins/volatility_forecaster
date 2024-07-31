
from setuptools import setup

setup(
    
    name="volatility_forecaster",
    version="0.1",
    author="Natalia Acevedo Prins",
    author_email="natalia.acevedop@udea.edu.co",
    license="MIT",
    url="",
    description="",
    long_description="",
    keywords="",
    platforms="any",
    provides=["volatility_forecaster"],
    install_requires=[
        #"solo librerias de la herramienta",
        "numpy",
        "plotly",
        "scikit-learn",
        "scipy",
        "wheel",
        "pandas",
        "yfinance",
        "keras",
        "tensorflow",
    ],
    packages=[
        
        "volatility_forecaster",
        "volatility_forecaster.deep_learning",
        #"solo carpetas de la herramienta",
    ],
    package_dir={"volatility_forecaster": "volatility_forecaster"},
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)

