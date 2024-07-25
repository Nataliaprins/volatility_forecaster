
from setuptools import setup

setup(
    
    name="XXX",
    version="0.1",
    author="Natalia Acevedo Prins",
    author_email="natalia.acevedop@udea.edu.co",
    license="MIT",
    url="",
    description="",
    long_description="",
    keywords="",
    platforms="any",
    provides=["XXX"],
    install_requires=[
        "solo librerias de la herramienta",
    ],
    packages=[
        "techminer2._core.metrics (ejemplo)",
        "src",
        "src.deep_learning",
        "solo carpetas de la herramienta",
    ],
    package_dir={"XXX": "src"},
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

