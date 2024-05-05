from setuptools import find_packages, setup

setup(
    name="src",
    packages=find_packages(),
    version="0.1.0",
    description="Quick mock up of a LLM agent for predicting hydration free"
    " energy for molecules when given a smile string.",
    author="Daniel Cutting",
    license="MIT",
)
