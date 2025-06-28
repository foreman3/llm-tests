from setuptools import setup, find_packages

setup(
    name="llm-pipelines",
    version="0.1.0",
    packages=find_packages(exclude=["examples", "examples.*"]),
    install_requires=["pandas", "openai", "numpy", "requests"],
    author="Bob Foreman",
    description="A reusable library for LLM-based pipelines.",
    url="https://github.com/foreman3/llm-tests",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
