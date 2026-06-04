from setuptools import find_packages, setup

VERSION = "0.15.0"
DESCRIPTION = "tools for utilizing generative AI."
LONG_DESCRIPTION = """This package leverages the capabilities of Large Language Models (LLMs) and retrievers to \
establish a comprehensive question-answering system. Additionally, it includes tools for uploading documents and texts \
to the retriever, alongside various features for our products."""

with open('requirements.txt') as f:
    required_packages = [line.strip() for line in f.readlines()]

setup(
    name="mtn_agents",
    version=VERSION,
    author="mtn_agents",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    package_data={"mtn_agents": ["files/*"]},
    install_requires=required_packages,
)
