import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spell_corrector_pt",
    version="0.0.1",
    author="Samuel Heinrichs",
    author_email="ti.samuelh@gmail.com",
    description="This is a simple tool to correct portuguese misspells automatically.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/samuelhei/spell-corrector-pt",
    install_requires=[r.split('==')[0] for r in open("requirements.txt").read().split("\n")],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)