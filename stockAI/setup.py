import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stockair", # Replace with your own username
    version="0.0.1",
    author="Eunsu Kim, Sieun Kim, Eunji Cha, Yujin Cha",
    author_email="stockai2023@gmail.com",
    description="Make your stock investment smarter, join StockAI!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stockAI-py/stockAI",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
