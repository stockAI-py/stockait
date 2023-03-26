from setuptools import setup, find_packages

setup(
    name="stockair",   # pypi 에 등록할 라이브러리 이름
    version="0.0.0",    # pypi 에 등록할 version (수정할 때마다 version up을 해줘야 함)
    description="Make your stock investment smarter, join StockAir!",
    author="Eunsu Kim, Sieun Kim, Eunji Cha, Yujin Cha",
    author_email="stockai2023@gmail.com",
    url="https://github.com/insutance/hey-insutance",
    python_requires=">= 3.8",
    packages=find_packages(),
    install_requires=[],
    zip_safe=False,
    # 중요한 부분
    entry_points={
        "console_scripts": [
            "hey = insutance.main:main"
        ]
    },
    package_data={},
    include_package_data=True
)