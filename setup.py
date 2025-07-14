from setuptools import setup, find_packages
import os

# README 파일 읽기
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# requirements.txt 읽기
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="idea-similarity-api",
    version="1.0.0",
    author="AI Study Team",
    author_email="ai.study.team@example.com",
    description="창업 아이디어 유사도 측정 및 추천 API",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-team/idea-similarity-api",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    include_package_data=True,
    package_data={
        "idea_similarity_api": ["data/*.csv"],
    },
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "idea-api=idea_similarity_api.api_server:main",
        ],
    },
    keywords="ai, machine-learning, similarity, startup, ideas, recommendation",
    project_urls={
        "Bug Reports": "https://github.com/your-team/idea-similarity-api/issues",
        "Source": "https://github.com/your-team/idea-similarity-api",
        "Documentation": "https://github.com/your-team/idea-similarity-api#readme",
    },
) 