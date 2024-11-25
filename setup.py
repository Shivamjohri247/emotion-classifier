from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="emotion_classifier",
    version="0.1.0",
    author="ShivamJohri",
    author_email="your.email@example.com",
    description="A production-grade emotion classification system using Hugging Face Transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ShivamJohri/emotion_classifier",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "pytest-mock>=3.6",
            "httpx>=0.23.0",
            "pytest-asyncio>=0.19.0"
        ],
    },
    entry_points={
        "console_scripts": [
            "emotion-train=emotion_classifier.main:main",
            "emotion-predict=emotion_classifier.inference:predict_emotion",
            "emotion-api=emotion_classifier.api:start_server",
        ],
    },
) 