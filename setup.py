from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="maritime-anomaly-detection",
    version="0.1.0",
    author="Marine Corps",
    author_email="jaden@korea.kr",
    description="AIS 기반 해상 이상 탐지 시스템",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maritime-ai/anomaly-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=21.0.0",
            "flake8>=4.0.0",
            "isort>=5.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "maritime-train=scripts.train:main",
            "maritime-predict=scripts.predict:main",
            "maritime-submit=scripts.submit:main",
        ],
    },
) 