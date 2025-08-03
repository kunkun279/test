"""
基于PISL样条学习和HumanMAC扩散模型的人体动作预测项目安装脚本
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="human-motion-prediction",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="基于PISL样条学习和HumanMAC扩散模型的人体动作预测",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/human-motion-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "human-motion-train=human_motion_prediction.training.train_fusion:main",
            "human-motion-predict=human_motion_prediction.inference:main",
        ],
    },
    include_package_data=True,
    package_data={
        "human_motion_prediction": [
            "configs/*.yaml",
            "configs/*.yml",
        ],
    },
    zip_safe=False,
)