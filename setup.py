from setuptools import setup, find_packages

setup(
    name="videogen",
    version="0.1.0",
    description="Local AI Video Generation Framework",
    author="VideoGen Team",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "einops>=0.7.0",
        "pyyaml>=6.0",
        "omegaconf>=2.3.0",
        "click>=8.1.0",
        "gradio>=4.0.0",
        "tqdm>=4.66.0",
    ],
    entry_points={
        "console_scripts": [
            "videogen-train=scripts.train:main",
            "videogen-generate=scripts.generate:main",
        ],
    },
)
