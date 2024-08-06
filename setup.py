from setuptools import setup, find_packages
from distutils.util import convert_path
import platform

system = platform.system()
if system in ["Windows","Linux"]:
    torch = "torch==2.2.0+cu121"
if system == "Darwin":
    torch = "torch==2.2.0"

main_ns = {}
ver_path = convert_path('aTrain_core/version.py')
print("Version file path:", ver_path) 
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setup(
    name='aTrain_core',
    version=main_ns['__version__'],
    readme="README.md",
    license="LICENSE",
    python_requires=">=3.10",
    install_requires=[
        torch,
        "torchaudio==2.2.0",
        "faster-whisper==1.0.2",
        "transformers",
        "ctranslate2==4.2.1",
        "ffmpeg-python>=0.2",
        "pandas",
        "pyannote.audio==3.2.0",
        "huggingface-hub==0.24.5",
        "numpy==1.26.4",
        "werkzeug==3.0.3"],
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        'console_scripts': ['aTrain_core = aTrain_core.cli:cli',]
    }
)
#

