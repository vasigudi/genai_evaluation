# genaievaluation/setup.py
from setuptools import setup, find_packages

def parse_requirements(filename):
    """Load requirements from a requirements file."""
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read requirements from requirements.txt
requirements = parse_requirements('requirements.txt')

setup(
    name='genaievaluation',
    version='0.1.0',
    description='A library for evaluating generated text using various metrics.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Vasishta Sharma Gudi',
    author_email='sharma.vasishta@gmail.com',
    url='https://github.com/vasigudi/genai_evaluation.git',
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'Programming Language :: Python :: 3',
        'Environment :: Console',
        'Private :: Internal Use Only',
    ],
    python_requires='>=3.6',
)