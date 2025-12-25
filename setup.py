from setuptools import setup, find_packages

requires = [
    'flask',
]

setup (
    name= 'Devanagari Character Recognizer',
    version= '0.0',
    description='Web Platform for Devanagari Character Rcognization',
    author='Pramod Khatiwada',
    author_email='pramodkhatiwada03@gmail.com',
    keywords='Devanagari, CNN, Neural Network',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requires,
)
