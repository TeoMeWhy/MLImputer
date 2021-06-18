from setuptools import setup, find_packages

with open("requirements.txt", "r") as open_file:
    requirements = open_file.read()

requirements = requirements.split("\n")

setup(
      name='ML Imputer',
      version='0.0.1',
      description='Machine Learning imputation',
      author='Téo Calvo',
      author_email='teo.bcalvo@gmail.com',
      url='https://github.com/TeoMeWhy/MLImputer',
      packages=find_packages(),
      install_requires=requirements,
)