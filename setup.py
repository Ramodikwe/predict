from setuptools import setup, find_packages

setup(
    name='seven_functions',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='EDSA Eskom predict',
    long_description=open('README.md').read(),
    install_requires=['numpy','pandas'],
    url='https://github.com/<username>/<package-name>',
    author='Lawrence Tjatjie',
    author_email='ltjatjie@gmail.com'
)