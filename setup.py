from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='dinogen',
    version='0.0.1',
    description='A dinosaur name generator.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='John Chase',
    author_email='ratteal@gmail.com',
    python_requires='>=3.5.3',
    url='https://github.com/RatJuggler/dinosaur-generator',
    packages=find_packages(exclude=['tests']),
    package_data={
        'dinogen': ['dinosaurs.csv'],
    },
    entry_points={
        'console_scripts': [
            'dinogen = dinogen.__main__:generate',
        ]
    },
    install_requires=[
        'numpy == 1.22.0'
    ],
    license='MIT',
)
