from setuptools import setup
import os

with open(os.path.join(os.path.dirname(__file__),
                       "requirements.txt")) as req_file:
    requirements = req_file.read().splitlines()

setup(
    name='shredsim',
    version='0.1',
    url='https://github.com/xa4a/shredsim',
    license='MIT',
    author='Ievgen Varavva',
    author_email='yvaravva@google.com',
    packages=["shredsim",
              "shredsim.classifiers"],
    package_data={"shredsim": ["shredsim/dataset/src/*", "shredsim/dataset/*.png"]},
    description='Set of tools for simulating paper shredders and evaluating document recovery approaches.',
    platforms='any',
    install_requires=requirements,
    scripts=[],
)
