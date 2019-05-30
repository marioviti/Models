from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='PytorchModels',
    url='https://github.com/marioviti/Models',
    author='Mario Viti',
    author_email='vitimario2@gmail.com',
    # Needed to actually package something
    packages=['models'],
    # Needed for dependencies
    install_requires=['numpy'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='An example of a python package from pre-existing code',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)
