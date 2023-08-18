from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    This function will return the list of requirements 
    based on requirements.txt file
    '''
    requirements = []

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n','') for req in requirements]

    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
name = 'ml_end_to_end_project',
version = '0.0.1',
author = 'Bhujith Madav V',
author_email = 'bhujith10@gmail.com',
packages = find_packages(),
install_requires = get_requirements('requirements.txt')
)