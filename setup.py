from setuptools import setup, find_packages
from typing import List

HYPEEN_E_DOT = '-e .'
def get_requirements(file_path:str)->list[str]:
    '''    
    This function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        if HYPEEN_E_DOT in requirements:
            requirements.remove(HYPEEN_E_DOT)
    return requirements
setup(
    name='MLproject',
    version='0.1.0',
    author='Harsh-896',
    author_email='harshkumar88307@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
