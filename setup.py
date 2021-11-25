from setuptools import setup, find_packages

setup(name='caslearn',
      version='0.1.0',
      description='machine learning with casadi',
      author='Huijie Tian',
      author_email='hut216@lehigh.edu',
      url='',
      packages=find_packages(),
      install_requires=[
        'python>= 3.7',
        'numpy',
        'casadi>= 3.4.2'
      ]
     )