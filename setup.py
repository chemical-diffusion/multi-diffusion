from distutils.core import setup
try:
    import setuptools
except ImportError:
    pass
import multidiff

setup(name='multidiff',
      version='0.2',
      packages=['multidiff'],
      author='Emmanuelle Gouillart',
      author_email='emmanuelle.gouillart@gmail.com',
      url='https://github.com/chemical-diffusion/multi-diffusion',
      description="A fitting package for the analysis of multicomponent diffusion data",
      install_requires=[
          "numpy",
          "scipy",
          "matplotlib",
          "derivative"
      ],
      )

