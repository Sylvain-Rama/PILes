from setuptools import find_packages, setup
setup(name="PILes",
      version="0.1.0",
      description="PIL improvement for drawing multiple geometric shapes.",
      install_requires=["matplotlib>=3.5.1", "setuptools==61.2.0", "numpy>=1.21.5", "pillow>=6.2.0"],
      author="Sylvain Rama",
      author_email='rama.sylvain@gmail.com',
      platforms=["any"],
      license="MIT",
      url="",
      packages=find_packages(),
      )
      
      
      