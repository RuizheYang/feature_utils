from setuptools import setup
from setuptools import find_packages

setup(name='nice_feat',
      version='0.0.1',
      description='feature engineering utils',
      author='The fastest man alive.',
      packages=find_packages(),
      include_package_data=True,
      install_requires=["redshift-connector","psycopg2","thrift","thrift-sasl","PyHive","pydantic<2.0.0"])

