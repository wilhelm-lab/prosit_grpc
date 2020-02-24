# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['prosit_grpc']

package_data = \
{'': ['*']}

install_requires = \
['grpcio>=1.26.0,<2.0.0',
 'numpy>=1.18.1,<2.0.0',
 'scikit-learn>=0.21.3,<0.22.0',
 'tensorflow-serving-api>=1.15.0,<2.0.0',
 'tensorflow>=1.15.2,<2.0.0']

setup_kwargs = {
    'name': 'prosit-grpc',
    'version': '1.0.1',
    'description': 'A Client to access Prosit via GRPC',
    'long_description': None,
    'author': 'Daniela Andrade Salazar',
    'author_email': 'danial.andrade@tum.de',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.6,<4.0',
}


setup(**setup_kwargs)

# This setup.py was autogenerated using poetry.
