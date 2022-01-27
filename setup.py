from setuptools import setup

setup(
    name='contentanalysis',
    url='https://github.com/hannesrosenbusch/contentanalysis',
    author='Hannes Rosenbusch',
    author_email='',
    packages=['contentanalysis'],
    package_dir={'contentanalysis': 'src/contentanalysis'},
    package_data={'contentanalysis': ['src/contentanalysis/data/*']},
    install_requires=['fasttext', 'sentence_transformers'],
    version='0.1',
    license='MIT',
    include_package_data=True,
    description='quickie classification and analyses'
)

      
      