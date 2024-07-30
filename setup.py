from setuptools import setup, find_packages

setup(
    name='rbyte',
    version='0.0.1',
    description='Multimodal datasets for spatial intelligence',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yaak-ai/rbyte',
    author='Yaak',
    author_email='rbyte@yaak.ai',
    license='MIT',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
