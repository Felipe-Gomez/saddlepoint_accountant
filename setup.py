import setuptools

version = '0.1'

with open('README.md') as f:
    long_description = f.read()

setuptools.setup(
    name='SaddlePoint',
    version=version,
    description='An algorithm to compose privacy guarantees of differentially private (DP) mechanisms.', 
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Felipe-Gomez/saddlepoint_accountant',
    author='Felipe Gomez',
    packages=["saddlepoint_accountant","saddlepoint_accountant.other_accountants"],
    python_requires=">=3.7.0",
    include_package_data=True,
    extras_require={
        "extra": [
            "jupyter",
        ]
    },
    install_requires=[
        "scipy>=1.8.0",
        "numpy",
        "mpmath",
        "sympy"
    ],
    zip_safe=False
)
