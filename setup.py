import setuptools

version = '0.1'

with open('README.md') as f:
    long_description = f.read()

setuptools.setup(
    name='SaddlePoint',
    version=version,
    description='The saddlepoint accountant for approximating DP guarantees for the DP-SGD algorithm under various DP mechanisms', 
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Felipe-Gomez/saddlepoint_accountant',
    author='Felipe Gomez',
    packages=["saddlepoint_accountant", "saddlepoint_accountant.gaussian_mechanism"],
    python_requires=">=3.7.0",
    include_package_data=True,
    setup_requires=["numpy", "scipy", "sympy", "mpmath"],
    extras_require={
        "extra": [
            "jupyter",
        ]
    },
    install_requires=[
        "prv_accountant",
        "scipy",
        "numpy",
        "mpmath",
        "sympy"
    ],
    zip_safe=False
)
