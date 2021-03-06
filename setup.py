import setuptools

setuptools.setup(
    name="coremotiontools",
    version="1.0.0",
    author="Shakshi3104",
    description="motion-coremltools is the wrapper tool for converting neural networks trained with motion sensor data.",
    url="https://github.com/Shakshi3104/motion-coremltools",
    packages=setuptools.find_packages(where='src'),
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],
    python_requires=">=3.6, <4",
    package_dir={'': 'src'},
)