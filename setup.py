from setuptools import setup

setup(name='czitools',
      version='0.0.1',
      description='Read and analyze CZI (and OME-TIFF) files',
      long_description='Collection of functions and tool to read and analyze CZI (and OME-TIFF) files',
      
      keywords='image analysis czi ome-tiff file read segmentation',
      url='http://github.com/storborg/funniest',
      author='Sebastian Rhode',
      author_email='sebastian.rhode@zeiss.com',
      license='BSD',
      packages=['funniest'],
      
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.7',
      ],
      
      install_requires=[
          'scikit-image>=0.16.2',
      ],
      include_package_data=True,
      zip_safe=False)