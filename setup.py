from distutils.core import setup

setup(name='cars_analysis_tb',
      version='0.1',
      description='',
      author='Priyank Shah',
      author_email='priyank.shah@kcl.ac.uk',
      license='MIT',
      packages=['cars_analysis_tb', 'cars_analysis_tb.preprocess', 'cars_analysis_tb.HyperspectralViewer',
                'cars_analysis_tb.MiscTools', 'cars_analysis_tb.CarsNN', 'cars_analysis_tb.kramers_kronig'],
      include_package_data=True)
