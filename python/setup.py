#!/usr/bin/env python

from distutils.core import setup

setup(name='mingpipe',
      version='1.0',
      description='Chinese Name Matcher',
      author='Nanyun Violet Peng',
      author_email='npeng1@jhu.edu',
      install_requires=['sklearn', 'numpy', 'scipy'],
      url='https://github.com/hltcoe/mingpipe/',
      packages=['mingpipe', 'mingpipe.pinyin'],
      package_data={'mingpipe': ['pinyin/*.dict', 'resources/*']},
      )
