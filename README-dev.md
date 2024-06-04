# Develper Guide for Maximo Visual Inspection (MVI) Validator



[MVI Validator](https://github.com/IBM/mvi-validator) is an accuracy validator for Maximo Visual Inspection.



## Getting Started



### Prerequisites

```sh
$ pip3 install --upgrade pip wheel
```



### Installing

```sh
$ git clone git@github.com:IBM/mvi-validator.git
$ cd mvi-validator
$ pip3 install -e '.[dev]'
```



## Running Tests

The test code is written in [RSpec](https://rspec.info) style using [USpec](https://github.com/MountainField/uspec) and [PyHamcrest](https://github.com/hamcrest/PyHamcrest). But to run the tests you just execute `unittest`  that is built in test framework

```sh
$ python3 -m unittest discover -v -s src/tests -p '*_spec.py'
```



## Formatting Code

```sh
$ yapf --style='{column_limit: 9999}' -r  -i src 
```



## Build

```sh
$ rm -rf dist && python3 -m build
```



# Release

```sh
VERSION="$(cat setup.cfg | grep version | cut -d ' ' -f 3)" && echo $VERSION

gh release create v${VERSION} --generate-notes && gh release upload v${VERSION} dist/*
```

