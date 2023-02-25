# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Require Python 3.8+
- Require Tensorflow 2.2+ for non-macOS
- Require Tensorflow 2.5+ for macOS via tensorflow_macos

### Fixed

- Declares numpy dependency explicitly

### Removed

- Dropped support for Python 3.7 and older
    - 3.7 is EOL in a few months and all others are already EOL

## [1.1.1] - 2021-12-26

### Changed

- break out numpy (nd array) function
- remove classic app run modes for argparse
- turn down verbosity in image load via file

### Added

- one more example in README for running

### Fixed 

- fix requirements for clean system (needs PIL)

## [1.2.0] - 2020-05-15

### Added

- New model release

## [1.1.0] - 2020-03-03

### Changed

- update to tensorflow 2.1.0 and updated mobilenet-based model

## [1.0.0] - 2019-04-04

### Added

- initial creation
