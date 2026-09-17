# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to Rust's notion of
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- `computed-generators` feature (off by default). When enabled, `HashDomain::hash_to_point`
  computes each generator `S_j` from its defining hash-to-curve construction instead of
  reading the precomputed coordinate table. The public `SINSEMILLA_S` constant remains
  exported; the feature is API-additive but selects a slower internal path for every user
  of the crate in that build.

## [0.1.0] - 2024-12-13
Initial release, extracted from `halo2_gadgets 0.3.0`. Includes minor changes
for `no-std` support.
