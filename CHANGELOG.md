# Changelog

All notable changes to the [crane-controller] project will be documented in this file.<br>
The changelog format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Changed

* Reduced repository Ruff violations to zero by tightening annotations, logging, and helper script structure.
* Cleaned agent, wrapper, and environment modules to satisfy stricter linting and formatting requirements.
* Added type annotations across all source, test, and script modules — pyright errors reduced from 54 to 0, warnings from 188 to 106.
* Parameterized `gym.Env` and wrapper base classes with concrete observation and action types.
* Narrowed `crane` parameter from `Callable[[], object]` to `Callable[..., Crane]` and typed `wire` as `Wire`.
* Used `TYPE_CHECKING` guarded imports for types only needed at check time.
* Added type stubs for `torch`, `matplotlib`, and `stable-baselines3` under `stubs/` — pyright warnings reduced from 106 to 77, informations from 15 to 8.
* Removed 7 stale `# type: ignore` comments that became unnecessary with the new stubs.

### Fixed

* Restored deterministic AntiPendulum environment seeding and aligned reset/step behavior with the existing environment tests.


## [0.0.2] - YYYY-MM-DD

### Changed
* ...


## [0.0.1] - YYYY-MM-DD

* Initial release

### Added

* added this

### Changed

* changed that

### Dependencies

* updated to some_package_on_pypi>=0.1.0

### Fixed

* fixed issue #12345

### Deprecated

* following features will soon be removed and have been marked as deprecated:
  * function x in module z

### Removed

* following features have been removed:
  * function y in module z


<!-- Markdown link & img dfn's -->
[unreleased]: https://github.com/dnv-opensource/crane-controller/compare/v0.0.2...HEAD
[0.0.2]: https://github.com/dnv-opensource/crane-controller/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/dnv-opensource/crane-controller/releases/tag/v0.0.1
[crane-controller]: https://github.com/dnv-opensource/crane-controller
