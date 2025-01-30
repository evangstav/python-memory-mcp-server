# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Fuzzy search capability for knowledge graph queries
  - New `SearchOptions` class for configuring search behavior
  - Configurable similarity threshold and field weights
  - Backward compatible with existing exact matching
  - Improved search relevance with weighted scoring

## [0.2.0] - 2024-01-07

### Added
- Observation management system with atomic operations
- Type-safe observation handling in Backend interface
- Pre-commit hooks for code quality
- EditorConfig for consistent styling
- Changelog tracking
- Documentation improvements

### Changed
- Enhanced project structure with additional directories
- Improved test suite with proper type validation
- Updated MCP tool handlers with consistent response formats

### Fixed
- Entity serialization in test responses
- TextContent validation in MCP handlers
- Error message format consistency

## [0.1.4] - 2024-01-07

### Added
- Pre-commit hooks for code quality
- EditorConfig for consistent styling
- Changelog tracking
- Documentation improvements

### Changed
- Enhanced project structure with additional directories

## [0.1.0] - 2024-01-07

### Added
- Initial release
- JSONL backend implementation
- Knowledge graph management
- MCP server implementation
- Basic test suite
