# Contributing to Overflow

Thank you for your interest in contributing to Overflow! This document provides guidelines and instructions for contributing to the project.

## Getting Started

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/overflow.git
   cd overflow
   ```

3. Create a development environment:
   ```bash
   conda create -n overflow-dev python gdal -c conda-forge
   conda activate overflow-dev
   ```

4. Install the package in development mode:
   ```bash
   uv pip install -e .[dev,docs]
   ```

A devcontainer is also provided with this repository that provides a pre-configured development environment.

### Running Tests

Run the test suite with pytest:
```bash
pytest
```

### Code Quality

We use several tools to maintain code quality:

- **Ruff**: For linting and code formatting
  ```bash
  ruff check
  ruff format
  ```

- **Mypy**: For type checking
  ```bash
  mypy src/ tests/
  ```

Format the code and run all checks before submitting:
```bash
ruff format
ruff check
mypy src/ tests/
pytest
```

## How to Contribute

### Reporting Bugs

Before submitting a bug report:
- Check the [issue tracker](https://github.com/fema-ffrd/overflow/issues) to see if the issue has already been reported
- Try to reproduce the issue with the latest version of Overflow

When submitting a bug report, include:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs actual behavior
- Overflow version, Python version, and OS
- Relevant code snippets or error messages
- Sample data if applicable (or minimal reproducible example)

### Suggesting Enhancements

Enhancement suggestions are welcome! Please:
- Use the [issue tracker](https://github.com/fema-ffrd/overflow/issues) with the "enhancement" label
- Provide a clear description of the proposed feature
- Explain the use case and potential benefits
- Consider implementation complexity and performance implications

### Pull Requests

1. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards:
   - Write clear, descriptive commit messages
   - Add tests for new functionality
   - Update documentation as needed
   - Ensure all tests pass
   - Run linting and type checking

3. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add feature: description of your changes"
   ```

4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request** on GitHub:
   - Provide a clear description of the changes
   - Reference any related issues
   - Describe how you tested the changes
   - Include before/after examples if applicable

### Pull Request Guidelines

- **Keep PRs focused**: One feature or bug fix per PR
- **Write tests**: All new code should include tests
- **Update docs**: Update documentation for API changes
- **Pass CI checks**: Ensure all automated checks pass
- **Respond to feedback**: Address review comments promptly
- **Maintain compatibility**: Avoid breaking changes when possible

## Coding Standards

### Python Style

- Follow PEP 8 style guidelines (enforced by Ruff)
- Use type hints for function signatures
- Write descriptive variable and function names
- Keep functions focused and concise

### Documentation

- Use Google-style docstrings for all public functions and classes
- Include type information in docstrings
- Provide usage examples for complex functions
- Update the user guide for new features

Example docstring:
```python
def flow_accumulation(fdr_path: str, output_path: str) -> None:
    """Calculate flow accumulation from flow direction.

    Args:
        fdr_path: Path to the flow direction raster file.
        output_path: Path where the output raster will be saved.

    Raises:
        FileNotFoundError: If fdr_path does not exist.
        ValueError: If flow direction values are invalid.

    Example:
        >>> flow_accumulation("flowdir.tif", "flowacc.tif")
    """
```

### Testing

- Write unit tests for all new functions
- Use pytest fixtures for common test data
- Test edge cases and error conditions
- Aim for high code coverage (>80%)
- Include integration tests for workflows

### Commit Messages

Follow these conventions:
- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and PRs when applicable

Good commit messages:
```
Add tiled flow accumulation algorithm

Implement parallel flow accumulation using graph-based
approach that maintains correctness across tile boundaries.

Fixes #123
```

## Development Workflow

### Typical Workflow

1. Check existing issues or create a new one
2. Discuss approach if it's a significant change
3. Create a feature branch
4. Implement your changes with tests
5. Run code quality checks
6. Submit a pull request
7. Address review feedback
8. Merge after approval

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test improvements

## Documentation

Documentation is built with MkDocs Material. To build and preview locally:

```bash
mkdocs serve
```

Visit http://127.0.0.1:8000 to preview changes.

## Release Process

Releases are handled by maintainers following semantic versioning:
- **Major** (X.0.0): Breaking changes
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes, backward compatible

## Questions?

- Open an issue for bug reports or feature requests
- Check the [documentation](https://fema-ffrd.github.io/overflow/) for detailed information

## License

By contributing to Overflow, you agree that your contributions will be licensed under the MIT License.

