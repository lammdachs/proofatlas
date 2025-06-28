# Contributing to ProofAtlas

We welcome contributions to ProofAtlas! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/proofatlas.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Set up the development environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

## Development Process

### Code Style

- We use Black for Python code formatting
- Run `black src/` before committing
- Follow PEP 8 guidelines
- Use type hints where possible
- Write docstrings for all public functions and classes

### Testing

- Write tests for new features
- Ensure all tests pass: `pytest tests/`
- Aim for >80% code coverage
- Run tests with: `pytest --cov=proofatlas tests/`

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb in present tense (e.g., "Add", "Fix", "Update")
- Reference issues when applicable: "Fix #123: Description"

Example:
```
Add GNN-based clause selection module

- Implement graph construction from clauses
- Add attention mechanism for relevance scoring
- Include unit tests and documentation
```

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Update the documentation
3. Ensure all tests pass
4. Request review from maintainers
5. Address review comments
6. Squash commits if requested

## Adding New Features

### New Models

1. Add implementation in `src/foreduce/models/`
2. Create configuration in `configs/model/`
3. Add tests in `tests/models/`
4. Update documentation

### New Parsers

1. Implement parser in `src/foreduce/parsers/`
2. Add parser tests with example files
3. Document the format supported

### New Datasets

1. Add dataset class in `src/foreduce/training/datasets/`
2. Include preprocessing scripts
3. Document data format and structure

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Respect differing opinions

## Questions?

Feel free to open an issue for:
- Bug reports
- Feature requests
- Questions about the codebase
- Discussion of research ideas

Thank you for contributing to ProofAtlas!