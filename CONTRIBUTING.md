# Contributing to AI-NIDS

First off, thank you for considering contributing to AI-NIDS! It's people like you that make AI-NIDS such a great tool for network security.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [security@ai-nids.org](mailto:security@ai-nids.org).

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Virtual environment (recommended)
- Basic understanding of Flask and machine learning concepts

### Setting Up Your Development Environment

1. **Fork the repository**
   ```bash
   # Click the 'Fork' button on GitHub
   ```

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/ai-nids.git
   cd ai-nids
   ```

3. **Create a virtual environment**
   ```bash
   python -m venv .venv
   
   # Windows
   .\.venv\Scripts\activate
   
   # Linux/macOS
   source .venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

5. **Set up the database**
   ```bash
   flask db upgrade
   flask seed-data  # Optional: Add sample data
   ```

6. **Run the development server**
   ```bash
   python run.py
   ```

## How Can I Contribute?

### 🐛 Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates.

**When reporting a bug, include:**
- A clear and descriptive title
- Steps to reproduce the behavior
- Expected behavior
- Actual behavior
- Screenshots (if applicable)
- Environment details (OS, Python version, browser)
- Relevant log output

**Use this template:**
```markdown
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment:**
 - OS: [e.g., Windows 11, Ubuntu 22.04]
 - Python Version: [e.g., 3.11.5]
 - Browser: [e.g., Chrome 120, Firefox 121]

**Additional context**
Add any other context about the problem here.
```

### 💡 Suggesting Enhancements

We welcome feature suggestions! Please include:

- A clear title and description
- The motivation for this feature
- How it would work
- Any relevant examples or mockups

### 🔧 Pull Requests

1. **Create a branch** for your feature or fix:
   ```bash
   git checkout -b feature/amazing-feature
   # or
   git checkout -b fix/bug-description
   ```

2. **Make your changes** following our [Style Guidelines](#style-guidelines)

3. **Test your changes**:
   ```bash
   pytest tests/
   ```

4. **Commit your changes**:
   ```bash
   git commit -m "feat: add amazing feature"
   # or
   git commit -m "fix: resolve bug in detection engine"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/amazing-feature
   ```

6. **Open a Pull Request** on GitHub

## Pull Request Process

1. Ensure your PR description clearly describes the problem and solution
2. Include the relevant issue number if applicable
3. Update documentation as needed
4. Add tests for new features
5. Ensure all tests pass
6. Request review from maintainers

### PR Title Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

## Style Guidelines

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: 100 characters max
- **Indentation**: 4 spaces
- **Imports**: Grouped (stdlib, third-party, local)
- **Docstrings**: Google style

```python
def detect_intrusion(flow_data: dict, model: str = 'ensemble') -> dict:
    """
    Analyze network flow for potential intrusions.
    
    Args:
        flow_data: Dictionary containing network flow features.
        model: The ML model to use for detection.
    
    Returns:
        Dictionary with detection results and confidence scores.
    
    Raises:
        ValueError: If flow_data is missing required fields.
    """
    pass
```

### JavaScript Style Guide

- Use ES6+ features
- 2 spaces for indentation
- Use `const` and `let`, avoid `var`
- Use template literals for string interpolation

### HTML/CSS Style Guide

- Use semantic HTML5 elements
- BEM naming convention for CSS classes
- CSS custom properties for theming

### Git Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters
- Reference issues and pull requests liberally after the first line

## Development Areas

### Core Areas

| Area | Description | Skills Needed |
|------|-------------|---------------|
| Detection Engine | ML-based intrusion detection | Python, scikit-learn, PyTorch |
| Web Dashboard | Flask-based UI | Flask, JavaScript, CSS |
| API | REST API endpoints | Flask, API design |
| Data Pipeline | Network traffic processing | Python, networking |
| Explainability | SHAP/LIME integration | ML interpretability |

### Good First Issues

Look for issues labeled `good first issue` - these are great for newcomers!

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_detection.py

# Run with verbose output
pytest -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Use descriptive test function names
- Include docstrings explaining what's being tested

## Documentation

- Update README.md for significant changes
- Add docstrings to all public functions
- Update API documentation when endpoints change
- Include examples in documentation

## Community

### Getting Help

- 📖 [Documentation](./docs/)
- 💬 [Discussions](https://github.com/ai-nids/discussions)
- 🐛 [Issue Tracker](https://github.com/ai-nids/issues)

### Acknowledgments

Contributors are recognized in our [CONTRIBUTORS.md](CONTRIBUTORS.md) file!

---

Thank you for contributing to AI-NIDS! 🛡️🤖
