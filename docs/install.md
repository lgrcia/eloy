# Installation

## pypi

The simplest way to install Eloy is through pip, Python's package installer.

```bash
pip install eloy
```

To install a specific version:

```bash
pip install eloy==1.2.3
```

To upgrade to the latest version:

```bash
pip install --upgrade eloy
```

## Virtual Environments

### Why Use Virtual Environments?

Virtual environments isolate project dependencies, preventing conflicts between packages that might require different versions of shared dependencies. They also make your projects more reproducible and portable.

### Using uv (Recommended)

`uv` is a modern, fast Python package installer and environment manager that works well with Eloy.

#### Install uv
```bash
pip install uv
```

#### Create a New Environment with uv
```bash
uv venv --python 3.11 # choose your version
```

#### Activate the Environment

On Windows:
```bash
.\venv\Scripts\activate
```

On Unix/MacOS:
```bash
source venv/bin/activate
```

#### Install Eloy in the Environment
```bash
uv pip install eloy
```

#### Install Development Dependencies
```bash
uv pip install eloy[dev]
```

### Alternative: Using virtualenv or venv

If you prefer traditional virtual environment tools:

```bash
# Create environment
python -m venv eloy-env

# Activate (platform-specific)
# On Windows:
eloy-env\Scripts\activate
# On Unix/MacOS:
source eloy-env/bin/activate

# Install
pip install eloy
```

## Installing from GitHub

To install the latest development version from GitHub:

```bash
pip install git+https://github.com/lgrcia/eloy.git
```

### Cloning and Installing Locally

For contributors or to access the newest features:

```bash
# Clone the repository
git clone https://github.com/lgrcia/eloy.git
cd eloy

# Install in development mode
pip install -e .
```

This creates an "editable" installation, meaning changes to the source code will be reflected immediately without reinstalling.

## Verify Installation

Let's check if Eloy is properly installed:

```python
import eloy

print(f"Eloy version: {eloy.__version__}")
```
## Troubleshooting

### Common Issues:

1. **Missing dependencies**: Ensure all required packages are installed
2. **Version conflicts**: Try using a fresh virtual environment
3. **Installation permission errors**: Use `--user` flag or virtual environments
4. **Compiler errors**: Install necessary build tools for your platform

For additional help, refer to the [documentation](https://eloy.readthedocs.io/) or open an issue on the [GitHub repository](https://github.com/lgrcia/eloy/issues).