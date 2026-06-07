#setup.sh
#!/bin/bash

echo "Setting up TabNet → CNN → MOL Pipeline..."

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install package in development mode
pip install -e ".[dev]"

# Verify installation
python -c "from api import PipelineAPI; print('✅ API loaded successfully')"
python -c "import streamlit as st; print('✅ Streamlit available')"

echo "✅ Setup complete! Activate with: source venv/bin/activate"