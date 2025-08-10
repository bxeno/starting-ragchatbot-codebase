#!/bin/bash
set -e

echo "🔧 Running code formatting..."
echo "📏 Formatting Python code with Black..."
uv run --group dev black .

echo "📋 Sorting imports with isort..."
uv run --group dev isort .

echo "✅ Code formatting complete!"