#!/bin/bash
set -e

echo "🔍 Running code quality checks..."

echo "📏 Checking code formatting with Black..."
uv run --group dev black --check --diff .

echo "📋 Checking import sorting with isort..."
uv run --group dev isort --check-only --diff .

echo "🔍 Running flake8 linter..."
uv run --group dev flake8 .

echo "🔧 Running mypy type checker..."
uv run --group dev mypy .

echo "✅ All quality checks passed!"