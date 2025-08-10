#!/bin/bash
set -e

echo "🚀 Running comprehensive code quality checks..."

echo "Step 1: Formatting code..."
./scripts/format.sh

echo ""
echo "Step 2: Running quality checks..."
./scripts/lint.sh

echo ""
echo "🎉 All checks completed successfully!"