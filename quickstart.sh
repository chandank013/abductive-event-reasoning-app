#!/bin/bash

# ============================================================================
# AER System Quick Start Script (Days 6-14)
# ============================================================================

set -e

echo "================================================"
echo "AER System Quick Start"
echo "================================================"

# Step 1: Build embeddings
echo ""
echo "📥 Step 1/4: Building embeddings..."
if [ -d "data/sample" ] && [ -f "data/sample/questions.jsonl" ]; then
    echo "Building embeddings for sample data..."
    python scripts/build_embeddings.py \
        --data-dir data/sample \
        --output-dir data/embeddings/sample \
        --batch-size 16
    echo "✓ Sample embeddings built"
fi

if [ -d "data/train" ] && [ -f "data/train/questions.jsonl" ]; then
    echo "Building embeddings for training data..."
    python scripts/build_embeddings.py \
        --data-dir data/train \
        --output-dir data/embeddings/train \
        --batch-size 32
    echo "✓ Training embeddings built"
fi

# Step 2: Run integration tests
echo ""
echo "🧪 Step 2/4: Running integration tests..."
python test_integration.py

# Step 3: Test Flask app
echo ""
echo "🌐 Step 3/4: Testing Flask application..."
python -c "
from app import create_app
app = create_app()
print('✓ Flask app ready')
"

# Step 4: Instructions
echo ""
echo "================================================"
echo "✅ Setup Complete!"
echo "================================================"
echo ""
echo "🚀 To start the application:"
echo ""
echo "   python run.py --debug"
echo ""
echo "Then open your browser to:"
echo "   http://localhost:5000"
echo ""
echo "Available pages:"
echo "   - Home:       http://localhost:5000/"
echo "   - Dashboard:  http://localhost:5000/dashboard"
echo "   - Predict:    http://localhost:5000/predict"
echo "   - Data:       http://localhost:5000/data"
echo "   - API:        http://localhost:5000/api/health"
echo ""
echo "================================================"