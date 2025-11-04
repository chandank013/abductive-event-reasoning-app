#!/bin/bash

# ============================================================================
# Setup Script for Data Organization
# ============================================================================

set -e

echo "================================================"
echo "AER Dataset Setup"
echo "================================================"

# Configuration
SAMPLE_DIR="sample_data"
TRAIN_DIR="train_data"
DEV_DIR="dev_data"
TEST_DIR="test_data"

DATA_SAMPLE="data/sample"
DATA_TRAIN="data/train"
DATA_DEV="data/dev"
DATA_TEST="data/test"

# Check if source directories exist
echo ""
echo "🔍 Checking for provided datasets..."

# Sample data
if [ -d "$SAMPLE_DIR" ]; then
    echo "✓ Found: $SAMPLE_DIR"
    if [ -f "$SAMPLE_DIR/questions.jsonl" ]; then
        cp "$SAMPLE_DIR/questions.jsonl" "$DATA_SAMPLE/"
        SAMPLE_Q=$(wc -l < "$DATA_SAMPLE/questions.jsonl")
        echo "  → Copied questions.jsonl ($SAMPLE_Q instances)"
    fi
    if [ -f "$SAMPLE_DIR/docs.json" ]; then
        cp "$SAMPLE_DIR/docs.json" "$DATA_SAMPLE/"
        echo "  → Copied docs.json"
    fi
else
    echo "⚠️  $SAMPLE_DIR not found!"
fi

# Training data
if [ -d "$TRAIN_DIR" ]; then
    echo "✓ Found: $TRAIN_DIR"
    if [ -f "$TRAIN_DIR/questions.jsonl" ]; then
        cp "$TRAIN_DIR/questions.jsonl" "$DATA_TRAIN/"
        TRAIN_Q=$(wc -l < "$DATA_TRAIN/questions.jsonl")
        echo "  → Copied questions.jsonl ($TRAIN_Q instances)"
    fi
    if [ -f "$TRAIN_DIR/docs.json" ]; then
        cp "$TRAIN_DIR/docs.json" "$DATA_TRAIN/"
        echo "  → Copied docs.json"
    fi
else
    echo "❌ Error: $TRAIN_DIR not found!"
    exit 1
fi

# Dev data
if [ -d "$DEV_DIR" ]; then
    echo "✓ Found: $DEV_DIR"
    if [ -f "$DEV_DIR/questions.jsonl" ]; then
        cp "$DEV_DIR/questions.jsonl" "$DATA_DEV/"
        DEV_Q=$(wc -l < "$DATA_DEV/questions.jsonl")
        echo "  → Copied questions.jsonl ($DEV_Q instances)"
    fi
    if [ -f "$DEV_DIR/docs.json" ]; then
        cp "$DEV_DIR/docs.json" "$DATA_DEV/"
        echo "  → Copied docs.json"
    fi
else
    echo "❌ Error: $DEV_DIR not found!"
    exit 1
fi

# Test data placeholder
if [ -d "$TEST_DIR" ]; then
    echo "✓ Found: $TEST_DIR"
    if [ -f "$TEST_DIR/questions.jsonl" ]; then
        cp "$TEST_DIR/questions.jsonl" "$DATA_TEST/"
        echo "  → Copied questions.jsonl"
    fi
    if [ -f "$TEST_DIR/docs.json" ]; then
        cp "$TEST_DIR/docs.json" "$DATA_TEST/"
        echo "  → Copied docs.json"
    fi
else
    echo "ℹ️  $TEST_DIR not found (will be released later)"
fi

echo ""
echo "================================================"
echo "✅ Data setup complete!"
echo "================================================"
echo ""
echo "📊 Dataset Summary:"
if [ -n "$SAMPLE_Q" ]; then
    echo "  Sample:   $SAMPLE_Q instances"
fi
if [ -n "$TRAIN_Q" ]; then
    echo "  Training: $TRAIN_Q instances"
fi
if [ -n "$DEV_Q" ]; then
    echo "  Dev:      $DEV_Q instances"
fi