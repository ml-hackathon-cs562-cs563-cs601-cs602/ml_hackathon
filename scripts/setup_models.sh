#!/bin/bash
# Setup script to prepare model artifacts for the API
# Run this on Laptop 1 before starting the server

set -e

echo "=== PathSnap Model Setup ==="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Create artifacts directory
echo -e "${YELLOW}Creating artifacts directory...${NC}"
mkdir -p artifacts

# Check and extract MLP model
echo ""
echo -e "${YELLOW}Checking MLP model...${NC}"
if [ -f "results/model5/outputs_mlp.zip" ]; then
    echo "Found outputs_mlp.zip, extracting..."
    cd results/model5
    unzip -o outputs_mlp.zip
    cd "$PROJECT_ROOT"
    echo -e "${GREEN}✓ MLP model extracted${NC}"
elif [ -f "results/model5/outputs_mlp/artifacts/mlp_untuned.joblib" ]; then
    echo -e "${GREEN}✓ MLP model already extracted${NC}"
else
    echo -e "${RED}✗ MLP model not found${NC}"
fi

# Check LightGBM model
echo ""
echo -e "${YELLOW}Checking LightGBM model...${NC}"
if [ -f "results/model4/outputs_lgbm/artifacts/lgbm_untuned.txt" ]; then
    echo -e "${GREEN}✓ LightGBM model found${NC}"
else
    echo -e "${RED}✗ LightGBM model not found${NC}"
fi

# Copy models to artifacts directory
echo ""
echo -e "${YELLOW}Copying models to artifacts/ directory...${NC}"

if [ -f "results/model4/outputs_lgbm/artifacts/lgbm_untuned.txt" ]; then
    cp results/model4/outputs_lgbm/artifacts/lgbm_untuned.txt artifacts/
    echo -e "${GREEN}✓ LightGBM copied to artifacts/${NC}"
fi

if [ -f "results/model5/outputs_mlp/artifacts/mlp_untuned.joblib" ]; then
    cp results/model5/outputs_mlp/artifacts/mlp_untuned.joblib artifacts/
    echo -e "${GREEN}✓ MLP copied to artifacts/${NC}"
fi

if [ -f "results/model5/outputs_mlp/artifacts/scaler.joblib" ]; then
    cp results/model5/outputs_mlp/artifacts/scaler.joblib artifacts/
    echo -e "${GREEN}✓ Scaler copied to artifacts/${NC}"
fi

# Check for Member C's models
echo ""
echo -e "${YELLOW}Checking Member C models...${NC}"
for model in randomforest logisticregression decisiontree; do
    if [ -f "artifacts/${model}_config.json" ]; then
        echo -e "${GREEN}✓ $model config found${NC}"
    else
        echo -e "${YELLOW}⚠ $model config not found${NC}"
    fi
done

# Verify data
echo ""
echo -e "${YELLOW}Checking data directory...${NC}"
if [ -d "data/raw/student_test" ]; then
    TRIPS=$(ls -d data/raw/student_test/*/ 2>/dev/null | wc -l)
    echo -e "${GREEN}✓ Found $TRIPS student test trips${NC}"
else
    echo -e "${RED}✗ Student test data not found${NC}"
fi

# Summary
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Model artifacts:"
ls -lh artifacts/ 2>/dev/null || echo "  (empty)"
echo ""
echo -e "${GREEN}Ready to start the API server:${NC}"
echo "  python -m uvicorn api.main:app --host 0.0.0.0 --port 8000"
echo ""
