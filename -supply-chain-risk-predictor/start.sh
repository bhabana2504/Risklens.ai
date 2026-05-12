

set -e
CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo -e "${CYAN}"
echo "  ╔═══════════════════════════════════════╗"
echo "  ║  ChainSight AI — Supply Chain Risk    ║"
echo "  ║  Predictor · Starting Up …            ║"
echo "  ╚═══════════════════════════════════════╝"
echo -e "${NC}"

# 1. Create & activate virtual env if not present
if [ ! -d ".venv" ]; then
  echo -e "${YELLOW}Creating virtual environment…${NC}"
  python3 -m venv .venv
fi

source .venv/bin/activate

# 2. Install dependencies
echo -e "${YELLOW}Installing dependencies…${NC}"
pip install -q -r requirements.txt

# 3. Train model if not already present
if [ ! -f "model/risk_model.joblib" ]; then
  echo -e "${YELLOW}Training ML model (first run)…${NC}"
  python model/train_model.py
fi

echo -e "${GREEN}✓ Model ready${NC}"

# 4. Launch Flask
echo -e "${GREEN}✓ Starting API server on http://localhost:5000${NC}"
echo -e "${GREEN}  Open http://localhost:5000 in your browser${NC}"
echo ""
cd backend && python app.py
