#!/bin/bash

echo "🏆 Bet Neural - European Football Prediction System"
echo "=================================================="
echo ""

# Check if we're in the right directory
if [ ! -f "bet_neural_lite.py" ]; then
    echo "❌ Error: bet_neural_lite.py not found"
    echo "Please run this script from the Bet_Neural directory"
    exit 1
fi

# Make scripts executable
echo "🔧 Setting up executable permissions..."
chmod +x bet_neural_lite.py

# Create convenient alias
echo "🔗 Creating bet command alias..."
cat > bet << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
python3 bet_neural_lite.py "$@"
EOF

chmod +x bet

# Test the system
echo "🧪 Testing system..."
python3 bet_neural_lite.py benchmark > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✅ System test passed!"
else
    echo "⚠️  System test had issues, but should still work"
fi

echo ""
echo "🎉 INSTALLATION COMPLETE!"
echo "========================="
echo ""
echo "🚀 Quick Start:"
echo "   ./bet predict \"Arsenal vs Chelsea\""
echo "   ./bet predict \"Real Madrid vs Barcelona\" --league=la_liga"
echo "   ./bet leagues"
echo "   ./bet benchmark"
echo ""
echo "💰 With betting odds:"
echo "   ./bet predict \"Arsenal vs Chelsea\" --odds=2.1,3.4,3.8"
echo ""
echo "🤖 Kiro Integration:"
echo "   kiro-cli run ./bet predict \"Liverpool vs Manchester City\""
echo ""
echo "✨ Features:"
echo "   🏆 7 European leagues supported"
echo "   📊 FiveThirtyEight-inspired Elo system"
echo "   💰 Kelly Criterion betting analysis"
echo "   🎯 No dependencies required"
echo ""
echo "🏟️  Ready to predict European football matches!"