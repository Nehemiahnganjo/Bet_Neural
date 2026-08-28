#!/bin/bash
# Launch Bet Neural GUI

cd "$(dirname "$0")"

# Check if GUI binary exists
if [ -f "bet_neural_gui/target/release/bet_neural_gui" ]; then
    echo "🚀 Launching Bet Neural GUI..."
    echo "   Enhanced with improved team matching and form analysis"
    echo "   All CLI improvements are active in GUI mode"
    echo
    ./bet_neural_gui/target/release/bet_neural_gui
else
    echo "❌ GUI not built yet. Building now..."
    cd bet_neural_gui
    cargo build --release
    if [ $? -eq 0 ]; then
        echo "✅ Build successful! Launching GUI..."
        ./target/release/bet_neural_gui
    else
        echo "❌ Build failed. Please check Rust installation."
    fi
fi