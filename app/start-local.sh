#!/bin/bash
# Simple script to start SAM3 locally (backend + frontend)

echo "🚀 Starting SAM3 Annotation Pipeline"
echo ""

# Check if backend is already running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend is already running on port 8000"
else
    echo "Starting backend..."
    echo "   (This will open in a new terminal window)"
    echo ""
    echo "If it doesn't start automatically, run in a separate terminal:"
    echo "  cd $(dirname "$0")/.."
    echo "  uv run python3 app/backend/main.py"
    echo ""
    
    # Try to start backend in background
    cd "$(dirname "$0")/.."
    uv run python3 app/backend/main.py &
    BACKEND_PID=$!
    echo "Backend started (PID: $BACKEND_PID)"
    sleep 5
fi

# Start frontend
echo ""
echo "Starting frontend..."
cd "$(dirname "$0")/frontend"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ Both services starting!"
echo ""
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both services"
echo ""

# Wait for interrupt
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait

