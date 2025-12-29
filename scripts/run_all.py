"""
Run All Services Script
Starts all microservices for local development

Usage:
    python scripts/run_all.py           # Start all services
    python scripts/run_all.py --api     # Start only API Gateway
    python scripts/run_all.py --minimal # Start only essential services
    python scripts/run_all.py --all     # Explicitly start all services

Options:
    --no-reload   Disable uvicorn auto-reload
    --wait        Wait for each service /health before continuing
"""
import subprocess
import sys
import os
import signal
import time
from pathlib import Path
from typing import List, Tuple

import httpx

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# All available services
SERVICES = {
    "api-gateway": {
        "name": "API Gateway (Main)",
        "path": "apps/api-gateway/src/main.py",
        "port": 8000,
        "essential": True
    },
    "nse-data": {
        "name": "NSE Data Service",
        "path": "services/nse-data/src/main.py",
        "port": 8020,
        "essential": False
    },
    "signal-service": {
        "name": "Signal Service",
        "path": "services/signal-service/src/main.py",
        "port": 8005,
        "essential": False
    },
    "instrument-service": {
        "name": "Instrument Service",
        "path": "services/instrument-service/src/main.py",
        "port": 8001,
        "essential": False
    },
    "feature-service": {
        "name": "Feature Service",
        "path": "services/feature-service/src/main.py",
        "port": 8004,
        "essential": False
    },
    "portfolio-service": {
        "name": "Portfolio Service",
        "path": "services/portfolio-service/src/main.py",
        "port": 8007,
        "essential": False
    },
    "backtest-service": {
        "name": "Backtest Service",
        "path": "services/backtest-service/src/main.py",
        "port": 8006,
        "essential": False
    },
}

processes: List[Tuple[str, subprocess.Popen]] = []


def run_service(key: str, service: dict, *, reload: bool) -> subprocess.Popen:
    """Start a single service"""
    name = service["name"]
    path = service["path"]
    port = service["port"]
    
    # Convert path to module format
    module_path = path.replace("/", ".").replace("\\", ".").replace(".py", "")
    
    print(f"  🚀 Starting {name} on port {port}...")
    
    cmd = [
        sys.executable, "-m", "uvicorn",
        f"{module_path}:app",
        "--host", "0.0.0.0",
        "--port", str(port),
    ]
    if reload:
        cmd.append("--reload")
    
    # Start the process
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=None,
        stderr=None,
        text=True
    )
    
    return proc


def stop_all():
    """Stop all running services"""
    global processes
    print("\n⏹️  Stopping all services...")
    for _, proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except:
            proc.kill()
    print("✅ All services stopped.")


def signal_handler(sig, frame):
    """Handle Ctrl+C"""
    stop_all()
    sys.exit(0)


def main():
    """Main function"""
    global processes
    
    # Handle Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse arguments
    args = sys.argv[1:]
    reload = "--no-reload" not in args
    wait = "--wait" in args
    args = [a for a in args if a not in {"--no-reload", "--wait"}]
    
    if "--help" in args or "-h" in args:
        print(__doc__)
        print("\nAvailable services:")
        for key, svc in SERVICES.items():
            print(f"  {key}: {svc['name']} (port {svc['port']})")
        return
    
    # Determine which services to start
    if "--api" in args:
        services_to_start = ["api-gateway"]
    elif "--minimal" in args:
        services_to_start = [k for k, v in SERVICES.items() if v.get("essential")]
    elif "--all" in args:
        services_to_start = list(SERVICES.keys())
    elif args:
        # Specific services provided
        services_to_start = [a for a in args if a in SERVICES]
    else:
        # Start all services
        services_to_start = list(SERVICES.keys())
    
    print()
    print("=" * 60)
    print("🇮🇳 INDIAN AI TRADER - Service Manager")
    print("=" * 60)
    print()
    
    # Check which services exist
    available_services = []
    for key in services_to_start:
        service = SERVICES[key]
        path = PROJECT_ROOT / service["path"]
        if path.exists():
            available_services.append(key)
        else:
            print(f"  ⚠️  Skipping {service['name']} (not found: {service['path']})")
    
    if not available_services:
        print("❌ No services to start!")
        return
    
    print(f"Starting {len(available_services)} service(s):\n")
    
    # Start services
    for key in available_services:
        service = SERVICES[key]
        try:
            proc = run_service(key, service, reload=reload)
            processes.append((key, proc))
            time.sleep(1)  # Small delay between starts
            
            if wait:
                _wait_for_health(service["port"], service["name"])
        except Exception as e:
            print(f"  ❌ Failed to start {service['name']}: {e}")
    
    print()
    print("=" * 60)
    print("✅ Services Started!")
    print("=" * 60)
    print()
    print("📍 Endpoints:")
    print("   Dashboard:     http://localhost:8000")
    print("   Watchlist:     http://localhost:8000/watchlist")
    print("   Signals:       http://localhost:8000/signals")
    print("   Backtests:     http://localhost:8000/backtests")
    print("   Settings:      http://localhost:8000/settings")
    print("   API Health:    http://localhost:8000/api/health")
    print()
    print("🛑 Press Ctrl+C to stop all services")
    print("=" * 60)
    print()
    
    # Keep running
    try:
        while True:
            # Check if processes are still running
            for key, proc in processes:
                if proc.poll() is not None:
                    print(f"⚠️  {SERVICES[key]['name']} exited with code {proc.returncode}")
            time.sleep(2)
    except KeyboardInterrupt:
        stop_all()


def _wait_for_health(port: int, name: str, timeout_s: float = 30.0) -> None:
    url = f"http://localhost:{port}/health"
    start = time.time()
    while True:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return
        except Exception:
            pass
        if time.time() - start > timeout_s:
            raise RuntimeError(f"Timeout waiting for {name} at {url}")
        time.sleep(0.5)


if __name__ == "__main__":
    main()
