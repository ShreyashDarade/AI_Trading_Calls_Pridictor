"""
Run All Services Script
Starts all microservices for local development
"""
import asyncio
import subprocess
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SERVICES = [
    {
        "name": "API Gateway",
        "module": "apps.api-gateway.src.main:app",
        "port": 8000
    },
    {
        "name": "Instrument Service",
        "module": "services.instrument-service.src.main:app",
        "port": 8001
    },
    {
        "name": "Market Ingestor",
        "module": "services.market-ingestor.src.main:app",
        "port": 8002
    },
    {
        "name": "Feature Service",
        "module": "services.feature-service.src.main:app",
        "port": 8004
    },
    {
        "name": "Signal Service",
        "module": "services.signal-service.src.main:app",
        "port": 8005
    },
]


def run_service(name: str, module: str, port: int):
    """Run a single service"""
    print(f"Starting {name} on port {port}...")
    module_path = module.replace("-", "_")
    cmd = [
        sys.executable, "-m", "uvicorn",
        module_path,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--reload"
    ]
    return subprocess.Popen(cmd, cwd=PROJECT_ROOT)


def main():
    """Start all services"""
    print("=" * 60)
    print("INDIAN AI TRADER - Starting All Services")
    print("=" * 60)
    
    processes = []
    
    try:
        for service in SERVICES:
            proc = run_service(
                service["name"],
                service["module"],
                service["port"]
            )
            processes.append(proc)
        
        print("\n" + "=" * 60)
        print("All services started!")
        print("=" * 60)
        print("\nEndpoints:")
        print("  - Dashboard:        http://localhost:8000")
        print("  - API Docs:         http://localhost:8000/api/docs")
        print("  - Instrument API:   http://localhost:8001")
        print("  - Feature API:      http://localhost:8004")
        print("  - Signal API:       http://localhost:8005")
        print("\nPress Ctrl+C to stop all services\n")
        
        # Wait for processes
        for proc in processes:
            proc.wait()
            
    except KeyboardInterrupt:
        print("\n\nShutting down services...")
        for proc in processes:
            proc.terminate()
        print("All services stopped.")


if __name__ == "__main__":
    main()
