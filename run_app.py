import subprocess
import sys
import time
import webbrowser
import os
import signal
from pathlib import Path

def print_status(msg, color="white"):
    # ANSI colors
    colors = {
        "white": "\033[97m",
        "cyan": "\033[96m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "magenta": "\033[95m",
    }
    end = "\033[0m"
    print(f"{colors.get(color, colors['white'])}[{color.upper()}] {msg}{end}")

def kill_port(port):
    """Kill any process listening on the specified port."""
    try:
        if os.name == 'nt':  # Windows
            # Use netstat to find PIDs using the port
            result = subprocess.run(
                ["netstat", "-ano", "-p", "tcp"],
                capture_output=True,
                text=True
            )
            for line in result.stdout.split('\n'):
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        print_status(f"Killing existing process on port {port} (PID: {pid})...", "yellow")
                        subprocess.run(["taskkill", "/F", "/PID", pid], check=False)
        else:  # Unix/Linux/macOS
            # Use lsof to find PIDs
            result = subprocess.run(
                ["lsof", "-t", "-i", f":{port}"],
                capture_output=True,
                text=True
            )
            pids = result.stdout.strip().split('\n')

            for pid in pids:
                if pid:
                    print_status(f"Killing existing process on port {port} (PID: {pid})...", "yellow")
                    subprocess.run(["kill", "-9", pid], check=False)
    except Exception as e:
        # Commands might not be available or other error, just ignore
        pass

def ensure_venv(root_dir):
    """Ensure a virtual environment exists and is used."""
    venv_dir = root_dir / "venv"

    # Platform-specific Python executable path
    if os.name == 'nt':  # Windows
        venv_python = venv_dir / "Scripts" / "python.exe"
    else:  # Unix/Linux/macOS
        venv_python = venv_dir / "bin" / "python"

    # If we are already running in the venv, continue
    if sys.prefix == str(venv_dir):
        return sys.executable

    print_status("Checking environment...", "cyan")

    # Create venv if it doesn't exist
    if not venv_dir.exists():
        print_status("Creating virtual environment...", "yellow")
        subprocess.run([sys.executable, "-m", "venv", "venv"], cwd=root_dir, check=True)

        # Install requirements
        print_status("Installing backend dependencies...", "yellow")
        subprocess.run([str(venv_python), "-m", "pip", "install", "-r", "requirements.txt"], cwd=root_dir, check=True)

    # Re-execute this script using the venv python
    print_status("Switching to virtual environment...", "cyan")
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)

def main():
    # Paths
    root_dir = Path(__file__).parent.resolve()
    frontend_dir = root_dir / "frontend"
    
    # 0. Ensure Venv
    ensure_venv(root_dir)
    
    # 1. Cleanup Ports
    kill_port(8001)
    kill_port(5173)
    
    # 2. Install Frontend Dependencies if needed
    if not (frontend_dir / "node_modules").exists():
        print_status("Installing frontend dependencies...", "cyan")
        try:
            # Use npm.cmd on Windows to ensure proper path resolution
            npm_cmd = "npm.cmd" if os.name == 'nt' else "npm"
            subprocess.run([npm_cmd, "install"], cwd=frontend_dir, check=True, shell=False)
        except subprocess.CalledProcessError as e:
            print_status(f"Failed to install frontend dependencies: {e}", "red")
            print_status("Please ensure Node.js and npm are properly installed", "yellow")
            return
    
    # 3. Start Backend
    print_status("Starting Backend API...", "green")
    backend_env = os.environ.copy()
    # Ensure backend sees the venv
    backend_env["VIRTUAL_ENV"] = str(root_dir / "venv")

    # Platform-specific PATH setup
    if os.name == 'nt':  # Windows
        backend_env["PATH"] = f"{root_dir}/venv/Scripts;{backend_env['PATH']}"
    else:  # Unix/Linux/macOS
        backend_env["PATH"] = f"{root_dir}/venv/bin:{backend_env['PATH']}"
    
    backend_process = subprocess.Popen(
        [sys.executable, "api_server.py"],
        cwd=root_dir,
        env=backend_env
    )
    
    # 4. Start Frontend
    print_status("Starting Frontend Dev Server...", "green")
    # Use npm.cmd on Windows to ensure proper path resolution
    npm_cmd = "npm.cmd" if os.name == 'nt' else "npm"

    # On Windows, we need to inherit stdout/stderr to see the output
    if os.name == 'nt':
        frontend_process = subprocess.Popen(
            [npm_cmd, "run", "dev"],
            cwd=frontend_dir,
            text=True,
            shell=False
        )
    else:
        frontend_process = subprocess.Popen(
            [npm_cmd, "run", "dev"],
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False
        )
    
    # Wait a bit for servers to spin up
    time.sleep(3)

    # 5. Open Browser
    print_status("Opening Fractal Notebook...", "magenta")
    # Check if frontend is running on expected port or alternative
    frontend_url = "http://localhost:5173"
    try:
        import requests
        response = requests.get(frontend_url, timeout=2)
    except:
        # If port 5173 is not accessible, try 5174 (common alternative)
        frontend_url = "http://localhost:5174"

    webbrowser.open(frontend_url)
    
    print_status("System Running. Press Ctrl+C to stop.", "cyan")
    
    try:
        while True:
            time.sleep(1)
            if backend_process.poll() is not None:
                print_status("Backend process exited unexpectedly.", "red")
                break
            if frontend_process.poll() is not None:
                print_status("Frontend process exited unexpectedly.", "red")
                # Print frontend output and error if it failed
                if frontend_process.stdout:
                    print("Frontend stdout:", frontend_process.stdout.read())
                if frontend_process.stderr:
                    print("Frontend stderr:", frontend_process.stderr.read())
                break
    except KeyboardInterrupt:
        print_status("\nStopping system...", "yellow")
    finally:
        backend_process.terminate()
        frontend_process.terminate()
        try:
            backend_process.wait(timeout=5)
            frontend_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend_process.kill()
            frontend_process.kill()
        print_status("Shutdown complete.", "green")

if __name__ == "__main__":
    main()
