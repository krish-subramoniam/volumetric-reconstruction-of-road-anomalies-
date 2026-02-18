#!/usr/bin/env python
"""
Quick launcher for the Gradio UI with dependency checking.
"""

import sys
import subprocess

def check_dependencies():
    """Check if required packages are installed."""
    required = ['gradio', 'matplotlib', 'PIL', 'cv2', 'numpy']
    missing = []
    
    for package in required:
        try:
            if package == 'PIL':
                __import__('PIL')
            elif package == 'cv2':
                __import__('cv2')
            else:
                __import__(package)
        except ImportError:
            missing.append(package)
    
    return missing

def install_dependencies(packages):
    """Install missing dependencies."""
    print(f"Installing missing dependencies: {', '.join(packages)}")
    
    # Map package names to pip install names
    pip_names = {
        'PIL': 'pillow',
        'cv2': 'opencv-python'
    }
    
    for package in packages:
        pip_name = pip_names.get(package, package)
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pip_name])
        except subprocess.CalledProcessError:
            print(f"Failed to install {pip_name}")
            return False
    
    return True

def main():
    """Main launcher function."""
    print("=" * 60)
    print("Advanced Stereo Vision Pipeline - Gradio UI Launcher")
    print("=" * 60)
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        response = input("Would you like to install them now? (y/n): ")
        
        if response.lower() == 'y':
            if install_dependencies(missing):
                print("✅ Dependencies installed successfully!")
            else:
                print("❌ Failed to install some dependencies. Please install manually.")
                return 1
        else:
            print("Please install missing dependencies manually:")
            for package in missing:
                pip_name = {'PIL': 'pillow', 'cv2': 'opencv-python'}.get(package, package)
                print(f"  pip install {pip_name}")
            return 1
    else:
        print("✅ All dependencies are installed!")
    
    print()
    print("Starting Gradio UI...")
    print("The interface will open at: http://localhost:7860")
    print("Press Ctrl+C to stop the server")
    print()
    
    # Import and launch
    try:
        from gradio_app import demo
        demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        return 0
    except Exception as e:
        print(f"\n❌ Error launching Gradio UI: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
