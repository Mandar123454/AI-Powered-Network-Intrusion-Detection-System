#!/usr/bin/env python
"""
AI-NIDS Setup Script
====================
Automated setup and initialization for the AI-NIDS system
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def print_banner():
    """Print setup banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   🛡️  AI-NIDS: AI Network Intrusion Detection System        ║
    ║                                                              ║
    ║   Setup & Installation Script                                ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_python_version():
    """Check Python version."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"❌ Python 3.10+ required, found {version.major}.{version.minor}")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")


def create_directories():
    """Create necessary directories."""
    print("\nCreating directories...")
    directories = [
        'instance',
        'logs',
        'models',
        'data/raw',
        'data/processed',
        'deployment/ssl'
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {directory}")


def create_env_file():
    """Create .env file if not exists."""
    print("\nChecking environment file...")
    
    env_file = Path('.env')
    example_file = Path('.env.example')
    
    if not env_file.exists():
        if example_file.exists():
            shutil.copy(example_file, env_file)
            print("  ✅ Created .env from .env.example")
            print("  ⚠️  Please update .env with your configuration")
        else:
            print("  ❌ .env.example not found")
    else:
        print("  ✅ .env already exists")


def install_dependencies():
    """Install Python dependencies."""
    print("\nInstalling dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ])
        print("✅ Dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    
    return True


def initialize_database():
    """Initialize the database."""
    print("\nInitializing database...")
    
    os.environ.setdefault('FLASK_APP', 'run.py')
    os.environ.setdefault('FLASK_ENV', 'development')
    
    try:
        # Initialize Flask-Migrate
        subprocess.run([sys.executable, '-m', 'flask', 'db', 'init'], check=False)
        subprocess.run([sys.executable, '-m', 'flask', 'db', 'migrate', '-m', 'Initial migration'], check=False)
        subprocess.run([sys.executable, '-m', 'flask', 'db', 'upgrade'], check=False)
        print("✅ Database initialized")
    except Exception as e:
        print(f"⚠️  Database migration may need manual setup: {e}")


def run_tests():
    """Run test suite."""
    print("\nRunning tests...")
    
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✅ All tests passed")
        else:
            print("⚠️  Some tests failed")
            print(result.stdout)
    except Exception as e:
        print(f"⚠️  Could not run tests: {e}")


def print_next_steps():
    """Print next steps."""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                     Setup Complete! 🎉                       ║
    ╚══════════════════════════════════════════════════════════════╝
    
    Next Steps:
    
    1. Configure your .env file:
       - Set SECRET_KEY
       - Configure DATABASE_URL (if using PostgreSQL)
       - Set up any optional integrations
    
    2. Download training data:
       - CICIDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
       - UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset
       Place datasets in: data/raw/
    
    3. Train ML models:
       Open notebooks/model_training.ipynb in Jupyter
       Or run: python -m ml.training
    
    4. Start the application:
       Development: python run.py
       Production:  gunicorn -c gunicorn.conf.py wsgi:app
       Docker:      docker-compose up -d
    
    5. Access the dashboard:
       http://localhost:5000
       Default admin: admin / admin123 (change immediately!)
    
    Documentation: README.md
    Issues: https://github.com/yourusername/ai-nids/issues
    """)


def main():
    """Main setup function."""
    print_banner()
    
    # Change to project directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print(f"Working directory: {os.getcwd()}\n")
    
    # Run setup steps
    check_python_version()
    create_directories()
    create_env_file()
    
    # Ask about installing dependencies
    response = input("\nInstall Python dependencies? [Y/n]: ").strip().lower()
    if response != 'n':
        if not install_dependencies():
            print("Setup incomplete due to dependency installation failure")
            sys.exit(1)
    
    # Initialize database
    response = input("\nInitialize database? [Y/n]: ").strip().lower()
    if response != 'n':
        initialize_database()
    
    # Run tests
    response = input("\nRun tests? [y/N]: ").strip().lower()
    if response == 'y':
        run_tests()
    
    print_next_steps()


if __name__ == '__main__':
    main()
