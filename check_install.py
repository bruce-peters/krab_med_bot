"""
Check Installation Status - Krab Med Bot
Verifies all required packages are installed and shows versions
"""

import sys

# Define required packages
REQUIRED_PACKAGES = {
    'fastapi': 'Web framework',
    'uvicorn': 'ASGI server',
    'pydantic': 'Data validation',
    'pydantic_settings': 'Settings management',
    'httpx': 'Async HTTP client',
    'dotenv': 'Environment variables',
    'aiofiles': 'Async file operations',
    'openai': 'OpenAI API client',
    'anthropic': 'Anthropic API client',
    'spacy': 'NLP library',
    'transformers': 'ML transformers',
    'torch': 'PyTorch',
}

OPTIONAL_PACKAGES = {
    'pytest': 'Testing framework',
    'black': 'Code formatter',
    'flake8': 'Code linter',
}

def check_package(package_name):
    """Check if a package is installed and return its version"""
    try:
        if package_name == 'dotenv':
            # python-dotenv imports as dotenv
            import dotenv
            return dotenv.__version__
        elif package_name == 'pydantic_settings':
            # pydantic-settings imports as pydantic_settings
            from pydantic_settings import __version__
            return __version__
        else:
            module = __import__(package_name)
            return getattr(module, '__version__', 'Unknown')
    except ImportError:
        return None

def main():
    print("=" * 70)
    print("Krab Med Bot - Installation Status Check")
    print("=" * 70)
    print()
    
    # Check Python version
    py_version = sys.version.split()[0]
    print(f"Python Version: {py_version}")
    
    if sys.version_info < (3, 8):
        print("⚠️  WARNING: Python 3.8+ is required!")
    else:
        print("✓ Python version OK")
    
    print()
    print("-" * 70)
    print("REQUIRED PACKAGES:")
    print("-" * 70)
    
    required_missing = []
    for package, description in REQUIRED_PACKAGES.items():
        version = check_package(package)
        if version:
            print(f"✓ {package:20} v{version:15} - {description}")
        else:
            print(f"✗ {package:20} {'NOT INSTALLED':15} - {description}")
            required_missing.append(package)
    
    print()
    print("-" * 70)
    print("OPTIONAL PACKAGES:")
    print("-" * 70)
    
    for package, description in OPTIONAL_PACKAGES.items():
        version = check_package(package)
        if version:
            print(f"✓ {package:20} v{version:15} - {description}")
        else:
            print(f"○ {package:20} {'Not installed':15} - {description}")
    
    print()
    print("-" * 70)
    
    # Check spaCy model
    print("\nCHECKING SPACY MODEL:")
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            print("✓ spaCy model 'en_core_web_sm' is installed")
        except OSError:
            print("✗ spaCy model 'en_core_web_sm' NOT installed")
            print("  Run: python -m spacy download en_core_web_sm")
            required_missing.append("spacy-model")
    except ImportError:
        print("✗ spaCy not installed (cannot check model)")
    
    print()
    print("=" * 70)
    
    # Summary
    if required_missing:
        print(f"\n❌ INSTALLATION INCOMPLETE - {len(required_missing)} required item(s) missing:")
        for item in required_missing:
            print(f"   - {item}")
        print("\nTo install missing packages:")
        print("   pip install -r requirements.txt")
        if "spacy-model" in required_missing:
            print("   python -m spacy download en_core_web_sm")
        return 1
    else:
        print("\n✅ ALL REQUIRED PACKAGES INSTALLED!")
        print("\nYou're ready to start the server:")
        print("   uvicorn server.main:app --reload")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
