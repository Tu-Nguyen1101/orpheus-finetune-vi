#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script showing how to use Orpheus TTS Gradio interface
Run this to start the web interface with Vietnamese TTS and voice cloning
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import gradio
        import torch
        import torchaudio
        import librosa
        print("✅ All required packages found")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("💡 Install with: pip install gradio>=4.0.0 psutil")
        return False

def main():
    """Run the demo"""
    
    print("🎤 Orpheus TTS Vietnamese - Gradio Demo")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print("\n📋 Demo Instructions:")
    print("1. 🔄 Load your trained model checkpoint")
    print("2. 📝 Enter Vietnamese text in the Text-to-Speech tab")
    print("3. 🎭 Try voice cloning with reference audio")
    print("4. ⚙️ Adjust parameters in Advanced tab")
    print("\n🌐 The interface will be available at:")
    print("   - Local: http://localhost:7860")
    print("   - Public: [Gradio will generate a public link]")
    print("\n🚀 Starting Gradio interface...")
    print("=" * 50)
    
    try:
        # Import and launch
        from gradio_app import main as launch_app
        launch_app()
        
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Make sure you have all dependencies installed")

if __name__ == "__main__":
    main()
