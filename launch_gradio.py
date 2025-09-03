#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick launcher for Orpheus TTS Gradio interface
"""

import os
import sys

# Fix tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    """Launch the Gradio interface"""
    
    print("🚀 Launching Orpheus TTS Gradio Interface...")
    print("📍 Default model path: /workspace/training_orpherus/h100_large_dataset_output/checkpoint-1600/")
    print("🌐 Share=True enabled - you'll get a public link!")
    print("---")
    
    try:
        # Import and run
        from gradio_app import main as run_gradio
        run_gradio()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install gradio>=4.0.0 psutil")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n👋 Gradio interface stopped by user")
        sys.exit(0)
        
    except Exception as e:
        print(f"❌ Error launching Gradio: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
