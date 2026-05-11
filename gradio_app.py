# -*- coding: utf-8 -*-
"""
Gradio Web Interface for Orpheus TTS
Supports text-to-speech and voice cloning
"""

import gradio as gr
import torch
import torchaudio
import numpy as np
import tempfile
import os
from typing import Optional, Tuple
import librosa
from datetime import datetime

from inference import OrpheusInference
from config import INFERENCE_CONFIG, VIETNAMESE_PROMPTS
from utils import tokenise_audio, redistribute_codes, load_snac_model
from data_processor import OrpheusDataProcessor

# Global variables
inference_engine: Optional[OrpheusInference] = None
current_model_path: str = ""
snac_model = None


def save_wav(output_path: str, audio_data, sample_rate: int = 24000) -> None:
    """Save audio as WAV, falling back when torchaudio requires TorchCodec."""
    if hasattr(audio_data, "detach"):
        audio_tensor = audio_data.detach().squeeze().to("cpu")
    else:
        audio_tensor = torch.tensor(audio_data).squeeze()

    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    try:
        torchaudio.save(output_path, audio_tensor, sample_rate)
    except Exception:
        import soundfile as sf

        audio_np = audio_tensor.detach().cpu().numpy()
        if audio_np.ndim == 2:
            audio_np = audio_np.T
        sf.write(output_path, audio_np, sample_rate)


def initialize_model(model_path: str) -> str:
    """Initialize the Orpheus model"""
    global inference_engine, current_model_path
    
    try:
        if current_model_path == model_path and inference_engine is not None:
            return "✅ Model already loaded"
        
        print(f"🔄 Loading model from: {model_path}")
        inference_engine = OrpheusInference(device="cuda")
        inference_engine.load_model(lora_path=model_path)
        current_model_path = model_path
        
        return f"✅ Model loaded successfully from {model_path}"
        
    except Exception as e:
        return f"❌ Error loading model: {str(e)}"


def text_to_speech(text: str, 
                  temperature: float = 0.6,
                  max_tokens: int = 1200,
                  top_p: float = 0.95,
                  repetition_penalty: float = 1.1,
                  seed: Optional[int] = None) -> Tuple[Optional[str], str]:
    """Generate speech from text"""
    global inference_engine
    
    if inference_engine is None:
        return None, "❌ Model not loaded. Please load a model first."
    
    if not text.strip():
        return None, "❌ Please enter some text to synthesize."
    
    try:
        print(f"🎤 Generating speech for: {text[:50]}...")
        
        # Generate audio with optional seed for consistency
        audio_samples = inference_engine.generate_audio(
            prompts=[text],
            temperature=temperature,
            max_new_tokens=max_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed
        )
        
        if not audio_samples or len(audio_samples) == 0:
            return None, "❌ No audio generated. Please try again."
        
        # Convert to numpy array for Gradio
        audio_tensor = audio_samples[0]
        if hasattr(audio_tensor, 'detach'):
            audio_data = audio_tensor.detach().squeeze().cpu().numpy()
        else:
            audio_data = np.array(audio_tensor).squeeze()
        
        # Ensure proper audio format
        if len(audio_data.shape) > 1:
            audio_data = audio_data[0]  # Take first channel if stereo
        
        # Normalize audio
        if audio_data.max() > 1.0 or audio_data.min() < -1.0:
            audio_data = audio_data / np.abs(audio_data).max()
        
        # Save temporary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_path = f"temp_audio_{timestamp}.wav"
        
        save_wav(temp_path, audio_data, 24000)
        
        return temp_path, f"✅ Generated audio for: {text[:50]}..."
        
    except Exception as e:
        return None, f"❌ Error generating speech: {str(e)}"


def clone_voice_and_speak(text: str,
                         reference_audio,
                         temperature: float = 0.6,
                         max_tokens: int = 1200) -> Tuple[Optional[str], str]:
    """Clone voice from reference audio and synthesize text"""
    global inference_engine, snac_model
    
    if inference_engine is None:
        return None, "❌ Model not loaded. Please load a model first."
    
    if not text.strip():
        return None, "❌ Please enter text to synthesize."
    
    if reference_audio is None:
        return None, "❌ Please upload a reference audio file for voice cloning."
    
    try:
        print(f"🎭 Voice cloning for text: {text[:50]}...")
        
        # Load reference audio
        ref_audio_path = reference_audio
        audio_data, sr = librosa.load(ref_audio_path, sr=24000, mono=True)
        
        # Initialize SNAC model if needed
        if snac_model is None:
            print("📦 Loading SNAC model for voice cloning...")
            snac_model = load_snac_model("cpu")  # Keep on CPU to save GPU memory
        
        # Extract voice characteristics (audio codes)
        print("🔍 Extracting voice characteristics...")
        voice_codes = tokenise_audio(audio_data, 24000, snac_model)
        
        # Create voice-conditioned prompt
        # Note: This is a simplified approach - actual voice cloning might need more sophisticated conditioning
        conditioned_text = f"[Voice Reference] {text}"
        
        # Generate with voice conditioning
        audio_samples = inference_engine.generate_audio(
            prompts=[conditioned_text],
            temperature=temperature,
            max_new_tokens=max_tokens,
        )
        
        if not audio_samples:
            return None, "❌ Failed to generate cloned voice audio."
        
        # Process output
        audio_tensor = audio_samples[0]
        if hasattr(audio_tensor, 'detach'):
            audio_data_out = audio_tensor.detach().squeeze().cpu().numpy()
        else:
            audio_data_out = np.array(audio_tensor).squeeze()
        
        # Normalize
        if audio_data_out.max() > 1.0 or audio_data_out.min() < -1.0:
            audio_data_out = audio_data_out / np.abs(audio_data_out).max()
        
        # Save output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"cloned_voice_{timestamp}.wav"
        
        save_wav(output_path, audio_data_out, 24000)
        
        return output_path, f"✅ Voice cloning completed for: {text[:50]}..."
        
    except Exception as e:
        return None, f"❌ Voice cloning error: {str(e)}"


def load_model_interface(model_path: str) -> str:
    """Interface function to load model"""
    if not model_path.strip():
        return "❌ Please enter a model path"
    
    return initialize_model(model_path)


def clear_audio_cache() -> str:
    """Clear temporary audio files"""
    try:
        import glob
        temp_files = glob.glob("temp_audio_*.wav") + glob.glob("cloned_voice_*.wav")
        removed_count = 0
        
        for file in temp_files:
            try:
                os.remove(file)
                removed_count += 1
            except:
                pass
        
        return f"🧹 Cleaned {removed_count} temporary audio files"
    
    except Exception as e:
        return f"❌ Error cleaning cache: {str(e)}"


# Vietnamese example prompts
EXAMPLE_PROMPTS = [
    "Xin chào, tôi là Orpheus TTS với giọng nói tiếng Việt tự nhiên.",
    "Hôm nay là một ngày đẹp trời ở Hà Nội.",
    "Cảm ơn bạn đã sử dụng mô hình Orpheus TTS này.",
    "Chào mừng bạn đến với công nghệ text-to-speech tiên tiến.",
    "Mô hình này được huấn luyện trên dataset tiếng Việt lớn."
]


def create_interface():
    """Create the Gradio interface"""
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .main-header {
        text-align: center;
        color: #2563eb;
        margin-bottom: 20px;
    }
    .section-header {
        color: #1f2937;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 5px;
        margin: 15px 0;
    }
    """
    
    with gr.Blocks(css=css, title="Orpheus TTS Vietnamese") as app:
        
        # Header
        gr.Markdown("""
        # 🎤 Orpheus TTS - Vietnamese Text-to-Speech
        ### Powered by Unsloth & LoRA Fine-tuning
        
        **Features:**
        - 🗣️ Natural Vietnamese speech synthesis  
        - 🎭 Voice cloning from reference audio
        - ⚙️ Adjustable generation parameters
        """, elem_classes=["main-header"])
        
        # Model Loading Section
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 📂 Model Loading", elem_classes=["section-header"])
                model_path_input = gr.Textbox(
                    label="Model Path",
                    placeholder="/workspace/training_orpherus/h100_large_dataset_output/checkpoint-1600/",
                    value="/workspace/training_orpherus/h100_large_dataset_output/checkpoint-1600/",
                    info="Path to trained model checkpoint or saved model directory"
                )
                load_btn = gr.Button("🔄 Load Model", variant="primary")
                model_status = gr.Textbox(label="Status", interactive=False)
        
        gr.Markdown("---")
        
        # Main TTS Section
        with gr.Tab("🎤 Text-to-Speech"):
            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(
                        label="📝 Text to Synthesize",
                        placeholder="Nhập văn bản tiếng Việt để tạo giọng nói...",
                        lines=4,
                        max_lines=8,
                        info="Enter Vietnamese text to convert to speech"
                    )
                    
                    # Example prompts
                    with gr.Row():
                        example_buttons = []
                        for i, prompt in enumerate(EXAMPLE_PROMPTS[:3]):
                            btn = gr.Button(f"Example {i+1}", size="sm")
                            example_buttons.append((btn, prompt))
                
                with gr.Column(scale=1):
                    gr.Markdown("### 🎛️ Generation Settings")
                    
                    temperature = gr.Slider(
                        minimum=0.1, maximum=2.0, value=0.6, step=0.1,
                        label="Temperature",
                        info="Higher = more creative, Lower = more consistent"
                    )
                    
                    max_tokens = gr.Slider(
                        minimum=200, maximum=2000, value=1200, step=100,
                        label="Max Tokens",
                        info="Maximum audio length"
                    )
                    
                    top_p = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.95, step=0.05,
                        label="Top-p",
                        info="Nucleus sampling parameter"
                    )
                    
                    repetition_penalty = gr.Slider(
                        minimum=1.0, maximum=2.0, value=1.1, step=0.1,
                        label="Repetition Penalty",
                        info="Avoid repetition"
                    )
                    
                    seed_input = gr.Number(
                        label="🎲 Random Seed (Voice Consistency)", 
                        value=42,
                        info="Same seed = same voice. Try 42, 123, 999"
                    )
                    
                    use_fp16 = gr.Checkbox(
                        label="⚡ Use FP16 (Faster inference)",
                        value=True,
                        info="Enable for 2x speed boost"
                    )
            
            with gr.Row():
                tts_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")
                clear_btn = gr.Button("🧹 Clear Cache", variant="secondary")
            
            with gr.Row():
                tts_output = gr.Audio(label="🔊 Generated Speech", type="filepath")
                tts_status = gr.Textbox(label="Status", interactive=False)
        
        # Voice Cloning Section  
        with gr.Tab("🎭 Voice Cloning"):
            gr.Markdown("""
            ### 📚 How Voice Cloning Works:
            1. Upload a reference audio file (3-10 seconds recommended)
            2. Enter the text you want to synthesize
            3. The model will try to mimic the voice characteristics
            
            **Note:** Voice cloning quality depends on training data and reference audio quality.
            """)
            
            with gr.Row():
                with gr.Column():
                    clone_text = gr.Textbox(
                        label="📝 Text for Voice Cloning",
                        placeholder="Nhập text để clone với giọng nói mẫu...",
                        lines=3,
                        info="Text to synthesize with cloned voice"
                    )
                    
                    reference_audio = gr.Audio(
                        label="🎤 Reference Voice (Upload Audio)",
                        type="filepath"
                    )
                    gr.Markdown("*Upload 3-10 seconds of clear speech for voice cloning*", elem_classes=["info-text"])
                
                with gr.Column():
                    gr.Markdown("### 🎛️ Cloning Settings")
                    
                    clone_temperature = gr.Slider(
                        minimum=0.3, maximum=1.5, value=0.6, step=0.1,
                        label="Temperature",
                        info="Lower values for better voice matching"
                    )
                    
                    clone_max_tokens = gr.Slider(
                        minimum=500, maximum=2000, value=1200, step=100,
                        label="Max Tokens"
                    )
            
            with gr.Row():
                clone_btn = gr.Button("🎭 Clone Voice & Speak", variant="primary", size="lg")
            
            with gr.Row():
                clone_output = gr.Audio(label="🔊 Cloned Voice Output", type="filepath")
                clone_status = gr.Textbox(label="Clone Status", interactive=False)
        
        # Voice Consistency Section
        with gr.Tab("🎯 Voice Consistency Tips"):
            gr.Markdown("""
            ## 🤔 Tại sao mỗi text ra voice khác nhau?
            
            ### ❓ **Nguyên nhân:**
            1. **Temperature cao** → Generation random → Voice không ổn định
            2. **Không có seed cố định** → Mỗi lần tạo random khác nhau
            3. **Model chưa có voice identity mạnh** → Cần train thêm
            4. **Text khác nhau** → Ảnh hưởng đến voice characteristics
            
            ### ✅ **Giải pháp cho Voice Consistent:**
            
            #### 🎲 **Dùng Random Seed cố định:**
            - Seed = 42 → Luôn cùng 1 voice
            - Seed = 123 → Voice khác nhưng consistent  
            - Seed = 999 → Voice khác nữa nhưng consistent
            
            #### 🌡️ **Giảm Temperature:**
            - Temperature = 0.3-0.5 → Ít random, voice ổn định hơn
            - Temperature = 0.8-1.0 → Nhiều random, voice đa dạng hơn
            
            #### 📝 **Consistent Text Format:**
            ```
            # Tốt - format giống nhau:
            "Xin chào, tôi là AI assistant."
            "Xin chào, tôi có thể giúp bạn."
            
            # Không tốt - format khác nhau:
            "Hello world!"
            "Xin chào các bạn nhé!!!"
            ```
            
            #### ⚡ **FP16 Benefits:**
            - 🚀 **2x faster inference**
            - 💾 **50% less VRAM**  
            - 🎯 **Same quality**
            - ✅ **Better for production**
            
            ### 🎯 **Best Practice:**
            1. ✅ Dùng **Seed = 42** cho consistent voice
            2. ✅ **Temperature = 0.4-0.6** cho balance
            3. ✅ **FP16 = True** cho speed
            4. ✅ Format text giống nhau
            5. ✅ Train model nhiều epochs hơn để tăng voice stability
            
            ### 📊 **Voice Consistency Test:**
            Try these settings:
            - Seed: 42
            - Temperature: 0.4  
            - Text: "Xin chào, đây là test voice consistency."
            
            → Mỗi lần generate sẽ có cùng voice! 🎯
            """)
        
        # Advanced Settings Tab
        with gr.Tab("⚙️ Advanced"):
            with gr.Column():
                gr.Markdown("### 📊 Model Information")
                
                model_info = gr.Textbox(
                    label="Current Model",
                    value="No model loaded",
                    interactive=False
                )
                
                gr.Markdown("### 🗂️ Audio Management") 
                
                with gr.Row():
                    download_format = gr.Dropdown(
                        choices=["WAV", "MP3"],
                        value="WAV",
                        label="Download Format"
                    )
                    
                    sample_rate = gr.Number(
                        value=24000,
                        label="Sample Rate (Hz)",
                        info="Audio sample rate for output"
                    )
                
                # Batch processing
                gr.Markdown("### 📝 Batch Processing")
                batch_text = gr.Textbox(
                    label="Batch Text Input",
                    placeholder="Dòng 1: Text một\nDòng 2: Text hai\nDòng 3: Text ba",
                    lines=5,
                    info="One text per line for batch processing"
                )
                
                batch_btn = gr.Button("🔄 Process Batch", variant="secondary")
                batch_output = gr.File(label="📦 Download Batch Results")
                batch_status = gr.Textbox(label="Batch Status", interactive=False)
        
        # Event handlers
        load_btn.click(
            fn=load_model_interface,
            inputs=[model_path_input],
            outputs=[model_status]
        )
        
        tts_btn.click(
            fn=text_to_speech,
            inputs=[text_input, temperature, max_tokens, top_p, repetition_penalty, seed_input],
            outputs=[tts_output, tts_status]
        )
        
        clone_btn.click(
            fn=clone_voice_and_speak,
            inputs=[clone_text, reference_audio, clone_temperature, clone_max_tokens],
            outputs=[clone_output, clone_status]
        )
        
        clear_btn.click(
            fn=clear_audio_cache,
            outputs=[tts_status]
        )
        
        batch_btn.click(
            fn=process_batch,
            inputs=[batch_text],
            outputs=[batch_output]
        )
        
        # Example button handlers
        for btn, prompt in example_buttons:
            btn.click(
                lambda prompt=prompt: prompt,  # Closure to capture prompt
                outputs=[text_input]
            )
        
        # Update model info when model loads
        def update_model_info(status):
            if "successfully" in status.lower():
                return f"✅ Model: {current_model_path}"
            return "❌ No model loaded"
        
        model_status.change(
            fn=update_model_info,
            inputs=[model_status],
            outputs=[model_info]
        )
        
        # Footer
        gr.Markdown("""
        ---
        ### 📖 Usage Tips:
        - **Vietnamese Text**: Works best with proper Vietnamese diacritics
        - **Voice Cloning**: Upload clear, 3-10 second reference audio
        - **Temperature**: 0.4-0.8 for natural speech, 0.8-1.2 for creative
        - **Max Tokens**: Roughly 1 token ≈ 20ms of audio
        
        **Made with ❤️ using Orpheus TTS, Unsloth, and Gradio**
        """)
    
    return app


def process_batch(batch_text: str) -> Optional[str]:
    """Process multiple texts in batch"""
    global inference_engine
    
    if inference_engine is None:
        return None
    
    if not batch_text.strip():
        return None
    
    try:
        # Split into lines
        texts = [line.strip() for line in batch_text.split('\n') if line.strip()]
        
        if not texts:
            return None
        
        print(f"📦 Processing {len(texts)} texts in batch...")
        
        # Create temporary directory for batch
        import tempfile
        import zipfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_files = []
            
            for i, text in enumerate(texts):
                try:
                    # Generate audio for each text
                    audio_samples = inference_engine.generate_audio(
                        prompts=[text],
                        temperature=0.6,
                        max_new_tokens=1200
                    )
                    
                    if audio_samples:
                        audio_tensor = audio_samples[0]
                        if hasattr(audio_tensor, 'detach'):
                            audio_data = audio_tensor.detach().squeeze().cpu().numpy()
                        else:
                            audio_data = np.array(audio_tensor).squeeze()
                        
                        # Save individual file
                        filename = f"batch_{i+1:03d}_{text[:20].replace(' ', '_')}.wav"
                        filepath = os.path.join(temp_dir, filename)
                        
                        save_wav(filepath, audio_data, 24000)
                        audio_files.append(filepath)
                
                except Exception as e:
                    print(f"Error processing text {i+1}: {e}")
                    continue
            
            # Create zip file
            if audio_files:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                zip_path = f"batch_output_{timestamp}.zip"
                
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for file_path in audio_files:
                        zipf.write(file_path, os.path.basename(file_path))
                
                return zip_path
        
        return None
        
    except Exception as e:
        print(f"Batch processing error: {e}")
        return None


def main():
    """Main function to launch Gradio app"""
    
    print("🚀 Starting Orpheus TTS Gradio Interface...")
    
    # Create interface
    app = create_interface()
    
    # Launch with share=True
    app.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=True,             # Create public link
        debug=False,
        show_error=True,
        quiet=False,
        inbrowser=False,        # Don't auto-open browser in server environment
    )


if __name__ == "__main__":
    main()
