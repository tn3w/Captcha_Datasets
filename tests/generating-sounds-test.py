from transformers import pipeline
import scipy.io.wavfile
import torch

# Check if CUDA (GPU) is available
cuda_available = torch.cuda.is_available()

# Print the device being used: 'cuda' if available, otherwise 'cpu'
print("Using device:", "cuda" if cuda_available else "cpu")

# Clear any unused memory from the GPU
torch.cuda.empty_cache()

# Initialize the text-to-audio synthesizer pipeline with the specified model
synthesizer = pipeline("text-to-audio", model="facebook/musicgen-medium", device=0 if cuda_available else -1)

# Define prompts for generating different types of music
chaotic_music = "Chaotic music with dissonant harmonies."
normal_music = "Instrumental music with a clear beat."

# Define a function to generate and save music based on a prompt
def generate_and_save_music(prompt, filename, duration=5):

    # Print the prompt for which music is being generated
    print(f"Generating music for prompt: {prompt}")

    # Generate music using the synthesizer with specified parameters
    music = synthesizer(prompt, forward_params={"do_sample": True, "max_length": duration * 50})

    # Extract audio data and sampling rate from the generated music
    audio_data = music["audio"]
    sampling_rate = music["sampling_rate"]

    # Check if audio data is valid and not empty
    if audio_data is not None and len(audio_data) > 0:
        audio_data = audio_data.squeeze() # Remove any singleton dimensions from the audio data
        audio_data = (audio_data * 32767).astype("int16") # Scale audio data to int16 format for WAV

        # Save the audio data to a WAV file with the specified filename and sampling rate
        scipy.io.wavfile.write(filename, rate=sampling_rate, data=audio_data)

        # Print a message indicating the file has been saved
        print(f"Generated and saved {filename}")
    else:

        # Print an error message if audio generation failed
        print(f"Failed to generate audio for prompt: {prompt}")


# Generate and save 10 clips of chaotic music
for i in range(10):
    generate_and_save_music(chaotic_music, f"mysterious_clip_{i+1}.wav", duration=5)

# Generate and save 10 clips of normal music
for i in range(10):
    generate_and_save_music(normal_music, f"normal_clip_{i+1}.wav", duration=5)  
