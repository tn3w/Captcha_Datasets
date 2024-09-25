from melo.api import TTS

# Set parameters
speed = 1.0  # Speed of speech
device = 'auto'  # Automatically use GPU if available

# Initialize the TTS model
model = TTS(language='EN', device=device)
speaker_ids = model.hps.data.spk2id

# Generate audio files with different texts
texts = ["Sound 1", "Sound 2", "Sound 3", "Sound 4"]
accents = 'EN-US'  # List of accents

for i, text in enumerate(texts):
    output_path = f'Sound_{i + 1}.wav'
    model.tts_to_file(text, speaker_ids['EN-US'], output_path, speed=speed)
