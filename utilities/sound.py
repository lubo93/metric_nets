import numpy as np
import wave
from io import BytesIO

def wave_number_to_frequency(k, speed_of_sound=343):
    """Convert wave number to frequency in Hz."""
    return (k * speed_of_sound) / (2 * np.pi)

def generate_tone(frequency, duration=1.0, sample_rate=44100):
    """Generate a guitar-like tone using the Karplus-Strong algorithm."""
    # Determine the number of samples in the buffer
    N = int(sample_rate / frequency)
    
    # Initialize a noise burst for the buffer (white noise)
    buffer = np.random.uniform(-1, 1, N)
    
    # Create an output array for the generated tone
    tone = np.zeros(int(sample_rate * duration))
    
    # Initialize the Karplus-Strong buffer processing
    for i in range(len(tone)):
        tone[i] = buffer[i % N]
        
        # Apply the simple low-pass filter by averaging
        avg = 0.5 * (buffer[i % N] + buffer[(i + 1) % N])
        
        # Update the buffer with the averaged value (this simulates decay)
        buffer[i % N] = avg * 0.996  # Damping factor close to 1 to control decay
    
    return tone

# Function to convert the tone to WAV format and return as BytesIO object
def create_wav_file(tone, sample_rate=44100):
    """Convert the tone to WAV format and return as BytesIO object."""
    # Scale the tone to 16-bit PCM format
    tone = (tone * 32767).astype(np.int16)
    
    # Create an in-memory bytes buffer
    buffer = BytesIO()
    
    # Write the tone to the buffer as a WAV file
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes for 16-bit PCM
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(tone.tobytes())
    
    buffer.seek(0)  # Rewind the buffer to the beginning
    return buffer
