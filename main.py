import os
from spleeter.separator import Separator

if 'TF_CONFIG' in os.environ:
    del os.environ['TF_CONFIG']

# Configure the AI model
# Use pre-trained 'spleeter: n stems' model (n:2,4,5) provided by Deezer
# model separates audio into 2 components: Vocals and Accompaniment
model_name = 'spleeter:2stems'
separator = Separator(model_name)

def separate_audio(input_filename):
    """
    Function to separate vocals from the audio file.
    
    Args:
        input_filename (str): Path to the input mp3 file.
    """
    print(f"Processing start: {input_filename}")
    
    # Define output directory
    output_directory = 'output'
    
    # Perform separation
    try:
        separator.separate_to_file(input_filename, output_directory)
        print(f"Vocal sepearted in '{output_directory}' folder.")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    # Hard coded
    target_song = "test_song.mp3" 
    
    # Check if file exists before processing
    if os.path.exists(target_song):
        separate_audio(target_song)
    else:
        print(f"File not found: {target_song}. Please place an mp3 file in this folder.")