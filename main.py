import os
import argparse
from spleeter.separator import Separator
from pathlib import Path

if 'TF_CONFIG' in os.environ:
    del os.environ['TF_CONFIG']

def separate_audio(input_filename, model_type):
    """
    Function to separate tracks from audio file
    
    Args:
        input_filename (str): Path to the input mp3 file.
        model_type (str): Spleeter model type (e.g., '2stems', '4stems', '5stems')
    """

    # Configure the AI model
    # Use pre-trained 'spleeter: n stems' model (n:2,4,5) provided by Deezer
    # model separates audio into 2 components: Vocals and Accompaniment
    model_name = f"spleeter:{model_type}"

    print(f"=== Settings ===")
    print(f"Input File: {input_filename}")
    print(f"Model: {model_name}")

    # Load the pre-trained model
    print(f"Loading Spleeter model...")
    try:
        separator = Separator(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Define output directory
    file_stem = Path(input_filename).stem
    output_directory = Path("output") / file_stem / model_type
    
    # Perform separation
    print(f"Processing start...")
    try:
        separator.separate_to_file(input_filename, output_directory)
        print(f"Song successfully separated in {output_directory}")
    except Exception as e:
        print(f"Error occurred during separation: {e}")

if __name__ == "__main__":
    # option parser generate
    parser = argparse.ArgumentParser(description="AI Vocal Separator")

    # model selection option(default: 2stems)
    parser.add_argument(
        '-m', '--model', 
        type=str, 
        default='2stems', 
        choices=['2stems', '4stems', '5stems'],
        help="Select Spleeter model (2stems, 4stems, 5stems)"
    )
    
    # file name option(default: "test_song.mp3")
    parser.add_argument(
        '-i', '--input', 
        type=str, 
        default='test_song.mp3', 
        help="Input audio filename (e.g., test_song.mp3)"
    )

    # parse input options
    args = parser.parse_args()
    
    # Check if file exists before processing
    if os.path.exists(args.input):
        separate_audio(args.input, args.model)
    else:
        print(f"File not found: {args.input}. Please place an mp3 file in this folder.")