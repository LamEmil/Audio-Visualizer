# Audio Visualizer Exporter

This Python script generates customizable audio visualizer videos from an audio file. It uses PySide6 for the user interface, Pygame for rendering the visuals headlessly, Librosa for audio analysis, and FFmpeg for encoding the final video output.

## Features

* **Graphical User Interface (GUI):** Easy-to-use interface built with PySide6 to configure all options.
* **Audio Input:** Supports common audio formats (MP3, WAV, OGG, FLAC) via Librosa.
* **Customizable Visuals:**
    * **Song Title Display:** Show the song title prominently.
    * **Progress Bar:** Visual representation of audio playback time.
    * **3D Typography Mountain Grid:** A dynamic grid in the bottom-left that reacts to the audio spectrum, with "AUDIO" text.
    * **Stereo Volume Bars:** Left and Right channel volume indicators.
    * **Audio Waveform Display:** Shows the shape of the audio wave.
    * **Spectral Waves:** Bar graph representing audio frequencies.
    * **Blank Video Box:** A designated area at the top, rendered with the background color, where users could potentially overlay their own video content using a separate video editor.
* **Configurable Output:**
    * **Resolution:** Set custom width and height for the output video.
    * **FPS (Frames Per Second):** Define the video's frame rate.
    * **Colors:** Customize background, foreground, accent, text, video border, and visualizer box background colors.
    * **Font:** Choose from available system fonts for text elements.
* **Chunked Video Export:** Processes and exports the video in manageable chunks to handle long audio files and conserve resources. Output is in MP4 format.
* **FFmpeg Integration:** Utilizes FFmpeg for efficient video encoding from generated frames and the original audio.
* **Live Preview Mockup:** The UI includes a panel that updates to show a static mockup of how the visual elements will be arranged based on current settings.

## Prerequisites

* **Python 3.x**
* **FFmpeg:** Must be installed and accessible via the system's PATH, or the full path to the `ffmpeg` executable must be provided in the UI.
* **Python Libraries:**
    * `pygame`
    * `librosa`
    * `numpy`
    * `PySide6`
    * `Pillow` (PIL Fork)

## Installation

1.  **Install Python:** If you don't have Python, download and install it from [python.org](https://www.python.org/).
2.  **Install FFmpeg:** Download and install FFmpeg from [ffmpeg.org](https://ffmpeg.org/). Ensure it's added to your system's PATH or note the installation location of `ffmpeg.exe` (Windows) or `ffmpeg` (macOS/Linux).
3.  **Install Python Libraries:**
    Open a terminal or command prompt and run:
    ```bash
    pip install pygame librosa numpy PySide6 Pillow
    ```
4.  **Download the Script:** Save the provided Python script as `gen.py` (or any other `.py` name) in a directory of your choice.

## How to Run

1.  Navigate to the directory where you saved the script using your terminal or command prompt.
2.  Run the script using Python:
    ```bash
    python gen.py
    ```
3.  The "Audio Visualizer Exporter" GUI window will appear.

## Using the Interface

* **Audio File:** Click "Browse" to select the audio file you want to visualize.
* **Song Title:** Enter the title of the song. This will be displayed in the video. Defaults to "Audio Visualizer".
* **Chunk Duration (s):** Duration in seconds for each video chunk. Shorter chunks use less memory during processing but result in more individual files.
* **FFmpeg Path:** Path to the FFmpeg executable. Defaults to "ffmpeg" (assumes it's in your system PATH). If not, click "Browse" to locate it.
* **Resolution:** Set the output video width and height in pixels.
* **FPS:** Set the frames per second for the output video.
* **Font:** Select a font from the dropdown list for text elements in the video.
* **Color Buttons (Background, Foreground, etc.):** Click these buttons to open a color picker and choose colors for different visual elements.
    * `Background`: Overall background color.
    * `Foreground`: Primary color for some visual elements (e.g., 3D grid lines, volume bars).
    * `Accent`: Secondary color for visual elements (e.g., progress bar fill, waveform, spectral waves).
    * `Text`: Color for the song title and other text.
    * `Border`: Color for the borders of the main visualizer boxes.
    * `Box BG`: Background color for the content boxes.
* **Example Output Preview:** This area on the right shows a static mockup that updates as you change the settings, giving an idea of the layout and colors.
* **Run Visualizer:** Once all settings are configured, click this button to start the video generation process. The GUI might become unresponsive during this time as processing happens in a separate thread. Progress will be printed to the console/terminal.

## Output

* The script will generate video chunks named `audio_visualizer_output_chunk_X.mp4` (where `X` is the chunk number) in the same directory where the script is located.
* Temporary frame image files will be created during processing for each chunk and then deleted automatically.

## How It Works

1.  The user configures visualization parameters through the PySide6 GUI.
2.  When "Run Visualizer" is clicked, the main processing logic starts in a separate thread.
3.  Pygame is initialized in a headless mode (no window is actively displayed for Pygame itself during rendering).
4.  The selected audio file is loaded using Librosa, and its duration and sample rate are determined.
5.  The script iterates through the audio duration, frame by frame, based on the selected FPS.
6.  For each frame's corresponding time point in the audio:
    * Audio features (amplitude, spectrum, waveform data) are extracted using Librosa.
    * A Pygame surface is filled with the background color.
    * Various visual elements (title, progress bar, 3D typography grid, volume bars, waveform, spectral waves) are drawn onto the surface using the extracted audio features and configured colors/fonts.
7.  Each rendered Pygame surface (frame) is converted to an image array (using `pygame.surfarray` and `numpy`) and stored.
8.  When the number of frames corresponding to the `Chunk Duration` is collected (or the audio ends):
    * The collected frames are saved as individual PNG images to a temporary directory.
    * FFmpeg is called via a `subprocess` to:
        * Take the sequence of PNG images.
        * Take the corresponding segment of the original audio file.
        * Combine them into an MP4 video file for that chunk.
    * The temporary image files for the chunk are deleted.
9.  This process repeats for all chunks until the entire audio is visualized.
10. Once completed, console messages will indicate the process is finished and the number of chunks generated. The "Run Visualizer" button in the GUI will be re-enabled.

## Notes

* The rendering process can be CPU and time-intensive, especially for long audio files, high resolutions, or high FPS.
* Ensure FFmpeg is correctly installed and accessible. If you encounter FFmpeg errors, double-check the path provided in the UI.
* Font availability depends on the fonts installed on your system. If a selected font is not found, a default Pygame font will be used.
* The "blank video box" at the top is simply a styled rectangle. If you wish to add actual video content there, you would need to do so in a post-processing step using a video editing tool, compositing it over the output from this script.