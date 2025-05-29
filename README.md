# OmniGen Text-to-Image Generator

A web application that generates high-quality images from text descriptions using the OmniGen model.

## Features

- Text-to-image generation using OmniGen
- Adjustable image dimensions
- Customizable guidance scale
- Seed control for reproducible results
- Image download functionality
- User-friendly interface

## Requirements

- Python 3.8+
- CUDA-capable GPU
- 16GB+ RAM recommended

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Enter your text prompt in the text area

4. Adjust the parameters in the sidebar if desired:
   - Image Height
   - Image Width
   - Guidance Scale
   - Seed

5. Click "Generate Image" to create your image

6. Use the "Download Image" button to save the generated image

## Tips for Better Results

- Be specific and detailed in your descriptions
- Include information about lighting, style, and mood
- Mention important details like colors, materials, and composition
- For best results, use prompts that are 50-100 words long

## License

This project is licensed under the MIT License - see the LICENSE file for details. 