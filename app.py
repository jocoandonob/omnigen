import streamlit as st
import torch
from diffusers import OmniGenPipeline
from PIL import Image
import io

print("=================OmniGenPipeline is available.===============")

# Set page config
st.set_page_config(
    page_title="OmniGen Text-to-Image Generator",
    page_icon="🎨",
    layout="wide"
)

# Title and description
st.title("🎨 OmniGen Text-to-Image Generator")
st.markdown("""
This application uses OmniGen to generate high-quality images from text descriptions.
Simply enter your prompt and adjust the parameters to create your image.
""")

# Initialize the model
@st.cache_resource
def load_model():
    pipe = OmniGenPipeline.from_pretrained(
        "Shitao/OmniGen-v1-diffusers",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    pipe.to("cpu")
    return pipe

# Sidebar for parameters
st.sidebar.header("Parameters")
height = st.sidebar.slider("Image Height", 512, 1024, 512, step=64)
width = st.sidebar.slider("Image Width", 512, 1024, 512, step=64)
guidance_scale = st.sidebar.slider("Guidance Scale", 1.0, 10.0, 3.0, step=0.1)
seed = st.sidebar.number_input("Seed", value=111, step=1)

# Main content
prompt = st.text_area(
    "Enter your prompt",
    height=150,
    placeholder="Describe the image you want to generate..."
)

if st.button("Generate Image"):
    if prompt:
        with st.spinner("Generating image..."):
            try:
                # Load model
                pipe = load_model()
                
                # Generate image
                generator = torch.Generator(device="cpu").manual_seed(seed)
                image = pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    generator=generator,
                ).images[0]
                
                # Display image
                st.image(image, caption="Generated Image", use_column_width=True)
                
                # Add download button
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                st.download_button(
                    label="Download Image",
                    data=buf.getvalue(),
                    file_name="generated_image.png",
                    mime="image/png"
                )
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a prompt first!")

# Add some helpful information
st.markdown("""
### Tips for better results:
- Be specific and detailed in your descriptions
- Include information about lighting, style, and mood
- Mention important details like colors, materials, and composition
- For best results, use prompts that are 50-100 words long
""") 