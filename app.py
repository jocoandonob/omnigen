import streamlit as st

# Set page config - must be the first Streamlit command
st.set_page_config(
    page_title="Text-to-Image & Image Editor",
    page_icon="🎨",
    layout="wide"
)

# Now import other modules
import torch
from diffusers import OmniGenPipeline
from PIL import Image
import io
import os
import time

# Check CUDA availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    st.warning("⚠️ CUDA is not available. Running on CPU will be very slow!")

# Custom CSS for professional look
st.markdown(
    """
    <style>
    .main-header {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: #666;
        font-size: 1.2rem;
        margin-top: 0;
    }
    .stButton > button {
        background-color: #4F8BF9;
        color: white;
        border-radius: 6px;
        padding: 0.5em 2em;
        font-weight: 600;
        font-size: 1.1em;
        margin-top: 1em;
    }
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
    }
    .image-container {
        display: flex;
        justify-content: center;
        margin: 1.5em 0;
    }
    .image-container img {
        max-width: 500px;
        width: 100%;
        border-radius: 12px;
        box-shadow: 0 2px 16px rgba(0,0,0,0.08);
    }
    .section {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 2em 2em 1em 2em;
        margin-bottom: 2em;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize the models with loading state
@st.cache_resource(show_spinner=False)
def load_t2i_model():
    try:
        # Show loading state
        with st.spinner("Loading text-to-image model... This may take a few minutes on first run."):
            # Text to Image model
            t2i_pipe = OmniGenPipeline.from_pretrained(
                "Shitao/OmniGen-v1-diffusers",
                torch_dtype=torch.float32
            )
            t2i_pipe.to(DEVICE)
            return t2i_pipe
    except ImportError as e:
        st.error(f"Error loading text-to-image model: {str(e)}")
        st.info("Please wait a moment and refresh the page. If the error persists, try reinstalling the requirements.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return None

@st.cache_resource(show_spinner=False)
def load_edit_model():
    try:
        # Show loading state
        with st.spinner("Loading image editing model... This may take a few minutes on first run."):
            # Image Editing model
            edit_pipe = OmniGenPipeline.from_pretrained(
                "Shitao/OmniGen-v1-diffusers",
                torch_dtype=torch.float32
            )
            edit_pipe.to(DEVICE)
            return edit_pipe
    except ImportError as e:
        st.error(f"Error loading image editing model: {str(e)}")
        st.info("Please wait a moment and refresh the page. If the error persists, try reinstalling the requirements.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return None

# Main header
st.markdown(
    """
    <div class="main-header">
        <h1>🎨 OmniGen Studio</h1>
        <p>Generate and edit images with AI. Powered by OmniGen.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Create tabs
tab1, tab2 = st.tabs(["Text to Image", "Image Editing"])

# Text to Image Tab
with tab1:
    # Load text-to-image model
    t2i_pipe = load_t2i_model()
    
    if t2i_pipe is None:
        st.markdown(
            """
            <div class="loading-container">
                <div class="loading-text">
                    <h3>⚠️ Text-to-Image model is still loading...</h3>
                    <p>Please wait a moment and refresh the page.</p>
                    <p>If the error persists, try:</p>
                    <ul>
                        <li>Refreshing the page</li>
                        <li>Reinstalling the requirements</li>
                        <li>Checking your internet connection</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.stop()
        
    with st.container():
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.subheader("Text to Image Generator")
        st.markdown("""
        Generate high-quality images from text descriptions using OmniGen.
        """)

        # Sidebar for parameters
        st.sidebar.header("Text-to-Image Parameters")
        height = st.sidebar.slider("Image Height", 512, 1024, 1024, step=64)
        width = st.sidebar.slider("Image Width", 512, 1024, 1024, step=64)
        guidance_scale = st.sidebar.slider("Guidance Scale", 1.0, 20.0, 3.0, step=0.1)
        num_inference_steps = st.sidebar.slider("Number of Steps", 20, 100, 50, step=1)
        seed = st.sidebar.number_input("Seed", value=111, step=1)

        # Main content
        prompt = st.text_area(
            "Enter your prompt",
            height=120,
            placeholder="Describe the image you want to generate..."
        )

        negative_prompt = st.text_area(
            "Negative prompt (optional)",
            height=60,
            placeholder="Describe what you don't want in the image..."
        )

        if st.button("Generate Image", key="generate_t2i"):
            if prompt:
                with st.spinner("Generating image..."):
                    try:
                        # Generate image
                        generator = torch.Generator(device=DEVICE).manual_seed(seed)
                        image = t2i_pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt if negative_prompt else None,
                            height=height,
                            width=width,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps,
                            generator=generator,
                        ).images[0]
                        
                        # Display image (max width 500px)
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.image(image, caption="Generated Image", use_column_width=False, width=500)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
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
        st.markdown('</div>', unsafe_allow_html=True)

# Image Editing Tab
with tab2:
    # Load image editing model
    edit_pipe = load_edit_model()
    
    if edit_pipe is None:
        st.markdown(
            """
            <div class="loading-container">
                <div class="loading-text">
                    <h3>⚠️ Image Editing model is still loading...</h3>
                    <p>Please wait a moment and refresh the page.</p>
                    <p>If the error persists, try:</p>
                    <ul>
                        <li>Refreshing the page</li>
                        <li>Reinstalling the requirements</li>
                        <li>Checking your internet connection</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.stop()
        
    with st.container():
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.subheader("Image Editor")
        st.markdown("""
        Edit your images using OmniGen. Upload an image and describe the changes you want to make.
        """)

        # Upload image
        uploaded_file = st.file_uploader("Upload an image to edit", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            # Display uploaded image (max width 500px)
            input_image = Image.open(uploaded_file)
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(input_image, caption="Uploaded Image", use_column_width=False, width=500)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Edit prompt
            edit_prompt = st.text_area(
                "Describe the changes you want to make",
                height=120,
                placeholder="Example: Remove the woman's earrings. Replace the mug with a clear glass filled with sparkling iced cola."
            )
            
            # Parameters
            st.sidebar.header("Edit Parameters")
            guidance_scale = st.sidebar.slider("Guidance Scale", 1.0, 10.0, 2.0, step=0.1, key="edit_guidance")
            img_guidance_scale = st.sidebar.slider("Image Guidance Scale", 1.0, 10.0, 1.6, step=0.1)
            edit_seed = st.sidebar.number_input("Seed", value=222, step=1, key="edit_seed")
            
            if st.button("Edit Image"):
                if edit_prompt:
                    with st.spinner("Editing image..."):
                        try:
                            # Format prompt for OmniGen
                            formatted_prompt = f"<img><|image_1|></img> {edit_prompt}"
                            
                            # Generate edited image
                            generator = torch.Generator(device=DEVICE).manual_seed(edit_seed)
                            edited_image = edit_pipe(
                                prompt=formatted_prompt,
                                input_images=[input_image],
                                guidance_scale=guidance_scale,
                                img_guidance_scale=img_guidance_scale,
                                use_input_image_size_as_output=True,
                                generator=generator
                            ).images[0]
                            
                            # Display edited image (max width 500px)
                            st.markdown('<div class="image-container">', unsafe_allow_html=True)
                            st.image(edited_image, caption="Edited Image", use_column_width=False, width=500)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Add download button
                            buf = io.BytesIO()
                            edited_image.save(buf, format="PNG")
                            st.download_button(
                                label="Download Edited Image",
                                data=buf.getvalue(),
                                file_name="edited_image.png",
                                mime="image/png"
                            )
                            
                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")
                else:
                    st.warning("Please enter an edit prompt!")
        st.markdown('</div>', unsafe_allow_html=True)

# Add some helpful information
st.markdown("""
### Tips for better results:
- Be specific and detailed in your descriptions
- Include information about lighting, style, and mood
- Mention important details like colors, materials, and composition
- For best results, use prompts that are 50-100 words long
- Use negative prompts to specify what you don't want in the image
""") 