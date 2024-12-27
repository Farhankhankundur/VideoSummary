import streamlit as st
import moviepy.editor as mp
import speech_recognition as sr
import cv2
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import spacy
from PIL import Image

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load Salesforce's BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Step 1: Extract audio from video
def extract_audio(video_path):
    video = mp.VideoFileClip(video_path)
    audio_path = "audio.wav"
    video.audio.write_audiofile(audio_path)
    return audio_path

# Step 2: Transcribe audio to text
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
    return text

# Step 3: Extract frames from video and analyze visual content using BLIP
def analyze_frames_with_blip(video_path, interval=2):
    cap = cv2.VideoCapture(video_path)
    frames_analysis = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every 'interval' seconds
        if frame_count % (interval * int(cap.get(cv2.CAP_PROP_FPS))) == 0:
            # Convert frame to PIL image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Generate caption using BLIP
            inputs = processor(frame_pil, return_tensors="pt").to("cpu")
            caption = model.generate(**inputs)
            caption_text = processor.decode(caption[0], skip_special_tokens=True)
            
            frames_analysis.append(caption_text)

        frame_count += 1
    
    cap.release()
    return frames_analysis

# Step 4: Generate a narrative summary based on audio and visual content
def generate_narrative(audio_text, frames_analysis):
    narrative_parts = []
    
    # Begin with audio description
    narrative_parts.append("The video opens with the following dialogue:")
    narrative_parts.append(f"'{audio_text}'")
    
    # Analyze visuals if available
    if frames_analysis:
        narrative_parts.append("As the video unfolds, we see the following scenes:")
        for idx, description in enumerate(frames_analysis):
            narrative_parts.append(f"Scene {idx + 1}: {description}")
    
    # Wrap-up narrative
    narrative_parts.append("The audio and visual elements combined provide a rich context, capturing the essence of the video.")
    return " ".join(narrative_parts)

# Step 5: Summarize the narrative using spaCy
def summarize_text_spacy(text):
    doc = nlp(text)
    sentences = list(doc.sents)
    key_sentences = sorted(sentences, key=lambda s: len(s), reverse=True)[:3]  # Extract top 3 longest sentences
    summary = " ".join([str(s).strip() for s in key_sentences])
    return summary

# Step 6: Create a more detailed narrative-like summary of the video content
def detailed_video_story(audio_text, frames_analysis):
    story = []
    
    story.append("### The Story of the Video:")

    # Detailed audio narrative
    story.append("In the opening moments of the video, the dialogue plays out:")
    story.append(f"  '{audio_text}'")
    
    if frames_analysis:
        story.append("The video progresses through different scenes. Here's how it unfolds:")
        for idx, scene in enumerate(frames_analysis):
            story.append(f"  Scene {idx + 1}: {scene}")
    else:
        story.append("There are no significant visual scenes detected.")
    
    story.append("Together, the audio and visuals give a rich context to the events that transpired throughout the video. It encapsulates a blend of spoken content and visual storytelling.")
    return "\n".join(story)

# Main function to process video and generate summary
def process_video(video_file):
    video_path = video_file.name

    # Save uploaded video to a temporary file
    with open(video_path, "wb") as f:
        f.write(video_file.getbuffer())

    # Step 1: Extract audio
    audio_path = extract_audio(video_path)
    
    # Step 2: Transcribe audio
    audio_text = transcribe_audio(audio_path)
    
    # Step 3: Analyze frames using BLIP
    frames_analysis = analyze_frames_with_blip(video_path)
    
    # Step 4: Generate narrative
    narrative = generate_narrative(audio_text, frames_analysis)
    
    # Step 5: Summarize narrative using spaCy
    summary = summarize_text_spacy(narrative)
    
    # Step 6: Provide a detailed narrative-like story
    story = detailed_video_story(audio_text, frames_analysis)
    
    return summary, story

# Streamlit interface
st.set_page_config(page_title="Video AI Assistant 🎥", layout="wide", initial_sidebar_state="expanded")

# Add background color and image
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: url('https://source.unsplash.com/1600x900/?abstract,gradient');
    background-size: cover;
    color: white;
}
[data-testid="stSidebar"] {
    background: #001f3f; /* Dark Navy */
    color: white;
}
[data-testid="stSidebar"] h1 {
    color: #FFD700; /* Gold */
}
button {
    background-color: #FFD700;
    color: #001f3f;
    font-size: 18px;
    border-radius: 8px;
    padding: 10px;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Sidebar for uploading the video
st.sidebar.title("🎥 Video AI Assistant")
st.sidebar.write("Upload a video to generate summaries and narratives based on audio and visuals.")

uploaded_file = st.sidebar.file_uploader("🎬 Upload Video", type=["mp4", "mov", "avi"])

st.sidebar.markdown("#### How to Use:")
st.sidebar.markdown(
    """
    - Upload a video file
    - Click 'Generate Summary'
    - View and interact with your results
    """
)

st.sidebar.info("🚀 Built for content creators, editors, and storytellers!")

# Main layout with columns
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 🌟 **Welcome to the Video AI Assistant**")
    st.image(
        "https://source.unsplash.com/400x300/?film,tech",
        caption="Smart Analysis Powered by AI",
        use_column_width=True,
    )
    st.markdown(
        """
        - **Generate narratives** from audio and video
        - **Analyze video frames** for captions
        - Create compelling **summaries** for your content
        """
    )

    if st.button("🤔 Learn More"):
        st.info("🚧 More features coming soon! Stay tuned!")

with col2:
    if uploaded_file:
        st.video(uploaded_file)  # Display video
        if st.button("📋 Generate Summary"):
            with st.spinner("⏳ Analyzing your video..."):
                summary_result, detailed_result = process_video(uploaded_file)
            st.success("✨ Summary Generated!")
            st.markdown("### Overall Story Summary:")
            st.write(summary_result)
            st.markdown("### Detailed Video Story:")
            st.write(detailed_result)
    else:
        st.warning("📂 Please upload a video to proceed!")

# Footer Section
st.sidebar.markdown("---")
st.sidebar.markdown("📩 **Contact Me**: farhankhankundur@gmail.com")
st.sidebar.markdown("💻 Developed by [Farhan Khan]")
