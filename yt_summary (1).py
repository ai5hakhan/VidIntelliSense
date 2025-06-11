import streamlit as st
import yt_dlp  # New import for yt-dlp
from haystack.nodes import PromptNode, PromptModel
from haystack.nodes.audio import WhisperTranscriber
from haystack.pipelines import Pipeline
from model_add import LlamaCPPInvocationLayer
import time

st.set_page_config(
    layout="wide"
)

def download_video(url):
    # Configure yt-dlp to download best audio and extract it to an m4a file.
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'downloaded_audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
            'preferredquality': '192',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
         ydl.download([url])
    # Return the downloaded file name.
    return "downloaded_audio.m4a"

def initialize_model(full_path):
    return PromptModel(
        model_name_or_path=full_path,
        invocation_layer_class=LlamaCPPInvocationLayer,
        use_gpu=False,
        max_length=512
    )
    


def initialize_prompt_node(model):
    summary_prompt = "deepset/summarization"
    return PromptNode(model_name_or_path=model, default_prompt_template=summary_prompt, use_gpu=False)

def transcribe_audio(file_path, prompt_node):
    whisper = WhisperTranscriber()
    pipeline = Pipeline()
    pipeline.add_node(component=whisper, name="whisper", inputs=["File"])
    pipeline.add_node(component=prompt_node, name="prompt", inputs=["whisper"])
    output = pipeline.run(file_paths=[file_path])
    return output

def main():
    st.title("YouTube Video Transcriber and Summarizer")
    st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)
    st.subheader('Built with the Llama 2 🦙, Haystack, Streamlit and ❤️')
    st.markdown('<style>h3{color: pink; text-align: center;}</style>', unsafe_allow_html=True)

    # App details
    with st.expander("About the App"):
        st.write("This app allows you to summarize while watching a YouTube video.")
        st.write("Enter a YouTube URL in the input box below and click 'Submit' to start. This app is built by AI Anytime.")

    # Input box for YouTube URL
    youtube_url = st.text_input("Enter YouTube URL")

    # Submit button
    if st.button("Submit") and youtube_url:
        start_time = time.time()  # Start the timer
        
        # Download video/audio using yt-dlp
        file_path = download_video(youtube_url)

        # Initialize model
        full_path = "model/llama-2-7b-32k-instruct.Q4_K_S.gguf"
        model = initialize_model(full_path)
        prompt_node = initialize_prompt_node(model)

        # Transcribe audio
        output = transcribe_audio(file_path, prompt_node)
        end_time = time.time()  # End the timer
        elapsed_time = end_time - start_time

        # Display layout with 2 columns
        col1, col2 = st.columns([1, 1])
        with col1:
            st.video(youtube_url)
        with col2:
            st.header("Summarization of YouTube Video")
            
            st.write(output)
            st.success(output["results"][0])
            

            
            st.write(f"Time taken: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()