


# from huggingface_hub import login
# login(token="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") 
# import streamlit as st
# st.set_page_config(page_title="AI Story Generator", layout="wide")
# # CHANGED: Remove llama_cpp import
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# import torch  # Added
# import json
# from datetime import datetime
# import re
# import os
# import nltk
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from nltk.tokenize import word_tokenize
# # Load model directly


# # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
# # model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

# # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
# # model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
# # --- Configuration ---
# # CHANGED: Use Hugging Face model name instead of GGUF path
# #MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
# FEEDBACK_DIR = "feedback_data"
# os.makedirs(FEEDBACK_DIR, exist_ok=True)

# # --- Setup ---
# nltk.download('punkt')

# # --- Model Loading ---
# # @st.cache_resource
# # def load_model():
# #     try:
# #         # CHANGED: Load via transformers
# #         model = AutoModelForCausalLM.from_pretrained(
# #             model,
# #             device_map="auto",          # Auto-select GPU/CPU
# #             torch_dtype=torch.float16,   # Use FP16 for lower memory
# #             # load_in_4bit=True,         # Uncomment for 4bit quantization if using <16GB VRAM
# #         )
# #         #tokenizer = AutoTokenizer.from_pretrained(model)
# #         return model, tokenizer
# #     except Exception as e:
# #         st.error(f"❌ Failed to load model: {str(e)}")
# #         st.stop()

# # model, tokenizer = load_model()

# @st.cache_resource
# def load_model():
#     tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
#     model = AutoModelForCausalLM.from_pretrained(
#         "mistralai/Mistral-7B-Instruct-v0.1",
#         device_map="auto",
#         torch_dtype=torch.float16,
#     )
#     return model, tokenizer

# model, tokenizer = load_model()

# # --- Core Functions ---
# def generate_story(prompt, max_length=400):
#     try:
#         # CHANGED: Use proper Mistral instruction format
#         # messages = [
#         #     {"role": "user", "content": "You are a creative writer. Generate a complete story with a beginning, middle, and satisfying ending with a twist."},
#         #     {"role": "user", "content": prompt}
#         # ]
#         messages = [
#     {
#         "role": "user",
#         "content": "You are a creative writer. Generate a complete story with a beginning, middle, and satisfying ending with a twist.\n\n" + prompt
#     }
# ]

#         # CHANGED: Use transformers' built-in chat template
#         formatted_prompt = tokenizer.apply_chat_template(
#             messages, 
#             tokenize=False, 
#             add_generation_prompt=True
#         )
        
#         # CHANGED: Use transformers pipeline for generation
#         pipe = pipeline(
#             "text-generation",
#             model=model,
#             tokenizer=tokenizer,
#             device_map="auto",
#             torch_dtype=torch.float16,
#         )

#         output = pipe(
#             formatted_prompt,
#             max_new_tokens=max_length,
#             temperature=0.8,
#             top_p=0.9,
#             do_sample=True,
#             pad_token_id=tokenizer.eos_token_id
#         )
        
#         story = output[0]["generated_text"].replace(formatted_prompt, "").strip()
        
#         # Post-processing remains similar
#         story = re.sub(r'\.{2,}', '.', story)
#         story = re.sub(r'\s+', ' ', story)
#         story = post_preprocessing(story)
        
#         return story

#     except Exception as e:
#         st.error(f"🚨 Story generation failed: {str(e)}")
#         return None


# # --- Corrected Post-Processing ---
# # def post_preprocessing(story):
# #     sentences = re.split(r'(?<=[.!?])\s+', story)  # Split into sentences properly
# #     complete_story = ' '.join(sentences)           # Recombine to ensure spacing
# #     if not complete_story.endswith(('.', '!', '?')):
# #         complete_story += '.'                     # Add period if missing
# #     return complete_story.strip()



# def post_preprocessing(story):
#     sentences = re.split(r'(?<=[.!?])\s+', story)
#     complete_story = ' '.join(sentences)

#     # If the last sentence is cut off (too short or no ending punctuation)
#     if not complete_story.endswith(('.', '!', '?')):
#         complete_story += '.'

#     return complete_story.strip()

# # --- Feedback Functions ---
# def save_feedback(data):
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"{FEEDBACK_DIR}/feedback_{timestamp}.json"
#     try:
#         with open(filename, "w") as f:
#             json.dump(data, f, indent=2)
#         return True
#     except Exception as e:
#         st.error(f"⚠️ Failed to save feedback: {str(e)}")
#         return False

# def export_training_data(min_rating=4, output_file="train.txt"):
#     if not os.path.exists(FEEDBACK_DIR):
#         return False
#     high_quality = []
#     for file in os.listdir(FEEDBACK_DIR):
#         with open(os.path.join(FEEDBACK_DIR, file), "r") as f:
#             data = json.load(f)
#             if data["rating"] >= min_rating:
#                 text = data["story"].strip().replace("\n", " ")
#                 high_quality.append(text)
#     if high_quality:
#         with open(output_file, "w") as f:
#             for s in high_quality:
#                 f.write(s + "\n")
#         return True
#     return False

# def calculate_bleu(candidate, min_rating=4):
#     if not os.path.exists(FEEDBACK_DIR):
#         return None
#     references = []
#     for file in os.listdir(FEEDBACK_DIR):
#         with open(os.path.join(FEEDBACK_DIR, file), "r") as f:
#             data = json.load(f)
#             if data["rating"] >= min_rating:
#                 ref_tokens = word_tokenize(data["story"].lower())
#                 references.append(ref_tokens)
#     if references:
#         cand_tokens = word_tokenize(candidate.lower())
#         smoothie = SmoothingFunction().method4
#         score = sentence_bleu(references, cand_tokens, smoothing_function=smoothie)
#         return score
#     return None

# # --- Streamlit UI ---


# st.markdown("""
#     <style>
#     .main {
#         background-color: #ffe6f0;
#     }
#     .stButton > button {
#         background-color: #ff66b2;
#         color: white;
#         font-weight: bold;
#     }
#     .stTitle {
#         color: #ff3399;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# st.title("Adaptive Story Generator (Mistral-7B GGUF) - by Phebe")

# with st.expander("ℹ️ Participant Information & Consent"):
#     st.markdown("""
#     **Purpose of the Study**:
#     This app is part of a research project aimed at developing an adaptive AI storytelling system.

#     **What Your Participation Involves**:
#     - You will interact with the AI to generate stories.
#     - You can rate the quality of the stories.
#     - Your feedback will help improve the system and generate high quality stories.

#     **Data Usage**:
#     - Your input and ratings will be recorded.
#     - Data will be used for research purposes only.

#     **Confidentiality**:
#     - All data is anonymised.
#     - Participation is voluntary and you can withdraw at any time.

#     By ticking the consent box, you agree to participate under these terms.
#     """)

# prompt_type = st.radio("Choose Prompt Style", ["Structured", "Open-Ended"], horizontal=True)
# length_choice = st.radio("Story Length", ["Short", "Medium", "Long"], horizontal=True)
# length_map = {"Short": 200, "Medium": 300, "Long": 350}

# if prompt_type == "Structured":
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         character = st.text_input("Character", "A young inventor")
#     with col2:
#         setting = st.text_input("Setting", "a steampunk city")
#     with col3:
#         conflict = st.text_input("Conflict", "discovers a dangerous conspiracy")

#     prompt = (
#         f"Write a story in 3 paragraphs about: {character} lives in {setting}. One day, they {conflict}. "
#         "As the character gets closer to the truth, they must choose between doing the right thing or protecting themselves. "
#         "End the story with a twist and satisfying end."
#     )
# else:
#     raw_idea = st.text_area("Describe your story idea:", "A dragon who forgets how to breathe fire...")
#     prompt = (
#         f"Write a detailed story in 3 paragraphs based on this idea: {raw_idea}. "
#         "Include strong character development and an unexpected twist at the end."
#     )

# if st.button("Generate Story ✍️"):
#     if not prompt.strip():
#         st.warning("Please enter a story idea!")
#         st.stop()
#     with st.spinner("Generating your story..."):
#         progress_bar = st.progress(10)
#         story_placeholder = st.empty()
#         try:
#             story = generate_story(prompt, max_length=length_map[length_choice])
#         except Exception as e:
#             st.error(f"⚠️ Generation failed: {str(e)}")
#             st.stop()
#         progress_bar.progress(80)
#         story_placeholder.write(story)
#         progress_bar.progress(100)
#         st.session_state.story = story
#         st.session_state.prompt = prompt
#         st.session_state.prompt_type = prompt_type

# # if "story" in st.session_state and st.session_state.story:
# #     st.subheader("Rate This Story 💖")
# #     rating = st.slider("How would you rate the story?", 1, 5, 3)
# #     consent = st.checkbox("I consent to the use of my feedback for this project.")
# #     if consent and st.button("Submit Rating"):
# #         feedback_data = {
# #             "prompt": st.session_state.prompt,
# #             "story": st.session_state.story,
# #             "rating": rating,
# #             "prompt_type": st.session_state.prompt_type,
# #             "timestamp": datetime.now().isoformat(),
# #             "model": "mistral-7b-instruct",
# #             "version": "v0.1"
# #         }
# #         if save_feedback(feedback_data):
# #             st.toast("Feedback saved successfully!", icon="✅")
# #             del st.session_state.story

# # if "story" in st.session_state and st.session_state.story:
# #     st.subheader("Rate This Story 💖")
# #     st.markdown("**Your Generated Story:**")
# #     #st.info(st.session_state.story)  # Keep the story visible
    
# #     rating = st.slider("How would you rate the story?", 1, 5, 3)
# #     consent = st.checkbox("I consent to the use of my feedback for this project.")
    
# #     if consent and st.button("Submit Rating"):
# #         feedback_data = {
# #             "prompt": st.session_state.prompt,
# #             "story": st.session_state.story,
# #             "rating": rating,
# #             "prompt_type": st.session_state.prompt_type,
# #             "timestamp": datetime.now().isoformat(),
# #             "model": "mistral-7b-instruct",
# #             "version": "v0.1"
# #         }
# #         if save_feedback(feedback_data):
# #             st.success("✅ Feedback saved successfully!")
# #             st.session_state.feedback_saved = True  # Mark feedback as saved

# #     # Option to clear only after saving
# #     if st.session_state.get("feedback_saved", False):
# #         if st.button("Clear Story and Generate New"):
# #             for key in ["story", "prompt", "prompt_type", "feedback_saved"]:
# #                 if key in st.session_state:
# #                     del st.session_state[key]
# #             st.session_state.pop(key, None)

# if "story" in st.session_state and st.session_state.story:
#     st.subheader("Rate This Story 💖")
#     st.markdown("**Your Generated Story:**")
#     st.info(st.session_state.story)  # Keep the story visible
    
#     rating = st.slider("How would you rate the story?", 1, 5, 3)
#     consent = st.checkbox("I consent to the use of my feedback for this project.")
    
#     if consent and st.button("Submit Rating"):
#         feedback_data = {
#             "prompt": st.session_state.prompt,
#             "story": st.session_state.story,
#             "rating": rating,
#             "prompt_type": st.session_state.prompt_type,
#             "timestamp": datetime.now().isoformat(),
#             "model": "mistral-7b-instruct",
#             "version": "v0.1"
#         }
#         if save_feedback(feedback_data):
#             st.success("✅ Feedback saved successfully!")
#             st.session_state.feedback_saved = True  # Mark feedback as saved

#     # Option to clear only after saving
#     if st.session_state.get("feedback_saved", False):
#         if st.button("Clear Story and Generate New"):
#             # Corrected: Remove session keys properly
#             for key in ["story", "prompt", "prompt_type", "feedback_saved"]:
#                 st.session_state.pop(key, None)
#             st.experimental_rerun()  # Refresh the app state (if supported in your version)



# # Export button
# st.divider()
# st.markdown("### Export for Fine-Tuning")
# if st.button("Export high-rated stories to train.txt"):
#     if export_training_data():
#         st.success("Exported successfully to train.txt ✅")
#     else:
#         st.warning("No high-rated stories to export yet.")

# # BLEU Score Evaluation
# st.divider()
# st.markdown("### Evaluate BLEU Score")
# if "story" in st.session_state:
#     bleu_score = calculate_bleu(st.session_state.story)
#     with st.expander("ℹ️ BLEU (Bilingual Evaluation Understudy)"): 
#         st.markdown("""
#      **What is BLEU?** 
#         BLEU is an algorithm for evaluating the quality of text which has been machine-translated from one natural language to another. 
#         Quality is considered to be the correspondence between a machine’s output and that of a human.
#         The closer a machine translation is to a professional human translation the better it is.
#         BLEU was one of the first metrics to claim a high correlation with human judgements of quality 
#         Scores are calculated for individual translated segments—generally sentences—by comparing them with a set of good quality reference translations.
#     """)
#     if bleu_score is not None:
#         st.metric("BLEU Score", f"{bleu_score:.2f}")
#     else:
#         st.info("No high-rated stories available for BLEU comparison yet.")

# Full Streamlit App for AI Story Generation with Feedback Collection
# Uses mistralai/Mistral-7B-Instruct-v0.1 via Hugging Face Transformers

# Full Streamlit App for AI Story Generation with Feedback Collection
# Uses mistralai/Mistral-7B-Instruct-v0.1 via Hugging Face Transformers
 

# import streamlit as st
# import json
# import os
# import re
# from datetime import datetime
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from nltk.tokenize import word_tokenize
# import nltk
# nltk.download('punkt')

# from mlx_lm import load
# from mlx_lm.generate import generate

# # --------------------
# # Setup
# # --------------------
# MODEL_PATH = "mlx_model"  # after conversion: `mlx_lm.convert --hf-path mistralai/Mistral-7B-Instruct-v0.1 -q --q-bits 4`
# FEEDBACK_DIR = "feedback_data"
# os.makedirs(FEEDBACK_DIR, exist_ok=True)

# @st.cache_resource
# def load_mlx():
#     return load(MODEL_PATH)

# model, tokenizer = load_mlx()

# # --------------------
# # Story Generation (Fixed)
# # --------------------


# # def generate_story(prompt, max_length=256):
# #     system_msg = "You are a creative writer. Generate coherent stories with characters, conflict, and a twist ending."
# #     full_prompt = f"""<s>[INST] <<SYS>>{system_msg}<</SYS>>\n{prompt} [/INST]"""

# #     output = model.create_completion(
# #         prompt=full_prompt,
# #         max_tokens=max_length,
# #         temperature=0.8,
# #         top_p=0.9,
# #         top_k=40,
# #         repeat_penalty=1.15,
# #         stop=["</s>", "INST"]
# #     )

# #     story = output["choices"][0]["text"].strip()
# #     story = re.sub(r'\.{2,}', '.', story)
# #     story = re.sub(r'\s+', ' ', story)
# #     return story
# def generate_story(prompt, max_length=400):
#     try:
#         system_msg = "You are a creative writer. Generate coherent stories with characters, conflict, and a twist ending."
#         full_prompt = f"""<s>[INST] <<SYS>>{system_msg}<</SYS>>\n{prompt} [/INST]"""
#         generated = generate(
#             model=model,
#             tokenizer=tokenizer,
#             prompt=full_prompt,
#             max_tokens=max_length
#         )
#         return generated.replace(full_prompt, "").strip()
#     except Exception as e:
#         st.error(f"⚠️ Generation failed: {str(e)}")
#         return None

# # --------------------
# # Feedback Logging
# # --------------------
# def save_feedback(data):
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = os.path.join(FEEDBACK_DIR, f"feedback_{timestamp}.json")
#     try:
#         with open(filename, "w") as f:
#             json.dump(data, f, indent=2)
#         return True
#     except Exception as e:
#         st.error(f"⚠️ Feedback save failed: {str(e)}")
#         return False

# def calculate_bleu(candidate, min_rating=4):
#     references = []
#     for file in os.listdir(FEEDBACK_DIR):
#         with open(os.path.join(FEEDBACK_DIR, file), "r") as f:
#             data = json.load(f)
#             if data.get("rating", 0) >= min_rating:
#                 ref_tokens = word_tokenize(data["story"].lower())
#                 references.append(ref_tokens)
#     if references:
#         cand_tokens = word_tokenize(candidate.lower())
#         smoothie = SmoothingFunction().method4
#         return sentence_bleu(references, cand_tokens, smoothing_function=smoothie)
#     return None

# # --------------------
# # Streamlit UI
# # --------------------

# st.title("📖 AI Story Generator (MLX Optimized)")

# prompt_type = st.radio("Choose Prompt Style", ["Structured", "Open-Ended"], horizontal=True)
# length_choice = st.radio("Story Length", ["Short", "Medium", "Long"], horizontal=True)
# length_map = {"Short": 200, "Medium": 300, "Long": 400}

# genre = st.selectbox("Genre", ["Fantasy", "Mystery", "Sci-Fi", "General"])

# if prompt_type == "Structured":
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         character = st.text_input("Character", "A young inventor")
#     with col2:
#         setting = st.text_input("Setting", "a steampunk city")
#     with col3:
#         conflict = st.text_input("Conflict", "discovers a dangerous conspiracy")
#     prompt = f"{character} in {setting} who {conflict}. End with a twist."
# else:
#     prompt = st.text_area("Your story idea:", "A dragon who forgets how to breathe fire...")

# if st.button("✨ Generate Story"):
#     full_prompt = f"Genre: {genre}\nYou are a creative writer. Generate a story with a twist ending.\n{prompt}"
#     with st.spinner("Crafting your story..."):
#         story = generate_story(full_prompt, max_length=length_map[length_choice])
#     st.session_state.story = story
#     st.text_area("Generated Story", value=story, height=300)

# if "story" in st.session_state:
#     st.subheader("Rate This Story 💖")
#     rating = st.slider("Rate (1-5)", 1, 5, 3)
#     comments = st.text_input("Comments (optional)")
#     if st.button("Submit Feedback"):
#         data = {
#             "prompt": prompt,
#             "story": st.session_state.story,
#             "rating": rating,
#             "genre": genre,
#             "comments": comments,
#             "timestamp": datetime.now().isoformat()
#         }
#         if save_feedback(data):
#             st.success("Thanks! Your feedback was saved.")

# st.divider()
# if st.button("📊 Show BLEU Score") and "story" in st.session_state:
#     score = calculate_bleu(st.session_state.story)
#     if score:
#         st.metric("BLEU Score", f"{score:.2f}")
#     else:
#         st.info("Not enough rated stories for comparison.")




###########################
# Full Streamlit App for AI Story Generation with Feedback Collection using MLX (Apple Silicon Optimized)

import streamlit as st
st.set_page_config(page_title="Adaptive AI Story Generator", layout="wide")
import json
import os
import re
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

from mlx_lm import load
from mlx_lm.generate import generate

# --------------------
# Setup
# --------------------
MODEL_PATH = "mlx_model"  # after conversion: `mlx_lm.convert --hf-path mistralai/Mistral-7B-Instruct-v0.1 -q --q-bits 4`
FEEDBACK_DIR = "feedback_data"
os.makedirs(FEEDBACK_DIR, exist_ok=True)

@st.cache_resource
def load_mlx():
    return load(MODEL_PATH)

model, tokenizer = load_mlx()

# --------------------
# Story Generation (Enhanced Prompt)
# --------------------
# def generate_story(prompt, max_length=400):
#     try:
#         system_msg = "You are a creative writer. Generate coherent stories with characters, conflict, and a twist ending."
#         formatted_prompt = f"<s>[INST] <<SYS>>{system_msg}<</SYS>>\n{prompt} [/INST]"
#         generated = generate(
#             model=model,
#             tokenizer=tokenizer,
#             prompt=formatted_prompt,
#             max_tokens=max_length
#         )
#         return generated.replace(formatted_prompt, "").strip()
#     except Exception as e:
#         st.error(f"⚠️ Generation failed: {str(e)}")
#         return None
    


# --------------------
# Post-Processing Fix for Abrupt Endings + Multi-Pass
# --------------------
def post_process(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'(\.|\?|!){2,}', r'\1', text)  # Fix multiple punctuation
    sentences = re.split(r'(?<=[.!?]) +', text)

    # Trim trailing incomplete or low-info sentences
    if sentences:
        while sentences and (len(sentences[-1].split()) < 5 or not sentences[-1][-1] in '.!?'):
            sentences.pop()

    return ' '.join(sentences).strip()

# --------------------
# Story Generation (Enhanced Prompt with Continuation)
# --------------------
def generate_story(prompt, max_length=400):
    try:
        system_msg = f"You are a creative writer. Generate coherent stories with characters, conflict, and a twist ending. Make sure the story is complete and ends with a strong, conclusive final sentence. It should fit within roughly {max_length} tokens."
        formatted_prompt = f"<s>[INST] <<SYS>>{system_msg}<</SYS>>\n{prompt} [/INST]"
        generated = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_length
        )
        story = generated.replace(formatted_prompt, "").strip()
        processed = post_process(story)

        # If post-processed story is too short or ends weirdly, try continuation
        if len(processed.split()) < max_length * 0.5:
            continuation_prompt = processed + " Continue and end this story with a strong conclusion."
            second_pass = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=continuation_prompt,
                max_tokens=128
            )
            processed += " " + post_process(second_pass.replace(continuation_prompt, "").strip())

        return processed

    except Exception as e:
        st.error(f"⚠️ Generation failed: {str(e)}")

# --------------------
# Feedback Logging
# --------------------
def save_feedback(data):
    if data.get("rating", 0) < 4:
        return False  # Only save high-rated feedback

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(FEEDBACK_DIR, f"feedback_{timestamp}.json")
    try:
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"⚠️ Feedback save failed: {str(e)}")
        return False

def calculate_bleu(candidate, min_rating=4):
    references = []
    for file in os.listdir(FEEDBACK_DIR):
        with open(os.path.join(FEEDBACK_DIR, file), "r") as f:
            data = json.load(f)
            if data.get("rating", 0) >= min_rating:
                ref_tokens = word_tokenize(data["story"].lower())
                references.append(ref_tokens)
    if references:
        cand_tokens = word_tokenize(candidate.lower())
        smoothie = SmoothingFunction().method4
        return sentence_bleu(references, cand_tokens, smoothing_function=smoothie)
    return None

# --------------------
# Streamlit UI
# --------------------

st.title("📖 Adaptive AI Story Generator (MLX Optimized)")
# Information Sheet Summary
with st.expander(" ℹ️ Participant Information & Consent"):
    st.markdown("""
    **Purpose of the Study**:
    This app is part of a research project aimed at developing an adaptive AI storytelling system.

    **What Your Participation Involves**:
    - You will interact with the AI to generate stories.
    - You can rate the quality of the stories.
    - Your feedback will help improve the system.

    **Data Usage**:
    - Your input and ratings will be recorded.
    - Data will be used for research purposes only.

    **Confidentiality**:
    - All data is anonymised.
    - Participation is voluntary and you can withdraw at any time.

    By ticking the consent box, you agree to participate under these terms.
    """)
    st.session_state.consent = st.checkbox("I agree to participate in this study.")

prompt_type = st.radio("Choose Prompt Style", ["Structured", "Open-Ended"], horizontal=True)
length_choice = st.radio("Story Length", ["Short", "Medium", "Long"], horizontal=True)
length_map = {"Short": 250, "Medium": 350, "Long": 450}

genre = st.selectbox("Genre", ["Fantasy", "Mystery", "Sci-Fi", "General"])

if prompt_type == "Structured":
    col1, col2, col3 = st.columns(3)
    with col1:
        character = st.text_input("Character", "A young inventor")
    with col2:
        setting = st.text_input("Setting", "a steampunk city")
    with col3:
        conflict = st.text_input("Conflict", "discovers a dangerous conspiracy")
    prompt = f"Write a story in 3 paragraphs about: {character} lives in {setting}. One day, they {conflict}. As the character gets closer to the truth, they must choose between doing the right thing or protecting themselves. End the story with a twist."
else:
    raw_idea = st.text_area("Describe your story idea:", "A dragon who forgets how to breathe fire...")
    prompt = f"Write a detailed story in 3 paragraphs based on this idea: {raw_idea}. Include strong character development and an unexpected twist at the end."

if st.button("✨ Generate Story"):
    full_prompt = f"Genre: {genre}\n{prompt}"
    with st.spinner("Crafting your story..."):
        story = generate_story(full_prompt, max_length=length_map[length_choice])
    st.session_state.story = story
    st.text_area("Generated Story", value=story, height=300)

if "story" in st.session_state:
    # st.subheader("Rate This Story 💖")
    # rating = st.slider("Rate (1-5)", 1, 5, 3)
    # comments = st.text_input("Comments (optional)")
    st.subheader("Your Generated Story 📘")
    # st.text_area("Story", value=st.session_state.story, height=300, disabled=True)
    st.subheader("Generated Story")
    st.markdown(
        f"""
        <div style='background-color:#fdf6e3;padding:1rem;border-radius:8px;border:1px solid #ccc;font-family:monospace;white-space:pre-wrap'>
        {st.session_state.story}
        </div>
        """,
        unsafe_allow_html=True
    )


    st.subheader("Rate This Story 💖")
    rating = st.slider("Rate (1-5)", 1, 5, 3)
    comments = st.text_input("Comments (optional)")

    if st.button("Submit Feedback"):
        data = {
            "prompt": prompt,
            "story": st.session_state.story,
            "rating": rating,
            "genre": genre,
            "comments": comments,
            "timestamp": datetime.now().isoformat()
        }
        if save_feedback(data):
            st.success("Thanks! Your feedback was saved.")

st.divider()
with st.expander(" 📌 What is BLEU Score?"):
    st.markdown("""
    **BLEU score**:
   BLEU (Bilingual Evaluation Understudy) is an algorithm for evaluating the quality of text which has been machine-translated from one natural language to another.
   Quality is considered to be the correspondence between a machine’s output and that of a human.
   The closer a machine translation is to a professional human translation, the better it is”.
   BLEU was one of the first metrics to claim a high correlation with human judgements of quality, and remains one of the most popular automated and inexpensive metrics.
   Scores are calculated for individual translated segments—generally sentences—by comparing them with a set of good quality reference translations. 
    """)
if st.button("📊 Show BLEU Score") and "story" in st.session_state:
    score = calculate_bleu(st.session_state.story)
    if score:
        st.metric("BLEU Score", f"{score:.2f}")
    else:
        st.info("Not enough rated stories for comparison.")
