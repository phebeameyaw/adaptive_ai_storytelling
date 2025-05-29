# import streamlit as st
# st.set_page_config(page_title="Adaptive AI Story Generator", layout="wide")
# import json
# import os
# import csv 
# import re
# import pandas as pd
# import random 
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
# ADAPTED_WEIGHTS = "adapted_weights.safetensors"
# EXPERIMENT_RESULTS = "experiment_results.csv"
# TEST_PROMPT_FILE = "test_prompts.json"


# @st.cache_resource
# def load_mlx():
#     return load(MODEL_PATH)

# model, tokenizer = load_mlx()

# # --------------------
# # Story Generation (Enhanced Prompt)
# # --------------------
# # def generate_story(prompt, max_length=400):
# #     try:
# #         system_msg = "You are a creative writer. Generate coherent stories with characters, conflict, and a twist ending."
# #         formatted_prompt = f"<s>[INST] <<SYS>>{system_msg}<</SYS>>\n{prompt} [/INST]"
# #         generated = generate(
# #             model=model,
# #             tokenizer=tokenizer,
# #             prompt=formatted_prompt,
# #             max_tokens=max_length
# #         )
# #         return generated.replace(formatted_prompt, "").strip()
# #     except Exception as e:
# #         st.error(f"⚠️ Generation failed: {str(e)}")
# #         return None


# # Load test prompts from JSON file
# with open(TEST_PROMPT_FILE, "r") as f:
#     test_prompts = json.load(f)

# # Select random prompt
# selected_prompt = random.choice(test_prompts)


# # --------------------
# # Post-Processing Fix for Abrupt Endings + Multi-Pass
# # --------------------
# def post_process(text):
#     text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
#     text = re.sub(r'(\.|\?|!){2,}', r'\1', text)  # Fix multiple punctuation
#     sentences = re.split(r'(?<=[.!?]) +', text)

#     # Trim trailing incomplete or low-info sentences
#     if sentences:
#         while sentences and (len(sentences[-1].split()) < 5 or not sentences[-1][-1] in '.!?'):
#             sentences.pop()

#     return ' '.join(sentences).strip()

# # --------------------
# # Story Generation (Enhanced Prompt with Continuation)
# # --------------------
# def generate_story(prompt, max_length=400):
#     try:
#         system_msg = f"You are a creative writer. Generate coherent stories with characters, conflict, and a twist ending. Make sure the story is complete and ends with a strong, conclusive final sentence. It should fit within roughly {max_length} tokens."
#         formatted_prompt = f"<s>[INST] <<SYS>>{system_msg}<</SYS>>\n{prompt} [/INST]"
#         generated = generate(
#             model=model,
#             tokenizer=tokenizer,
#             prompt=formatted_prompt,
#             max_tokens=max_length
#         )
#         story = generated.replace(formatted_prompt, "").strip()
#         processed = post_process(story)

#         # If post-processed story is too short or ends weirdly, try continuation
#         if len(processed.split()) < max_length * 0.5:
#             continuation_prompt = processed + " Continue and end this story with a strong conclusion."
#             second_pass = generate(
#                 model=model,
#                 tokenizer=tokenizer,
#                 prompt=continuation_prompt,
#                 max_tokens=128
#             )
#             processed += " " + post_process(second_pass.replace(continuation_prompt, "").strip())

#         return processed

#     except Exception as e:
#         st.error(f"⚠️ Generation failed: {str(e)}")

# # --------------------
# # Feedback Logging
# # --------------------
# def save_feedback(data):
#     if data.get("rating", 0) < 4:
#         return False  # Only save high-rated feedback

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

# st.title("📖 Adaptive AI Story Generator (MLX Optimized)")
# # Information Sheet Summary
# with st.expander(" ℹ️ Participant Information & Consent"):
#     st.markdown("""
#     **Purpose of the Study**:
#     This app is part of a research project aimed at developing an adaptive AI storytelling system.

#     **What Your Participation Involves**:
#     - You will interact with the AI to generate stories.
#     - You can rate the quality of the stories.
#     - Your feedback will help improve the system.

#     **Data Usage**:
#     - Your input and ratings will be recorded.
#     - Data will be used for research purposes only.

#     **Confidentiality**:
#     - All data is anonymised.
#     - Participation is voluntary and you can withdraw at any time.

#     By ticking the consent box, you agree to participate under these terms.
#     """)
#     st.session_state.consent = st.checkbox("I agree to participate in this study.")

# prompt_type = st.radio("Choose Prompt Style", ["Structured", "Open-Ended"], horizontal=True)
# length_choice = st.radio("Story Length", ["Short", "Medium", "Long"], horizontal=True)
# length_map = {"Short": 250, "Medium": 350, "Long": 450}

# genre = st.selectbox("Genre", ["Fantasy", "Mystery", "Sci-Fi", "General"])

# if prompt_type == "Structured":
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         character = st.text_input("Character", "A young inventor")
#     with col2:
#         setting = st.text_input("Setting", "a steampunk city")
#     with col3:
#         conflict = st.text_input("Conflict", "discovers a dangerous conspiracy")
#     prompt = f"Write a story in 3 paragraphs about: {character} lives in {setting}. One day, they {conflict}. As the character gets closer to the truth, they must choose between doing the right thing or protecting themselves. End the story with a twist."
# else:
#     prompt_type = "Open-ended"
#     raw_idea = st.text_area("Describe your story idea:", "A dragon who forgets how to breathe fire...")
#     prompt = f"Write a detailed story in 3 paragraphs based on this idea: {raw_idea}. Include strong character development and an unexpected twist at the end."

# if st.button("✨ Generate Story"):
#     full_prompt = f"Genre: {genre}\n{prompt}"
#     with st.spinner("Crafting your story..."):
#         story = generate_story(full_prompt, max_length=length_map[length_choice])
#     st.session_state.story = story
#     st.text_area("Generated Story", value=story, height=300)

# if "story" in st.session_state:
#     # st.subheader("Rate This Story 💖")
#     # rating = st.slider("Rate (1-5)", 1, 5, 3)
#     # comments = st.text_input("Comments (optional)")
#     st.subheader("Your Generated Story 📘")
#     # st.text_area("Story", value=st.session_state.story, height=300, disabled=True)
#     st.subheader("Generated Story")
#     st.markdown(
#         f"""
#         <div style='background-color:#fdf6e3;padding:1rem;border-radius:8px;border:1px solid #ccc;font-family:monospace;white-space:pre-wrap'>
#         {st.session_state.story}
#         </div>
#         """,
#         unsafe_allow_html=True
#     )


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
            
#             # Define result_row INSIDE the success block
#             result_row = {
#                 "timestamp": data["timestamp"],
#                 "prompt": data["prompt"],
#                 "story": data["story"],
#                 "rating": data["rating"],
#                 "genre": data["genre"],
#                 "comments": data["comments"],
#                 "prompt_type": prompt_type,
#                 "model_version": "Mistral-7B",
#                 "coherence_rating": data["rating"],
#                 "creativity_rating": data["rating"],
#                 "engagement_rating": data["rating"],
#                 "overall_quality": data["rating"]  # Add this line
#             }

#             bleu = calculate_bleu(data["story"])
#             if bleu is not None:
#                 result_row["bleu_score"] = bleu

#             # CSV handling
#         if os.path.exists(EXPERIMENT_RESULTS):
#             df = pd.read_csv(EXPERIMENT_RESULTS, 
#                                 on_bad_lines='skip',
#                                 quoting=csv.QUOTE_ALL,
#                                 escapechar='\\')
#         else:
#             df = pd.DataFrame()

#         df = pd.concat([df, pd.DataFrame([result_row])], ignore_index=True)
#         df.to_csv(EXPERIMENT_RESULTS, 
#                                 index=False,
#                                 quoting=csv.QUOTE_ALL,
#                                 escapechar='\\')
#         st.success("📊 Logged to experiment_results.csv")

#     # if st.button("Submit Feedback"):

#     #     data = {
#     #         "prompt": prompt,
#     #         "story": st.session_state.story,
#     #         "rating": rating,
#     #         "genre": genre,
#     #         "comments": comments,
#     #         "timestamp": datetime.now().isoformat()
#     #     }
#     #     if save_feedback(data):
#     #         st.success("Thanks! Your feedback was saved.")
#     #         result_row = {
#     #         "timestamp": data["timestamp"],
#     #         "prompt": data["prompt"],
#     #         "story": data["story"],
#     #         "rating": data["rating"],
#     #         "genre": data["genre"],
#     #         "comments": data["comments"],
#     #         "prompt_type": prompt_type,
#     #         "model_version": "Mistral-7B",  
#     #         "coherence_rating": data["rating"],
#     #         "creativity_rating": data["rating"],
#     #         "engagement_rating": data["rating"]
#     #     }
            
#     #     result_row["overall_quality"] = data["rating"]  # same score for all 3

#     #     bleu = calculate_bleu(data["story"])
#     #     if bleu is not None:
#     #         result_row["bleu_score"] = bleu
#     #     if os.path.exists(EXPERIMENT_RESULTS):
#     #         df = pd.read_csv(EXPERIMENT_RESULTS, 
#     #                         on_bad_lines='skip',
#     #                         quoting=csv.QUOTE_ALL,
#     #                         escapechar='\\')
#     #     else:
#     #         df = pd.DataFrame()

#     #     df = pd.concat([df, pd.DataFrame([result_row])], ignore_index=True)
#     #     df.to_csv(EXPERIMENT_RESULTS, 
#     #             index=False,
#     #             quoting=csv.QUOTE_ALL,
#     #             escapechar='\\')

#     #     # if os.path.exists(EXPERIMENT_RESULTS):
#     #     #     df = pd.read_csv(EXPERIMENT_RESULTS)
#     #     #     df = pd.concat([df, pd.DataFrame([result_row])], ignore_index=True)
#     #     # else:
#     #     #     df = pd.DataFrame([result_row])
#     #     # df.to_csv(EXPERIMENT_RESULTS, index=False, quoting=csv.QUOTE_ALL) 
#     #     # quoting=csv.QUOTE_ALL

#     #     # if os.path.exists(EXPERIMENT_RESULTS):
#     #     #     with open(EXPERIMENT_RESULTS, "r") as f:
#     #     #         data = [json.loads(line) for line in f]
#     #     #     df = pd.DataFrame(data)
#     #     # else:
#     #     #     df = pd.DataFrame()

#     #     # # Add new entry
#     #     # df = pd.concat([df, pd.DataFrame([result_row])], ignore_index=True)

#     #     # # Save as JSON lines
#     #     # df.to_json(EXPERIMENT_RESULTS, orient='records', lines=True)
#     #     # st.success("📊 Logged to experiment_results.csv")

# st.divider()
# with st.expander(" 📌 What is BLEU Score?"):
#     st.markdown("""
#     **BLEU score**:
#    BLEU (Bilingual Evaluation Understudy) is an algorithm for evaluating the quality of text which has been machine-translated from one natural language to another.
#    Quality is considered to be the correspondence between a machine’s output and that of a human.
#    The closer a machine translation is to a professional human translation, the better it is”.
#    BLEU was one of the first metrics to claim a high correlation with human judgements of quality, and remains one of the most popular automated and inexpensive metrics.
#    Scores are calculated for individual translated segments—generally sentences—by comparing them with a set of good quality reference translations. 
#     """)
# if st.button("📊 Show BLEU Score") and "story" in st.session_state:
#     score = calculate_bleu(st.session_state.story)
#     if score:
#         st.metric("BLEU Score", f"{score:.2f}")
#     else:
#         st.info("Not enough rated stories for comparison.")


import streamlit as st
st.set_page_config(page_title="Adaptive AI Story Generator", layout="wide")
import json
import os
import csv 
import re
import pandas as pd
import random 
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
MODEL_PATH = "mlx_model"
FEEDBACK_DIR = "feedback_data"
os.makedirs(FEEDBACK_DIR, exist_ok=True)
ADAPTED_WEIGHTS = "adapted_weights.safetensors"
EXPERIMENT_RESULTS = "experiment_results.csv"
TEST_PROMPT_FILE = "test_prompts.json"

@st.cache_resource
def load_mlx():
    return load(MODEL_PATH)

model, tokenizer = load_mlx()

# --------------------
# Story Generation Functions
# --------------------
def post_process(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(\.|\?|!){2,}', r'\1', text)
    sentences = re.split(r'(?<=[.!?]) +', text)
    if sentences:
        while sentences and (len(sentences[-1].split()) < 5 or not sentences[-1][-1] in '.!?'):
            sentences.pop()
    return ' '.join(sentences).strip()

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
# Feedback Handling
# --------------------
def save_feedback(data):
    if data.get("rating", 0) < 4:
        return False
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

with st.expander(" ℹ️ Participant Information & Consent"):
    st.markdown("""[Your consent form text]""")
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

if "story" in st.session_state:
    st.subheader("Generated Story")
    st.markdown(
        f"""<div style='background-color:#fdf6e3;padding:1rem;border-radius:8px;border:1px solid #ccc;font-family:monospace;white-space:pre-wrap'>
        {st.session_state.story}</div>""",
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
            
            # Create result_row AFTER successful feedback save
            result_row = {
                "timestamp": data["timestamp"],
                "prompt": data["prompt"],
                "story": data["story"],
                "rating": data["rating"],
                "genre": data["genre"],
                "comments": data["comments"],
                "prompt_type": prompt_type,
                "model_version": "Mistral-7B",
                "coherence_rating": data["rating"],
                "creativity_rating": data["rating"],
                "engagement_rating": data["rating"],
                "overall_quality": data["rating"]
            }

            bleu = calculate_bleu(data["story"])
            if bleu is not None:
                result_row["bleu_score"] = bleu

            # Handle CSV logging
            try:
                if os.path.exists(EXPERIMENT_RESULTS):
                    df = pd.read_csv(EXPERIMENT_RESULTS, 
                                   on_bad_lines='skip',
                                   quoting=csv.QUOTE_ALL,
                                   escapechar='\\')
                else:
                    df = pd.DataFrame()

                df = pd.concat([df, pd.DataFrame([result_row])], ignore_index=True)
                df.to_csv(EXPERIMENT_RESULTS, 
                         index=False,
                         quoting=csv.QUOTE_ALL,
                         escapechar='\\')
                st.success("📊 Logged to experiment_results.csv")
            except Exception as e:
                st.error(f"Failed to log experiment: {str(e)}")

st.divider()
with st.expander(" 📌 What is BLEU Score?"):
    st.markdown("""[Your BLEU explanation text]""")

if st.button("📊 Show BLEU Score") and "story" in st.session_state:
    score = calculate_bleu(st.session_state.story)
    if score:
        st.metric("BLEU Score", f"{score:.2f}")
    else:
        st.info("Not enough rated stories for comparison.")