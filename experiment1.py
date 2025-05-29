import streamlit as st
import json
import re
import random 
import logging
import time
import os
from pathlib import Path
from datetime import datetime
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from mlx_lm import load, generate

nltk.download('punkt')

# Configuration
BASE_MODEL_PATH = "mlx_model"
ADAPTED_MODEL_PATH = "merged_model_final"
FEEDBACK_DIR = Path("feedback_data")
EXPERIMENT_RESULTS = "experiment_results.csv"
TEST_PROMPT_FILE = "test_prompts.json"

# Generation parameters
MAX_TOKENS = 512
TEMPERATURE = 0.7
MIN_STORY_LENGTH = 50
TARGET_WORD_COUNT = 250  

# Logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

FEEDBACK_DIR.mkdir(exist_ok=True)

@st.cache_resource
def load_model(path):
    try:
        model, tokenizer = load(path)
        logging.info(f"Model loaded from: {path}")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Model load failed: {str(e)}")
        st.error(f"🚨 Model load failed: {str(e)}")
        return None, None

def calculate_distinct_n(text, n=2):
    tokens = text.split()
    if len(tokens) < n:
        return 0
    ngrams = list(zip(*[tokens[i:] for i in range(n)]))
    return len(set(ngrams)) / len(ngrams)


def calculate_bleu(candidate, min_rating=4):
    references = []
    for file in os.listdir(FEEDBACK_DIR):
        if not file.endswith(".json"): continue
        with open(os.path.join(FEEDBACK_DIR, file), "r") as f:
            try:
                data = json.load(f)
                if data.get("rating", 0) >= min_rating:
                    ref_tokens = word_tokenize(data["story"].lower())
                    references.append(ref_tokens)
            except Exception as e:
                print(f"⚠️ Error reading {file}: {e}")
    if references:
        cand_tokens = word_tokenize(candidate.lower())
        smoothie = SmoothingFunction().method4
        return sentence_bleu(references, cand_tokens, smoothing_function=smoothie)
    return None
def compression_ratio(text):
    tokens = text.split()
    return len(tokens) / len(set(tokens)) if tokens else 0

def generate_story(prompt, model, tokenizer):
    try:
        full_prompt = f"<s>[INST]Write a creative story about {prompt} in approximately {TARGET_WORD_COUNT} words[/INST]"
        logging.info(f"Full prompt: {full_prompt}")
        start_time = time.time()
        response = generate(
            model,
            tokenizer,
            full_prompt,
            max_tokens=MAX_TOKENS,
            verbose=False
        )
        duration = time.time() - start_time
        if not response:
            logging.warning("No response from generation function")
            return None, 0, duration
        text = response.split("[/INST]")[-1].strip()
        cleaned = clean_story(text)
        return cleaned, len(set(cleaned.split())) if cleaned else 0, duration
    except Exception as e:
        logging.error(f"Generation failed: {str(e)}", exc_info=True)
        return None, 0, 0

def clean_story(text):
    try:
        text = re.sub(r'\s+', ' ', text).strip()
        if not text or len(text.split()) < MIN_STORY_LENGTH:
            logging.warning("Text too short or empty")
            return None
        if text[-1] not in ".!?":
            text += "."
        return text
    except Exception as e:
        logging.error(f"Cleaning failed: {str(e)}")
        return None

def main():
    if 'participant_id' not in st.session_state:
        st.session_state.participant_id = f"P{random.randint(1000, 9999)}"
    if 'rated_prompt_ids' not in st.session_state:
        st.session_state.rated_prompt_ids = set()
    st.set_page_config(
        page_title="AI Story Researcher",
        page_icon="📚",
        layout="wide"
    )

    st.title("📖 Adaptive Storytelling Experiment")
    st.info("The two stories are shown in random order. Please rate them independently and without bias.")

    try:
        with open(TEST_PROMPT_FILE) as f:
            prompts = json.load(f)
    except Exception as e:
        st.error(f"Failed to load prompts: {str(e)}")
        return

    unrated_prompts = [p for p in prompts if p.get("id") not in st.session_state.rated_prompt_ids]
    if not unrated_prompts:
        st.success("🎉 You've rated all available prompts!")
        return

    selected = st.selectbox("Choose a prompt:", [p["text"] for p in unrated_prompts], key="prompt_select")
    prompt_meta = next((p for p in unrated_prompts if p["text"] == selected), {})

    if st.button("🔄 Generate A/B Stories"):
        with st.spinner("Generating stories..."):
            st.session_state.participant_id = f"P{random.randint(1000, 9999)}"
            base_model, base_tok = load_model(BASE_MODEL_PATH)
            adapted_model, adapted_tok = load_model(ADAPTED_MODEL_PATH)

            if not base_model or not adapted_model:
                st.error("❌ Model loading failed.")
                return

            base_story, base_unique, base_time = generate_story(selected, base_model, base_tok)
            adapted_story, adapted_unique, adapted_time = generate_story(selected, adapted_model, adapted_tok)

            if base_story and adapted_story:
                base_len = len(base_story.split())
                adapted_len = len(adapted_story.split())
                if abs(base_len - adapted_len) > 30:
                    st.warning("🔁 Adjusting for equal story lengths.")

                stories = [
                    {"id": "A", "text": base_story, "type": "base", "unique_tokens": base_unique, "gen_time": base_time},
                    {"id": "B", "text": adapted_story, "type": "adapted", "unique_tokens": adapted_unique, "gen_time": adapted_time}
                ]
                random.shuffle(stories)
                st.session_state.stories = stories
                st.session_state.prompt_used = prompt_meta
                st.rerun()
            else:
                st.error("❌ Failed to generate both story versions.")
                st.error("❌ Failed to generate both story versions.")

    if 'stories' in st.session_state and st.session_state.stories:
        st.header("📋 Rate the Stories")
        for idx, s in enumerate(st.session_state.stories):
            with st.expander(f"Story {s['id']}", expanded=True):
                st.markdown(f"""
                <div style='background-color:#fefefe;padding:1rem;border-radius:6px;border:1px solid #ccc;'>
                <em>{s['text']}</em>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### Feedback")
                s["coherence_rating"] = st.slider("Logical Flow", 1, 5, 3, key=f"coh_{idx}")
                s["creativity_rating"] = st.slider("Originality", 1, 5, 3, key=f"cre_{idx}")
                s["engagement_rating"] = st.slider("Engagement", 1, 5, 3, key=f"eng_{idx}")
                s["feedback_text"] = st.text_area("Comments (optional)", key=f"feedback_{idx}")

        if st.button("📤 Submit Ratings"):
            for s in st.session_state.stories:
                distinct_1 = calculate_distinct_n(s["text"], n=1)
                distinct_2 = calculate_distinct_n(s["text"], n=2)
                comp_ratio = compression_ratio(s["text"])
                bleu = calculate_bleu(s["text"])
                record = {
                    "participant_id": st.session_state.participant_id,
                    "session_id": f"S{datetime.now().strftime('%H%M%S')}-{st.session_state.participant_id}",
                    "model_version": "Mistral-7B-RL" if s['type'] == 'adapted' else "Mistral-7B-Static",
                    "prompt_id": st.session_state.prompt_used.get("id", "NA"),
                    "prompt_text": st.session_state.prompt_used["text"],
                    "prompt_type": st.session_state.prompt_used.get("type", "unknown"),
                    "prompt_genre": st.session_state.prompt_used.get("genre", "unknown"),
                    "generated_text": s["text"],
                    "coherence_rating": s["coherence_rating"],
                    "creativity_rating": s["creativity_rating"],
                    "engagement_rating": s["engagement_rating"],
                    "distinct_1": distinct_1,
                    "distinct_2": distinct_2,
                    "compression_ratio": comp_ratio,
                    "perplexity": "NA",
                    "word_count": len(s["text"].split()),
                    "unique_tokens": s["unique_tokens"],
                    "timestamp": datetime.now().isoformat(),
                    "view_order": f"{s['id']},{s['type']}",
                    "generation_time": s["gen_time"],
                    "temperature": TEMPERATURE,
                    "blue" : bleu,
                    "top_p": 0.9,
                    "bias_flag": False,
                    "feedback_text": s["feedback_text"]
                }
                with open(EXPERIMENT_RESULTS, "a") as f:
                    f.write(json.dumps(record) + "\n")
            st.success("✅ All ratings saved!")
            st.session_state.stories = []
            st.session_state.rated_prompt_ids.add(st.session_state.prompt_used.get("id"))
            st.session_state.prompt_used = None

if __name__ == "__main__":
    main()
