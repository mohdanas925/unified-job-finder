import re


import re
from bs4 import BeautifulSoup

def clean_text(text):
    if not isinstance(text, str):
        return text
        
    text = text.lower()
    text = re.sub(r'[\n\r\t]+', ' ', text)
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r"""[!"$%()*,:;<=>?@\[\\\]^_`{|}~.]""", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    if not isinstance(text, str):
        return text

    return ' '.join(
        word for word in text.split()
        if word not in stop_words
    )


from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

model_name = "AchrafSoltani/jobbert-ner-sonnet-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

ner = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
    device=0 if torch.cuda.is_available() else -1
)

def extract_skills(text):
    if not isinstance(text, str) or not text.strip():
        return []

    tokens = tokenizer.encode(text, add_special_tokens=False)

    chunk_size = 450
    skills = []

    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = tokenizer.decode(
            chunk_tokens,
            skip_special_tokens=True
        )

        entities = ner(chunk_text)

        for ent in entities:
            if ent["entity_group"].upper() == "SKILL":
                skill = ent["word"].strip()

                # Remove standalone WordPiece fragments
                if skill.startswith("##"):
                    continue

                # Remove empty or invalid fragments
                if not skill or skill in {"s", "ing", "ation", "-"}:
                    continue

                if skill not in skills:
                    skills.append(skill)

    return skills



import ast
import re
import string

def clean_skills(skills_text):
    # Handle missing or invalid input
    if not isinstance(skills_text, str) or not skills_text.strip():
        return []

    # Convert string representation into a Python list
    try:
        skills = ast.literal_eval(skills_text)
    except (ValueError, SyntaxError):
        return []

    if not isinstance(skills, list):
        return []

    cleaned_skills = []

    for skill in skills:
        if not isinstance(skill, str):
            continue

        # Remove all whitespace
        skill = re.sub(r"\s+", "", skill)

        # Skip empty skills
        if not skill:
            continue

        # Remove skills containing only punctuation
        if all(char in string.punctuation for char in skill):
            continue

        # Remove unwanted prefix punctuation
        skill = re.sub(r"^[^\w]+", "", skill)

        # Remove unwanted suffix punctuation, preserving + and #
        skill = re.sub(r"[^\w+#]+$", "", skill)

        if not skill:
            continue

        # Remove standalone numbers
        if re.fullmatch(r"\d+", skill):
            continue

        # Remove single alphabet characters except valid C and R language variants
        if re.fullmatch(r"[a-zA-Z]", skill) and skill.lower() not in {"c", "r"}:
            continue

        cleaned_skills.append(skill)

    return cleaned_skills


import re

skill_mapping = {
    "excel": "msexcel",
    "microsoftexcel": "msexcel",
    "mswordexcel": "msexcel",
    "communication": "communicationskills",
    "communications": "communicationskills",
    "communicationskill": "communicationskills",
    "problem-solving": "problemsolving",
    "problem-solvingskills": "problemsolving",
    "git": "github",
    "pipelines": "pipeline",
    "ml": "machinelearning",
    "/ml": "machinelearning",
    "ai": "artificialintelligence",
    "&development": "development",
    "accesscontrols": "accesscontrol",
    "-sales": "sales",
    "sale": "sales",
    "mysql": "sql",
    "mssql": "sql",
    "js": "javascript",
    "java-script": "javascript",
    "postgres": "postgresql",
    "aws": "amazonwebservices",
    "gcp": "googlecloudplatform",
    "googlecloud": "googlecloudplatform",
    "cloudservices": "cloud",
    "-cloud": "cloud",
    "analysis": "analyticalskills",
    "leadership": "leadershipskills",
    "teamleadership": "leadershipskills"
}


def normalize_skills(skills):
    # Handle missing values or unexpected data types
    if not isinstance(skills, list):
        return []

    normalized_skills = []

    for skill in skills:
        if not isinstance(skill, str):
            continue

        skill = skill.lower().strip()

        # Apply longer mappings first
        for variation, standard in sorted(
            skill_mapping.items(),
            key=lambda item: len(item[0]),
            reverse=True
        ):
            pattern = r"(?<!\w)" + re.escape(variation) + r"(?!\w)"
            skill = re.sub(pattern, standard, skill)

        if skill:
            normalized_skills.append(skill)

    # Remove duplicates while preserving order
    return list(dict.fromkeys(normalized_skills))





# def save_object(file_path, obj):
#     """Save a Python object, such as a fitted vectorizer."""
#     try:
#         os.makedirs(os.path.dirname(file_path), exist_ok=True)

#         with open(file_path, "wb") as file:
#             pickle.dump(obj, file)

#         logging.info("Saved artifact: %s", file_path)

#     except Exception:
#         logging.exception("Could not save artifact.")
#         raise


# def load_object(file_path):
#     """Load a previously saved Python object."""
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(
#             f"Artifact not found: {file_path}"
#         )

#     with open(file_path, "rb") as file:
#         return pickle.load(file)