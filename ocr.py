# install req : !pip install openai pillow
import base64
from openai import OpenAI

# -------------------------
# CONFIG: set your API key and image
# -------------------------
API_KEY = "PASTE API KEY HERE "   # <- set this
IMAGE_PATH = "PASTE IMAGE PATH HERE "    # <- set your image file path

# -------------------------
# CLIENT
# -------------------------
client = OpenAI(base_url="https://api.mistral.ai/v1", api_key=API_KEY)

# -------------------------
# Helper: get validated user choice
# -------------------------
def get_user_choice():
    option_map = {
        "1": "medicine_names_only",
        "2": "medicine_with_time",
        "3": "usage_one_line",
        "4": "detailed_usage_composition",
        "5": "full_prescription_summary",
        "6": "patient_details_only"
    }

    print("Choose an option:")
    print("1. Medicine Names Only")
    print("2. Medicine + Timing")
    print("3. One-line Usage")
    print("4. Detailed Usage + Composition")
    print("5. Full Prescription Summary (everything)")
    print("6. Patient Details Only")

    while True:
        choice = input("\nEnter 1, 2, 3, 4, 5, or 6: ").strip()
        if choice in option_map:
            return option_map[choice]
        print("Invalid choice. Please enter 1, 2, 3, 4, 5, or 6.")


# -------------------------
# Build the strict prompt
# -------------------------
def build_prompt(user_option):
    prompt = f"""
You are a highly accurate medical prescription reader.

PROCESSING MODE (STRICT): "{user_option}"

GLOBAL RULES (APPLY ALWAYS):
• NEVER hallucinate details that are not visible.
• If handwriting is unclear, mark the item as "uncertain".
• ALWAYS read exactly what is visible in the prescription.
• NEVER add timing/dosage/purpose unless visible.
• NEVER ask questions or add notes.

MODE BEHAVIOR:

MODE: "medicine_names_only"
    • Output only the medicine names.

MODE: "medicine_with_time"
    • Medicine + timing only.
    • If timing not visible → "no timing mentioned".

MODE: "usage_one_line"
    • Medicine name + one-line purpose (general usage).
    • No timing, dosage, frequency.

MODE: "detailed_usage_composition"
    • Medicine name + purpose + composition + visible instructions.
    • Timing only if written in the prescription.

MODE: "full_prescription_summary"
    • Output EVERYTHING visible in the prescription:
        - patient details
        - doctor details
        - clinic/hospital name
        - medicines
        - timing/dosage
        - instructions and notes
        - signature/stamp
        - diagnosis (if mentioned)
    • Do not assume anything not visible.

MODE: "patient_details_only"
    • Extract only patient info:
        - Patient name
        - Age
        - Gender
        - Date
        - Patient ID / OPD No. / Contact / Address (if present)
    • Do NOT output medicine names or doctor info.

FINAL:
Process the attached image according to the selected mode "{user_option}".
"""
    return prompt


# -------------------------
# Load and encode image
# -------------------------
def load_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# -------------------------
# Send request to Mistral (pixtral)
# -------------------------
def call_mistral(prompt, image_b64, model_name="pixtral-large-latest"):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"},
                    {"type": "text", "text": "Please process the prescription image according to the selected mode."}
                ]
            }
        ],
        # you can set max_tokens, temperature, etc. if supported by your client
    )
    # Access content correctly
    return response.choices[0].message.content

# -------------------------
# Main flow
# -------------------------
def main():
    # 1) get user's numeric choice and canonical option string
    user_option = get_user_choice()
    print(f"\nYou selected: {user_option}\n")

    # 2) build prompt and load image
    prompt = build_prompt(user_option)
    try:
        image_b64 = load_image_base64(IMAGE_PATH)
    except FileNotFoundError:
        print(f"Image not found at {IMAGE_PATH}. Update IMAGE_PATH and retry.")
        return

    # 3) call Mistral and print result
    try:
        print("Processing image with Mistral... (this may take a few seconds)\n")
        output = call_mistral(prompt, image_b64)
        print("==== OUTPUT ====\n")
        print(output)
    except Exception as e:
        print("Error calling Mistral API:", str(e))

if __name__ == "__main__":
    main()
