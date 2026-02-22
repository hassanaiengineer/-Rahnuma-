import re

# Simple heuristic-based safety checks for Urdu
FORBIDDEN_KEYWORDS = [
    "مفتی", "فتویٰ", "حدیث", "قرآن", "آیت", # Religious authority/rulings
    "ڈاکٹر", "دوائی", "علاج", "بیماری", "نسخہ", # Medical
    "وکیل", "قانون", "عدالت", "جرم", # Legal
    "کوڈ", "پروگرام", "ہیک", "پائیتھون", "جاوا" # Blocking deep technical help as requested
]

def check_safety_heuristics(text):
    """
    Checks if the text contains keywords related to blocked domains.
    """
    for word in FORBIDDEN_KEYWORDS:
        if word in text:
            return False
    return True
