"""Deterministic out-of-scope policy gate.

Refuses medical advice (diagnosis, symptoms, treatments, medicines, food-caused conditions),
calorie targets/deficits/allowances, and body-weight/BMI targets before any embedding,
retrieval, or language model call runs.
"""

import re
from typing import Literal

from src.schemas import AskResponse

GateOutcome = Literal["allow", "out_of_scope"]

OUT_OF_SCOPE_MESSAGE: str = (
    "I cannot provide medical advice, calorie targets, or body-weight targets. "
    "Please consult a qualified healthcare professional or registered dietitian."
)

# 1. Medical advice: diagnosis, cause of symptoms, treatment, medicine, or whether a condition was caused by food
MEDICAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(rash|hives|swelling|itching|itchy|nausea|vomiting|diarrhea|diarrhoea|"
        r"stomach\s*ache|abdominal\s*pain|cramps|fever|headache|migraine|dizziness|"
        r"fatigue|bloating|constipation|heartburn|acid\s*reflux|chest\s*pain|wheezing|"
        r"shortness\s*of\s*breath|anaphylaxis|infection|symptom|symptoms|flare[\s-]*up)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(diagnos(?:e|is|ing)|treat(?:ment|ing)?|cure|heal|remedy|prescri(?:be|ption)|"
        r"medicat(?:e|ion|ions)|medicine|medicines|drug|drugs|pill|pills|antibiotic|"
        r"antihistamine|insulin|dosage|dose|supplement\s+for|therapy)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(is\s+this\s+\w+\s+from\s+something\s+i\s+ate|"
        r"did\s+(?:what|something)\s+i\s+(?:ate|eat)\s+cause|"
        r"caused\s+by\s+(?:something\s+i\s+ate|food|eating)|"
        r"food\s+allergy|allergic\s+reaction|intolerance\s+test|"
        r"do\s+i\s+have\s+(?:food\s+poisoning|diabetes|celiac|coeliac|ibs|anemia|anaemia))\b",
        re.IGNORECASE,
    ),
)

# 2. Calorie targets, deficits, and daily energy allowances
CALORIE_TARGET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(calorie|calories|kcal|kilocalories|caloric|energy)\s+"
        r"(target|targets|goal|goals|deficit|deficits|surplus|allowance|allowances|"
        r"budget|limit|requirement|requirements|needs|intake\s+to\s+(?:lose|gain|maintain))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bhow\s+many\s+(?:calories|kcal|kilocalories)\s+"
        r"(?:should\s+i|do\s+i\s+need\s+to|to\s+(?:eat|consume|have)|can\s+i\s+(?:eat|have)|per\s+day\s+for\s+me)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:daily|my|personal|recommended)\s+(?:calorie|caloric|kcal)\s+"
        r"(?:intake|limit|allowance|target|goal|needs|budget)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:eat|consume|cut|burn)\s+\d+\s*(?:calories|kcal)\s+(?:a\s+day|per\s+day|daily|to\s+lose|to\s+gain)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:calories|kcal)\s+to\s+(?:lose|gain|maintain|drop)\b",
        re.IGNORECASE,
    ),
)

# 3. Body-weight targets, BMI goals, and questions about what someone should weigh
WEIGHT_TARGET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:what|how\s+much)\s+should\s+(?:i|a\s+person|someone|my\s+\w+)\s+weigh\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:ideal|target|goal|healthy|normal|right)\s+(?:body\s+)?(?:weight|bmi|body\s+mass\s+index)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:weight|bmi|body\s+mass\s+index)\s+(?:target|targets|goal|goals)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:lose|drop|shed|gain|put\s+on)\s+\d+(?:\.\d+)?\s*(?:kg|kgs|kilograms?|lbs?|pounds?|stones?|st)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bhow\s+to\s+(?:lose|gain)\s+weight\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:am\s+i|is\s+my\s+weight)\s+(?:overweight|underweight|obese)\b",
        re.IGNORECASE,
    ),
)

ALL_BLOCKED_PATTERNS: tuple[re.Pattern[str], ...] = (
    *MEDICAL_PATTERNS,
    *CALORIE_TARGET_PATTERNS,
    *WEIGHT_TARGET_PATTERNS,
)


def check(question: str) -> GateOutcome:
    """Classify a question as 'allow' or 'out_of_scope' using deterministic rules."""
    normalized = " ".join(question.strip().split())
    if not normalized:
        return "allow"

    for pattern in ALL_BLOCKED_PATTERNS:
        if pattern.search(normalized):
            return "out_of_scope"

    return "allow"


def make_out_of_scope_response() -> AskResponse:
    """Build the fixed out-of-scope AskResponse envelope with empty searched and sections."""
    return AskResponse(
        outcome="out_of_scope",
        searched=[],
        sections=[],
        message=OUT_OF_SCOPE_MESSAGE,
    )
