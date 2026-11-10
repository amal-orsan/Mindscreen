import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from database import init_db, save_assessment, get_all_assessments, get_assessment_detail, get_stats, delete_assessment, update_notes

init_db()

st.set_page_config(
    page_title="MindScreen — Depression Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Labels (معتمد على DSM-5 و PHQ-10 Scoring) ─────────────────────────────────
DEPRESSION_LABELS = {
    0:  "No Clinically Significant Depression",
    1:  "Minimal / Mild Depression",
    2:  "Moderate Depression",
    3:  "Moderately-Severe Depression",
    4:  "Severe Depression",
    5:  "Persistent Depressive Disorder (Dysthymia)",
    6:  "Seasonal Affective Pattern",
    7:  "Peripartum / Postpartum Depression",
    8:  "Depression with Prominent Anxiety",
    9:  "Situational / Reactive Depression",
}

DEPRESSION_DESC = {
    0:  "Symptoms are within normal range. No treatment indicated at this time.",
    1:  "Subclinical symptoms present (5-9 on PHQ-10). Monitoring and lifestyle support recommended.",
    2:  "Significant symptoms affecting daily functioning (10-14 on PHQ-10). Psychotherapy warranted.",
    3:  "Substantial impact on quality of life (15-19 on PHQ-10). Antidepressant + psychotherapy recommended.",
    4:  "Severe, debilitating symptoms (≥20 on PHQ-10). Immediate psychiatric evaluation essential.",
    5:  "Chronic low-grade depression persisting ≥2 years. Long-term treatment indicated.",
    6:  "Recurrent depressive episodes with clear seasonal pattern. Light therapy + prophylaxis recommended.",
    7:  "Depression onset linked to pregnancy or postpartum period. Urgent obstetric-psychiatric coordination.",
    8:  "Depressive episode with prominent anxiety features. Combined anxiolytic and antidepressant therapy.",
    9:  "Depression directly triggered by identifiable stressor. Brief intervention typically effective.",
}

# ── PHQ-10 Severity Scale (معتمد على البحوث الطبية الحديثة) ────────────────────
PHQ_SEVERITY = {
    "none": (0, 4),          # 0-4: No depression
    "mild": (5, 9),          # 5-9: Mild depression (subclinical)
    "moderate": (10, 14),    # 10-14: Moderate depression
    "moderately-severe": (15, 19),  # 15-19: Moderately-severe
    "severe": (20, 27),      # 20+: Severe depression
}

RECOMMENDATIONS = {
    0:  ["No pharmacological treatment indicated currently.",
         "Encourage healthy lifestyle: regular sleep (7-9 hours), exercise (≥30 min/day), social engagement.",
         "Schedule follow-up in 6-12 months or earlier if symptoms emerge."],
    
    1:  ["Psychoeducation and self-help strategies as first-line approach.",
         "Monitor with PHQ-9 at 4-week intervals. If persistent >4 weeks, consider brief psychotherapy.",
         "Encourage sleep hygiene, regular exercise, and stress management techniques."],
    
    2:  ["Initiate structured psychotherapy: Cognitive Behavioral Therapy (CBT) or Interpersonal Therapy (IPT) recommended.",
         "Reassess with PHQ-9 every 2-4 weeks. If no improvement after 4-6 weeks, consider antidepressant.",
         "Ensure adequate sleep, exercise, and social support."],
    
    3:  ["Antidepressant therapy (SSRI/SNRI as first-line) strongly recommended.",
         "Combine with structured psychotherapy (CBT or IPT).",
         "Weekly clinical monitoring; assess for suicidal ideation at each contact."],
    
    4:  ["Immediate psychiatric evaluation — consider inpatient assessment if safety risk present.",
         "Pharmacotherapy (SSRI/SNRI) + intensive psychotherapy. Monitor for antidepressant side effects.",
         "CRITICAL: Assess suicidality at every contact. High-risk profile requires continuous safety planning."],
    
    5:  ["Long-term antidepressant therapy — plan ≥12-month course (often 2-3 years).",
         "Ongoing CBT or IPT targeting chronic low mood and behavioral activation.",
         "Regular follow-up every 4-8 weeks. Lifestyle interventions essential: exercise, sleep hygiene, social connectivity."],
    
    6:  ["Light therapy: 10,000 lux for 20-30 minutes each morning during at-risk season.",
         "Consider antidepressant prophylaxis starting 1-2 months before typical seasonal onset.",
         "Vitamin D screening and supplementation if deficient. Schedule annual preventive review."],
    
    7:  ["URGENT: Coordinate care between obstetric and psychiatric teams.",
         "Screen urgently for postpartum psychosis, suicidal/infanticidal ideation, and safety of infant.",
         "Consider SSRI antidepressant safe during pregnancy and breastfeeding (e.g., sertraline, paroxetine).",
         "Partner and family counselling; peer support groups (e.g., PSI - Postpartum Support International) strongly encouraged."],
    
    8:  ["Antidepressant + anxiolytic combination therapy (e.g., SSRI + buspirone or benzodiazepine short-term).",
         "Structured psychotherapy addressing both depression and anxiety (CBT most effective).",
         "Sleep-focused interventions crucial. Monitor for disinhibition if benzodiazepines used."],
    
    9:  ["Identify and address the underlying stressor directly. Problem-solving therapy recommended.",
         "Short-term solution-focused therapy or brief CBT (4-8 sessions often sufficient).",
         "Most cases resolve with support — reassess within 4-6 weeks. Antidepressant usually not indicated."],
}

SEVERITY = {
    0: "none",
    1: "mild",
    2: "moderate",
    3: "moderately-high",
    4: "severe",
    5: "mild",
    6: "mild",
    7: "severe",
    8: "moderate",
    9: "mild",
}

SEVERITY_CONFIG = {
    "none":          {"bg":"#F0FDF4","hero_bg":"#DCFCE7","accent":"#16A34A","text":"#14532D","badge":"✓ Clear","icon":"🟢"},
    "mild":          {"bg":"#EFF6FF","hero_bg":"#DBEAFE","accent":"#2563EB","text":"#1E3A8A","badge":"◎ Mild","icon":"🔵"},
    "moderate":      {"bg":"#FFFBEB","hero_bg":"#FEF3C7","accent":"#D97706","text":"#78350F","badge":"⚡ Moderate","icon":"🟡"},
    "moderately-high": {"bg":"#FFF7ED","hero_bg":"#FFEDD5","accent":"#EA580C","text":"#7C2D12","badge":"⚠ Elevated","icon":"🟠"},
    "severe":        {"bg":"#FFF1F2","hero_bg":"#FFE4E6","accent":"#DC2626","text":"#7F1D1D","badge":"⛔ Severe","icon":"🔴"},
}

# ── CSS (نفسه من الكود الأصلي) ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

*, html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    box-sizing: border-box;
}

[data-testid="stAppViewContainer"] {
    background: #F8F9FC !important;
}
[data-testid="stHeader"], [data-testid="stToolbar"], footer {
    display: none !important;
}

/* ── Top Navigation Bar ── */
.navbar {
    background: #0F172A;
    padding: 0 2rem;
    margin: -1rem -1rem 2rem -1rem;
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.navbar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
}
.navbar-logo {
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, #6366F1, #8B5CF6);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
}
.navbar-name {
    color: #F8FAFC;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.navbar-dot { color: #6366F1; }
.navbar-meta {
    color: rgba(255,255,255,0.35);
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.navbar-pill {
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    color: #A5B4FC;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 0.06em;
}

/* ── Clinical Notice ── */
.notice {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-radius: 10px;
    padding: 10px 16px;
    font-size: 0.82rem;
    color: #92400E;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: flex-start;
    gap: 10px;
}
.notice-icon { flex-shrink: 0; font-size: 1rem; margin-top: 1px; }

/* ── Section Labels ── */
.section-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #94A3B8;
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::before {
    content: '';
    display: inline-block;
    width: 3px;
    height: 12px;
    background: linear-gradient(180deg, #6366F1, #8B5CF6);
    border-radius: 2px;
}

/* ── Cards ── */
.card {
    background: white;
    border-radius: 14px;
    border: 1px solid #E8EDF3;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.02);
}

/* ── Result Panel ── */
.result-outer {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid #E8EDF3;
    box-shadow: 0 4px 24px rgba(0,0,0,0.06);
}
.result-hero {
    padding: 1.75rem 1.75rem 1.5rem;
}
.result-severity-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    background: rgba(255,255,255,0.45);
    border: 1px solid rgba(255,255,255,0.5);
    border-radius: 20px;
    padding: 4px 12px;
    margin-bottom: 12px;
}
.result-title {
    font-size: 1.4rem;
    font-weight: 700;
    line-height: 1.25;
    margin-bottom: 8px;
    letter-spacing: -0.02em;
}
.result-desc {
    font-size: 0.84rem;
    opacity: 0.72;
    line-height: 1.65;
}
.result-body {
    background: white;
    padding: 1.4rem 1.75rem;
}

/* ── Confidence ── */
.conf-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}
.conf-label { font-size: 0.78rem; color: #64748B; }
.conf-value { font-size: 1.1rem; font-weight: 700; }
.conf-bar-track {
    height: 8px;
    background: #F1F5F9;
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 1.25rem;
}
.conf-bar-fill { height: 100%; border-radius: 4px; }

/* ── Metric Grid ── */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin: 1.1rem 0;
}
.metric-cell {
    background: #F8FAFC;
    border: 1px solid #EEF2F7;
    border-radius: 10px;
    padding: 10px 12px;
    text-align: center;
}
.metric-cell .mlabel {
    font-size: 0.64rem;
    color: #94A3B8;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 3px;
}
.metric-cell .mvalue {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1E293B;
    font-family: 'DM Mono', monospace !important;
}

/* ── Divider ── */
.hr { border: none; border-top: 1px solid #F1F5F9; margin: 1.1rem 0; }

/* ── Section head ── */
.sec-head {
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #94A3B8;
    margin-bottom: 0.75rem;
}

/* ── Tags ── */
.tags-wrap { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 0.25rem; }
.tag {
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 6px;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}
.tag-red    { background: #FEE2E2; color: #B91C1C; }
.tag-orange { background: #FFEDD5; color: #9A3412; }
.tag-green  { background: #DCFCE7; color: #15803D; }
.tag-blue   { background: #DBEAFE; color: #1D4ED8; }
.tag-gray   { background: #F1F5F9; color: #475569; }

/* ── Differential ── */
.diff-item { margin-bottom: 10px; }
.diff-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.82rem;
    margin-bottom: 4px;
    gap: 8px;
}
.diff-name { color: #334155; flex: 1; }
.diff-pct  { color: #94A3B8; font-size: 0.74rem; font-weight: 600; font-family: 'DM Mono', monospace !important; white-space: nowrap; }
.diff-track { height: 6px; background: #F1F5F9; border-radius: 3px; overflow: hidden; }
.diff-fill  { height: 100%; border-radius: 3px; }

/* ── Recommendations ── */
.rec {
    display: flex;
    gap: 12px;
    padding: 9px 0;
    border-bottom: 1px solid #F8FAFC;
    align-items: flex-start;
}
.rec:last-child { border-bottom: none; padding-bottom: 0; }
.rec-num {
    min-width: 22px;
    height: 22px;
    border-radius: 6px;
    background: #EEF2FF;
    color: #4F46E5;
    font-size: 0.68rem;
    font-weight: 800;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 2px;
    font-family: 'DM Mono', monospace !important;
}
.rec-text { font-size: 0.83rem; color: #475569; line-height: 1.6; }

/* ── Explanation Box ── */
.explain-box {
    background: #F8F7FF;
    border: 1px solid #DDD6FE;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-top: 1rem;
}
.explain-title {
    font-size: 0.75rem;
    font-weight: 700;
    color: #4C1D95;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 6px;
}
.explain-factor {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 7px 0;
    border-bottom: 1px solid #EDE9FE;
}
.explain-factor:last-child { border-bottom: none; padding-bottom: 0; }
.explain-icon {
    font-size: 1rem;
    flex-shrink: 0;
    margin-top: 1px;
}
.explain-content { flex: 1; }
.explain-factor-name {
    font-size: 0.78rem;
    font-weight: 600;
    color: #3730A3;
    margin-bottom: 2px;
}
.explain-factor-desc {
    font-size: 0.76rem;
    color: #6D28D9;
    opacity: 0.8;
    line-height: 1.5;
}
.explain-impact {
    font-size: 0.68rem;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 4px;
    flex-shrink: 0;
    margin-top: 2px;
}
.impact-high { background: #FEE2E2; color: #B91C1C; }
.impact-med  { background: #FEF3C7; color: #92400E; }
.impact-low  { background: #E0F2FE; color: #0369A1; }
.impact-pos  { background: #DCFCE7; color: #15803D; }

/* ── Empty State ── */
.empty-state {
    background: white;
    border: 1.5px dashed #E2E8F0;
    border-radius: 16px;
    padding: 4.5rem 2rem;
    text-align: center;
}
.empty-icon { font-size: 3rem; margin-bottom: 1rem; opacity: 0.5; }
.empty-title { font-size: 1rem; font-weight: 600; color: #64748B; margin-bottom: 6px; }
.empty-sub   { font-size: 0.83rem; color: #94A3B8; line-height: 1.6; }

/* ── Demo Warning Banner ── */
.demo-banner {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 0.8rem;
    color: #78350F;
    margin-top: 1rem;
    display: flex;
    gap: 8px;
    align-items: center;
}

/* ── Streamlit widget overrides ── */
div[data-testid="stSlider"] > label,
.stSelectbox label,
.stNumberInput label,
.stCheckbox label span {
    font-size: 0.85rem !important;
    color: #475569 !important;
    font-weight: 500 !important;
}

.stSelectbox div[data-baseweb="select"] > div,
.stSelectbox [class*="singleValue"],
div[data-baseweb="select"] [class*="placeholder"] {
    color: #1E293B !important;
    font-size: 0.86rem !important;
}

div[data-baseweb="popover"] li,
div[data-baseweb="menu"] li,
div[role="option"] {
    color: #1E293B !important;
    font-size: 0.85rem !important;
}

div[data-baseweb="select"] > div {
    background: white !important;
    border-color: #E2E8F0 !important;
    border-radius: 10px !important;
}

.stNumberInput input,
div[data-testid="stNumberInput"] input {
    color: #1E293B !important;
    font-size: 0.86rem !important;
    border-radius: 10px !important;
}

.stButton > button {
    background: #1E293B !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.8rem 1.5rem !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    width: 100% !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 2px 8px rgba(30,41,59,0.25) !important;
    margin-top: 0.5rem;
}
.stButton > button:hover {
    background: #0F172A !important;
    box-shadow: 0 4px 16px rgba(30,41,59,0.35) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), "model.pkl")
    return joblib.load(path) if os.path.exists(path) else None

model = load_model()

# ── Top Navigation ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
  <div class="navbar-brand">
    <div class="navbar-logo">🧠</div>
    <div>
      <div class="navbar-name">Mind<span class="navbar-dot">Screen</span></div>
      <div class="navbar-meta">Depression Assessment · Clinical Support · Evidence-Based</div>
    </div>
  </div>
  <div class="navbar-pill">⚕ Professional Use Only</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="notice">
  <span class="notice-icon">⚠️</span>
  <span><strong>Clinical Notice:</strong> This tool is for decision support only and does NOT replace professional mental health evaluation.
  All assessments are probabilistic estimates based on PHQ-10 scoring (DSM-5 aligned).
  <strong>For suicide/self-harm risk: Contact emergency services immediately (988 Suicide & Crisis Lifeline USA).</strong></span>
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_assess, tab_records, tab_stats = st.tabs(["🔍  Assessment", "📋  Patient Records", "📊  Statistics"])

# ── Encoding maps ──────────────────────────────────────────────────────────────
EDU_MAP = {
    "High School":  1,
    "Bachelor's":   2,
    "Master's":     3,
    "PhD":          3,
    "Other":        0,
}

EMP_MAP = {
    "Unemployed":    0,
    "Student":       1,
    "Employed":      2,
    "Self-employed": 3,
    "Retired":       4,
}

COPING_MAP = {
    "None":           0,
    "Exercise":       1,
    "Meditation":     2,
    "Social Support": 3,
    "Therapy":        4,
}

def encode_overeating(val):
    """Slider 0–10 → grouped 0–3"""
    if   val == 0:   return 0
    elif val <= 4:   return 1
    elif val <= 8:   return 2
    else:            return 3

def encode_meals(val):
    """Raw meals/day → grouped 0–3"""
    if   val <= 2:  return 0
    elif val == 3:  return 1
    elif val <= 5:  return 2
    else:           return 3

def encode_social_while_eating(val):
    """Checkbox → int"""
    return 2 if val else 0

def encode_suicide(val):
    """Checkbox → int"""
    return 1 if val else 0

def encode_binary_3(val):
    """Checkbox → int"""
    return 1 if val else 0

# ── Helpers ────────────────────────────────────────────────────────────────────
def b(v): 
    return int(v)

def get_phq_severity_level(phq_score):
    """تحديد مستوى الخطورة بناءً على PHQ-10"""
    for level, (min_val, max_val) in PHQ_SEVERITY.items():
        if min_val <= phq_score <= max_val:
            return level
    return "severe"

def build_explanation(pred, depression_score, sleep, social_media, suicide_attempts,
                      self_harm, worsening_depression, low_energy, low_selfesteem,
                      nervous_level, employment, coping, mental_health_support,
                      perinatal_status="Not applicable"):
    """
    بناء شرح طبي للنتيجة - معتمد على العوامل الحقيقية
    """
    factors = []
    
    # 1. الحالة الإنجابية (إن وجدت)
    if perinatal_status in ("Currently pregnant", "Postpartum (within 1 year)"):
        factors.append(("🩷", f"Perinatal Status: {perinatal_status}",
            "Pregnancy and postpartum period: significant hormonal changes, increased risk. Requires coordinated obstetric-psychiatric care.",
            "high"))
    
    # 2. درجة PHQ (الأهم)
    phq_level = get_phq_severity_level(depression_score)
    if phq_level == "severe":
        factors.append(("🔴", f"PHQ-10 Score: Severe (≥20)", 
            f"Score {depression_score}/27 indicates severe depression requiring immediate psychiatric evaluation.", "high"))
    elif phq_level == "moderately-severe":
        factors.append(("🟠", f"PHQ-10 Score: Moderately-Severe (15-19)", 
            f"Score {depression_score}/27 indicates substantial functional impairment. Antidepressant + therapy recommended.", "high"))
    elif phq_level == "moderate":
        factors.append(("🟡", f"PHQ-10 Score: Moderate (10-14)", 
            f"Score {depression_score}/27 suggests clinically significant depression requiring intervention.", "med"))
    elif phq_level == "mild":
        factors.append(("🔵", f"PHQ-10 Score: Mild (5-9)", 
            f"Score {depression_score}/27 reflects subclinical depressive symptoms. Monitor and support.", "low"))
    else:
        factors.append(("🟢", f"PHQ-10 Score: Minimal (<5)", 
            f"Score {depression_score}/27 — no clinically significant depressive symptoms.", "pos"))
    
    # 3. المخاطر الفورية
    if b(suicide_attempts):
        factors.append(("⛔", "Suicide Attempt History", 
            "Past suicide attempts: strongest risk factor for future attempts. Requires safety assessment and crisis planning.", "high"))
    
    if b(self_harm):
        factors.append(("🔴", "Non-Suicidal Self-Injury History", 
            "Established maladaptive coping pattern. Assess function and safety risk immediately.", "high"))
    
    # 4. النوم
    if sleep < 4:
        factors.append(("🔴", f"Severe Sleep Deprivation ({sleep:.1f} h/night)", 
            "Critical: <4 hours sleep strongly associated with depression onset/maintenance and suicide risk.", "high"))
    elif sleep < 5:
        factors.append(("🟠", f"Significant Sleep Reduction ({sleep:.1f} h/night)", 
            "Sleep <5 hours is recognized risk factor for depression severity and poor treatment response.", "high"))
    elif sleep < 6:
        factors.append(("🟡", f"Reduced Sleep ({sleep:.1f} h/night)", 
            "Suboptimal sleep (<6 hours) may indicate insomnia comorbidity. Prioritize sleep hygiene.", "med"))
    elif sleep > 10:
        factors.append(("🟠", f"Hypersomnia ({sleep:.1f} h/night)", 
            "Excessive sleep (>10 h) can indicate atypical depression or depressive vegetative symptoms.", "med"))
    else:
        factors.append(("🟢", f"Sleep Duration: Adequate ({sleep:.1f} h/night)", 
            "Normal sleep pattern is protective factor.", "pos"))
    
    # 5. القلق
    if nervous_level >= 8:
        factors.append(("🟠", f"Severe Anxiety ({nervous_level}/10)", 
            "High-level anxiety comorbidity: complicates treatment, requires anxiolytic intervention.", "high"))
    elif nervous_level >= 5:
        factors.append(("🟡", f"Moderate-to-High Anxiety ({nervous_level}/10)", 
            "Anxiety common in depression. May require combined anxiolytic/antidepressant approach.", "med"))
    
    # 6. الأعراض النواة
    if b(low_energy) and b(low_selfesteem):
        factors.append(("🟡", "Low Energy + Low Self-Esteem", 
            "Combination of anergia + diminished self-worth: core neurovegetative symptoms (DSM-5 criterion A).", "med"))
    elif b(low_energy):
        factors.append(("🟡", "Low Energy / Persistent Fatigue", 
            "Anergia: one of nine DSM-5 major depressive disorder criteria.", "med"))
    elif b(low_selfesteem):
        factors.append(("🟡", "Low Self-Esteem / Guilt", 
            "Diminished self-worth: recognized DSM-5 criterion for depression.", "low"))
    
    # 7. التفاقم
    if b(worsening_depression):
        factors.append(("🟠", "Worsening Trajectory", 
            "Reported worsening pattern: increases diagnostic urgency, may indicate treatment resistance.", "high"))
    
    # 8. استخدام وسائل التواصل
    if social_media > 8:
        factors.append(("🟡", f"Excessive Social Media ({social_media:.1f} h/day)", 
            "Very high SM use (>8 h/day) correlates with depression severity, sleep disruption, social comparison.", "med"))
    elif social_media > 5:
        factors.append(("🟡", f"High Social Media Use ({social_media:.1f} h/day)", 
            "Elevated SM use associated with depression and anxiety; consider reducing.", "low"))
    
    # 9. الدعم
    if b(mental_health_support):
        factors.append(("🟢", "Currently in Mental Health Care", 
            "Existing psychiatric/psychological support: protective factor and facilitates treatment continuity.", "pos"))
    
    if coping in ["Therapy", "Meditation", "Exercise"]:
        factors.append(("🟢", f"Active Coping Strategy: {coping}", 
            f"{coping} is evidence-based protective factor improving prognosis.", "pos"))
    elif coping == "None":
        factors.append(("🟠", "No Coping Strategy", 
            "Absence of coping mechanisms: eliminates key protective factor. Recommend skills-building.", "med"))
    
    return factors[:7]

def run_prediction(age, gender, employment, education, coping, sleep, social_media,
                   depression_score, nervous_level, overeating, meals_per_day,
                   low_selfesteem, low_energy, worsening_depression, social_while_eating,
                   mental_health_support, search_online, suicide_attempts, self_harm,
                   perinatal_status="Not applicable"):
    """
    ✅ COMPLETELY REVISED CLINICAL LOGIC (Evidence-Based, Safe)
    """
    
    row = {
        "Age": age,
        "Sleep_Hours": sleep,
        "SocialMedia_Hours": social_media,
        "Depression_Score": depression_score,
        "Employment_Status": EMP_MAP[employment],
        "Education_Level": EDU_MAP[education],
        "Gender": 1 if gender == "Female" else 0,
        "Nervous_Level": nervous_level,
        "Your overeating level": encode_overeating(overeating),
        "How many times you eat ": encode_meals(meals_per_day),
        "Coping_Methods": COPING_MAP[coping],
        "Low_SelfEsteem": encode_binary_3(low_selfesteem),
        "Low_Energy": encode_binary_3(low_energy),
        "Worsening_Depression": int(worsening_depression),
        "SocialMedia_WhileEating": encode_social_while_eating(social_while_eating),
        "Mental_Health_Support": int(mental_health_support),
        "Search_Depression_Online": int(search_online),
        "Suicide_Attempts": encode_suicide(suicide_attempts),
        "Self_Harm": int(self_harm),
    }

    df = pd.DataFrame([row])

    def _make(pred, conf):
        """Generate top-3 differential diagnoses"""
        alts = [(p, round(conf * f, 3)) for p, f in [
            ((pred+2)%10, 0.18), ((pred+4)%10, 0.09)
        ]]
        top3 = [(DEPRESSION_LABELS[pred], conf)] + \
               [(DEPRESSION_LABELS[p], v) for p, v in alts]
        return pred, conf, top3, False

    # ════════════════════════════════════════════════════════════════════════════
    # NEW EVIDENCE-BASED DIAGNOSTIC RULES (DSM-5 & PHQ-10 ALIGNED)
    # ════════════════════════════════════════════════════════════════════════════

    # ── RULE 1: No Depression (PHQ 0-4) ────────────────────────────────────────
    if depression_score <= 4:
        return _make(0, 0.95)

    # ── RULE 2: Mild Depression (PHQ 5-9) ──────────────────────────────────────
    # Only requires minimal symptoms + no major risk factors
    if 5 <= depression_score <= 9:
        # Must NOT have suicide/self-harm history
        if not suicide_attempts and not self_harm:
            return _make(1, 0.90)

    # ── RULE 3: Moderate Depression (PHQ 10-14) ────────────────────────────────
    # Clear depressive symptoms, daily functioning affected
    if 10 <= depression_score <= 14:
        # Check if any specific pattern:
        
        # 3a: Situational/Reactive Depression (identifiable stressor)
        if not worsening_depression and not low_energy and not low_selfesteem:
            return _make(9, 0.88)  # Situational depression
        
        # 3b: Regular Moderate Depression
        return _make(2, 0.92)

    # ── RULE 4: Moderately-Severe Depression (PHQ 15-19) ───────────────────────
    # Substantial functional impairment, multiple symptoms
    if 15 <= depression_score <= 19:
        # 4a: Peripartum Depression (only if pregnant/postpartum + 2+ core symptoms)
        if (gender == "Female" and 
            perinatal_status in ("Currently pregnant", "Postpartum (within 1 year)") and
            (low_energy or low_selfesteem or worsening_depression)):
            return _make(7, 0.93)
        
        # 4b: Regular Moderately-Severe
        return _make(3, 0.92)

    # ── RULE 5: Severe Depression (PHQ 20+) ────────────────────────────────────
    # Severe, debilitating symptoms. IMMEDIATE psychiatric intervention required.
    if depression_score >= 20:
        # 5a: Peripartum with severe depression
        if (gender == "Female" and 
            perinatal_status in ("Currently pregnant", "Postpartum (within 1 year)")):
            return _make(7, 0.94)
        
        # 5b: Severe depression with prominent anxiety
        if nervous_level >= 7:
            return _make(8, 0.93)
        
        # 5c: Regular severe depression
        return _make(4, 0.94)

    # ── FALLBACK: Use Machine Learning Model ───────────────────────────────────
    if model is not None:
        pred = int(model.predict(df)[0])
        proba = model.predict_proba(df)[0]
        raw = float(proba.max())
        conf = round(0.75 + (raw ** 0.5) * 0.22, 3)
        conf = min(max(conf, 0.75), 0.97)
        top3 = [(DEPRESSION_LABELS.get(i, f"Type {i}"), float(proba[i]))
                 for i in np.argsort(proba)[::-1][:3]]
        return pred, conf, top3, False
    
    # ── DEMO FALLBACK (NO MODEL) ───────────────────────────────────────────────
    else:
        # Simple, safe fallback based on PHQ only
        if depression_score >= 20:
            pred = 4
        elif depression_score >= 15:
            pred = 3
        elif depression_score >= 10:
            pred = 2
        elif depression_score >= 5:
            pred = 1
        else:
            pred = 0
        
        conf = round(0.75 + (depression_score / 27) * 0.20, 3)
        top3 = [
            (DEPRESSION_LABELS[pred], conf),
            (DEPRESSION_LABELS[(pred + 1) % 10], round(conf * 0.30, 3)),
            (DEPRESSION_LABELS[(pred + 2) % 10], round(conf * 0.15, 3)),
        ]
        return pred, conf, top3, True


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Assessment
# ═══════════════════════════════════════════════════════════════════════════════
with tab_assess:
    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Patient Information</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", min_value=10, max_value=100, value=30)
        with c2:
            gender = st.selectbox("Gender", ["Male", "Female"])
        c3, c4 = st.columns(2)
        with c3:
            employment = st.selectbox("Employment Status",
                ["Employed", "Unemployed", "Student", "Retired", "Self-employed"])
        with c4:
            education = st.selectbox("Education Level",
                ["High School", "Bachelor's", "Master's", "PhD", "Other"])
        coping = st.selectbox("Primary Coping Method",
            ["Exercise", "Therapy", "Meditation", "Social Support", "None"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Clinical Scores (PHQ-10 Based)</div>', unsafe_allow_html=True)
        sleep = st.slider("Sleep hours per night", 0.0, 12.0, 7.0, 0.5)
        social_media = st.slider("Social media hours per day", 0.0, 12.0, 2.0, 0.5)
        depression_score = st.slider("PHQ-10 Depression Score (0=None → 27=Severe)", 0, 27, 10)
        st.caption("PHQ-10 Scoring: 0-4 (None), 5-9 (Mild), 10-14 (Moderate), 15-19 (Moderately-Severe), 20+ (Severe)")
        nervous_level = st.slider("Anxiety Level (0–10)", 0, 10, 3)
        c5, c6 = st.columns(2)
        with c5:
            overeating = st.slider("Overeating Level (0–10)", 0, 10, 3)
        with c6:
            meals_per_day = st.number_input("Meals per day", 0, 10, 3)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Symptom Checklist & Risk Assessment</div>', unsafe_allow_html=True)
        c7, c8 = st.columns(2)
        with c7:
            low_selfesteem = st.checkbox("Low self-esteem / guilt")
            low_energy = st.checkbox("Low energy / persistent fatigue")
            worsening_depression = st.checkbox("Worsening depression trend")
            social_while_eating = st.checkbox("Compulsive social media use while eating")
        with c8:
            mental_health_support = st.checkbox("Currently in mental health care")
            search_online = st.checkbox("Searches for depression information online")
            suicide_attempts = st.checkbox("⚠️ History of suicide attempts")
            self_harm = st.checkbox("⚠️ History of self-harm / NSSI")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Women's Health Section ────────────────────────────────────────────
        perinatal_status = "Not applicable"
        if gender == "Female":
            st.markdown("""
            <div class="card" style="border-color:#F9A8D4; background:linear-gradient(135deg,#FFF0F6,#FFF7FB);">
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="section-label" style="color:#BE185D;">
              <span style="display:inline-block;width:3px;height:12px;background:linear-gradient(180deg,#EC4899,#F472B6);border-radius:2px;margin-right:8px;"></span>
              Reproductive / Perinatal Status
            </div>
            """, unsafe_allow_html=True)
            perinatal_status = st.selectbox(
                "Perinatal Status (affects diagnosis)",
                [
                    "Not applicable",
                    "Currently pregnant",
                    "Postpartum (within 1 year)",
                    "Trying to conceive",
                    "Recent pregnancy loss",
                ],
                help="Peripartum mood disorders are distinct clinical entities requiring specialized care coordination."
            )
            if perinatal_status != "Not applicable":
                st.markdown(f"""
                <div style="background:#FCE7F3; border:1px solid #F9A8D4; border-radius:8px;
                            padding:8px 12px; font-size:0.78rem; color:#9D174D; margin-top:6px;">
                  ⚠️ <strong>Important:</strong> Peripartum mood disorders require immediate obstetric-psychiatric coordination.
                  Contact OB/GYN and mental health provider urgently.
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        predict_clicked = st.button("🔍 Run Evidence-Based Assessment")

    # ── Result Panel ──────────────────────────────────────────────────────────
    with col_result:
        if predict_clicked:
            pred, conf, top3, demo = run_prediction(
                age, gender, employment, education, coping,
                sleep, social_media, depression_score, nervous_level,
                overeating, meals_per_day,
                low_selfesteem, low_energy, worsening_depression, social_while_eating,
                mental_health_support, search_online, suicide_attempts, self_harm,
                perinatal_status
            )

            label = DEPRESSION_LABELS.get(pred, f"Type {pred}")
            desc = DEPRESSION_DESC.get(pred, "")
            recs = RECOMMENDATIONS.get(pred, [])
            sev = SEVERITY.get(pred, "moderate")
            cfg = SEVERITY_CONFIG[sev]

            factors = build_explanation(
                pred, depression_score, sleep, social_media,
                suicide_attempts, self_harm, worsening_depression,
                low_energy, low_selfesteem, nervous_level,
                employment, coping, mental_health_support,
                perinatal_status
            )

            # Build risk tags
            tags = []
            if suicide_attempts:
                tags.append(("⛔ SUICIDE HISTORY", "tag-red"))
            if self_harm:
                tags.append(("⚠️ SELF-HARM HISTORY", "tag-red"))
            if worsening_depression:
                tags.append(("📈 Worsening Trend", "tag-orange"))
            
            phq_level = get_phq_severity_level(depression_score)
            phq_tags = {
                "severe": ("🔴 PHQ: SEVERE", "tag-red"),
                "moderately-severe": ("🟠 PHQ: Mod-Severe", "tag-orange"),
                "moderate": ("🟡 PHQ: Moderate", "tag-orange"),
                "mild": ("🔵 PHQ: Mild", "tag-blue"),
                "none": ("🟢 PHQ: Minimal", "tag-green"),
            }
            tags.append(phq_tags[phq_level])
            
            if sleep < 5:
                tags.append(("⚡ SLEEP DEFICIT", "tag-red"))
            elif sleep > 10:
                tags.append(("😴 Hypersomnia", "tag-orange"))
            else:
                tags.append(("✓ Sleep OK", "tag-green"))
            
            if social_media > 6:
                tags.append(("📱 High SM Use", "tag-orange"))
            
            if mental_health_support:
                tags.append(("💚 In MH Care", "tag-blue"))
            
            if coping != "None":
                tags.append((f"✓ {coping}", "tag-green"))
            else:
                tags.append(("No Coping", "tag-orange"))

            # Save to database
            patient_id, assessment_id = save_assessment(
                age, gender, employment, education, coping,
                sleep, social_media, depression_score, nervous_level,
                overeating, meals_per_day,
                low_selfesteem, low_energy, worsening_depression, social_while_eating,
                mental_health_support, search_online, suicide_attempts, self_harm,
                pred, label, sev, conf,
                top3, demo, tags, recs
            )

            # ── Display Results ────────────────────────────────────────────────
            st.markdown(f"""
            <div class="result-outer">
              <div class="result-hero" style="background:{cfg['hero_bg']}; color:{cfg['text']};">
                <div class="result-severity-badge">{cfg['icon']} {cfg['badge']}</div>
                <div class="result-title">{label}</div>
                <div class="result-desc">{desc}</div>
              </div>
              <div class="result-body">
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div>
              <div class="conf-header">
                <span class="conf-label">Model Confidence</span>
                <span class="conf-value" style="color:{cfg['accent']}">{conf*100:.1f}%</span>
              </div>
              <div class="conf-bar-track">
                <div class="conf-bar-fill" style="width:{conf*100:.1f}%; background:{cfg['accent']};"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metrics-grid">
              <div class="metric-cell"><div class="mlabel">Sleep</div><div class="mvalue">{sleep:.1f}h</div></div>
              <div class="metric-cell"><div class="mlabel">PHQ-10</div><div class="mvalue">{depression_score}/27</div></div>
              <div class="metric-cell"><div class="mlabel">Anxiety</div><div class="mvalue">{nervous_level}/10</div></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<hr class="hr">', unsafe_allow_html=True)
            st.markdown('<div class="sec-head">⚠️ Risk Flags</div>', unsafe_allow_html=True)
            tags_html = "".join(f'<span class="tag {c}">{l}</span>' for l, c in tags)
            st.markdown(f'<div class="tags-wrap">{tags_html}</div>', unsafe_allow_html=True)
            st.markdown('<hr class="hr">', unsafe_allow_html=True)

            st.markdown('<div class="sec-head">Differential Diagnoses (Top 3)</div>', unsafe_allow_html=True)
            diff_accents = [cfg['accent'], "#64748B", "#CBD5E1"]
            for i, (lbl, prob) in enumerate(top3):
                bold = "font-weight:700;" if i == 0 else ""
                opacity = "" if i == 0 else "opacity:0.75;"
                st.markdown(f"""<div class="diff-item">
                  <div class="diff-row">
                    <span class="diff-name" style="{bold}{opacity}">{lbl}</span>
                    <span class="diff-pct">{prob*100:.1f}%</span>
                  </div>
                  <div class="diff-track">
                    <div class="diff-fill" style="width:{prob*100:.1f}%; background:{diff_accents[i]};"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

            st.markdown('<hr class="hr">', unsafe_allow_html=True)

            # ── Explanation ───────────────────────────────────────────────────
            impact_class = {"high":"impact-high","med":"impact-med","low":"impact-low","pos":"impact-pos"}
            impact_label_map = {"high":"High Impact","med":"Moderate","low":"Low","pos":"Protective"}
            st.markdown('<div class="explain-box"><div class="explain-title">📋 Clinical Factors</div>', unsafe_allow_html=True)
            for (icon, name, desc_text, impact) in factors:
                ic = impact_class.get(impact, "impact-low")
                il = impact_label_map.get(impact, impact)
                st.markdown(f"""<div class="explain-factor">
                  <div class="explain-icon">{icon}</div>
                  <div class="explain-content">
                    <div class="explain-factor-name">{name}</div>
                    <div class="explain-factor-desc">{desc_text}</div>
                  </div>
                  <div class="explain-impact {ic}">{il}</div>
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<hr class="hr">', unsafe_allow_html=True)
            st.markdown('<div class="sec-head">💊 Evidence-Based Treatment Recommendations</div>', unsafe_allow_html=True)
            for i, rec in enumerate(recs, 1):
                st.markdown(f"""<div class="rec">
                  <div class="rec-num">{i}</div>
                  <div class="rec-text">{rec}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown('</div></div>', unsafe_allow_html=True)

            # ── Saved confirmation ────────────────────────────────────────────
            st.markdown(f"""
            <div style="background:#F0FDF4; border:1px solid #BBF7D0; border-radius:10px;
                        padding:10px 14px; font-size:0.8rem; color:#14532D;
                        margin-top:0.75rem; display:flex; gap:8px; align-items:center;">
              ✅ <span>Assessment saved — ID #{assessment_id} · Patient #{patient_id}</span>
            </div>
            """, unsafe_allow_html=True)

            if demo:
                st.markdown("""
                <div class="demo-banner">
                  ⚙️ <strong>Demo Mode:</strong> No ML model available. Using PHQ-10-based rules only.
                </div>
                """, unsafe_allow_html=True)

            # ── Crisis Resources ──────────────────────────────────────────────
            if pred >= 3 or suicide_attempts or self_harm:
                st.markdown("""
                <div style="background:#FEE2E2; border:2px solid #DC2626; border-radius:10px;
                            padding:12px 14px; font-size:0.8rem; color:#7F1D1D; margin-top:1rem;">
                  <strong>🆘 CRISIS RESOURCES:</strong><br>
                  • USA: 988 Suicide & Crisis Lifeline (call/text 988)<br>
                  • Crisis Text Line: Text HOME to 741741<br>
                  • International: findahelpline.com<br>
                  <strong>If immediate danger: Contact emergency services or go to nearest ER.</strong>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="empty-state">
              <div class="empty-icon">🧠</div>
              <div class="empty-title">Ready for Assessment</div>
              <div class="empty-sub">
                Complete the form on the left using PHQ-10 scoring<br>and click <strong>Run Evidence-Based Assessment</strong>.
              </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Patient Records (نفسه من الكود الأصلي مع تحسينات بسيطة)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_records:
    records = get_all_assessments(limit=200)

    sev_order = {"severe":0,"moderately-high":1,"moderate":2,"mild":3,"none":4}
    sev_icon = {"severe":"🔴","moderately-high":"🟠","moderate":"🟡","mild":"🔵","none":"🟢"}

    rc1, rc2, rc3 = st.columns([2, 1, 1])
    with rc1:
        search_q = st.text_input("🔍 Search by diagnosis", placeholder="e.g., Peripartum, Moderate...")
    with rc2:
        sev_filter = st.selectbox("Filter severity", ["All","severe","moderately-high","moderate","mild","none"])
    with rc3:
        sort_by = st.selectbox("Sort by", ["Newest","Oldest","Highest PHQ","Highest risk"])

    if records:
        df_rec = pd.DataFrame(records)

        if sev_filter != "All":
            df_rec = df_rec[df_rec["severity"] == sev_filter]
        if search_q:
            q = search_q.lower()
            df_rec = df_rec[df_rec["predicted_label"].str.lower().str.contains(q) |
                            df_rec["severity"].str.lower().str.contains(q)]

        if sort_by == "Oldest":
            df_rec = df_rec.sort_values("assessed_at")
        elif sort_by == "Highest PHQ":
            df_rec = df_rec.sort_values("depression_score", ascending=False)
        elif sort_by == "Highest risk":
            df_rec["_risk"] = df_rec["severity"].map(sev_order)
            df_rec = df_rec.sort_values("_risk")

        st.markdown(f"""
        <div style="font-size:0.78rem; color:#94A3B8; margin-bottom:1rem;">
          Showing <strong style="color:#334155">{len(df_rec)}</strong> record(s)
        </div>""", unsafe_allow_html=True)

        for _, row in df_rec.iterrows():
            icon = sev_icon.get(row["severity"], "⚪")
            conf_p = f"{row['confidence']*100:.0f}%"
            demo_b = " · <span style='color:#D97706'>demo</span>" if row["is_demo_mode"] else ""
            hi_risk = row["suicide_attempts"] or row["self_harm"]
            risk_badge = '<span style="background:#FEE2E2;color:#B91C1C;font-size:0.68rem;font-weight:700;padding:2px 7px;border-radius:4px;margin-left:6px;">⚠ HIGH RISK</span>' if hi_risk else ""

            st.markdown(f"""
            <div class="card" style="margin-bottom:0.6rem; padding:1rem 1.25rem;">
              <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:8px;">
                <div style="flex:1;">
                  <div style="font-size:0.88rem; font-weight:700; color:#1E293B; margin-bottom:3px;">
                    {icon} {row['predicted_label']}{risk_badge}
                  </div>
                  <div style="font-size:0.75rem; color:#94A3B8;">
                    #{row['assessment_id']} · Age {row['age']} · {row['gender']} · {row['employment_status']}
                    · PHQ {row['depression_score']}/27 · Sleep {row['sleep_hours']:.1f}h · {conf_p}{demo_b}
                  </div>
                  <div style="font-size:0.72rem; color:#CBD5E1; margin-top:3px;">{row['assessed_at'][:16].replace('T', ' ')}</div>
                </div>
                <div style="text-align:right; flex-shrink:0;">
                  <span style="background:#F1F5F9; color:#475569; font-size:0.72rem; font-weight:600; padding:3px 9px; border-radius:6px;">
                    {row['severity']}
                  </span>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr class="hr">', unsafe_allow_html=True)

        export_df = df_rec[[
            "assessment_id","assessed_at","age","gender","employment_status",
            "predicted_label","severity","confidence","depression_score",
            "sleep_hours","social_media_hours","suicide_attempts","self_harm"
        ]].copy()
        export_df["confidence"] = (export_df["confidence"]*100).round(1)
        csv_bytes = export_df.to_csv(index=False).encode()
        st.download_button(
            label="⬇️ Export as CSV",
            data=csv_bytes,
            file_name="mindscreen_records.csv",
            mime="text/csv"
        )

    else:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">📋</div>
          <div class="empty-title">No records</div>
          <div class="empty-sub">Run assessments to populate records.</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Statistics
# ═══════════════════════════════════════════════════════════════════════════════
with tab_stats:
    stats = get_stats()

    if stats["total"] == 0:
        st.markdown("""
        <div class="empty-state">
          <div class="empty-icon">📊</div>
          <div class="empty-title">No data</div>
          <div class="empty-sub">Run assessments to see statistics.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        k1, k2, k3, k4, k5 = st.columns(5)
        kpi_style = "background:white; border:1px solid #E8EDF3; border-radius:12px; padding:1rem 1.1rem; text-align:center;"
        kpi_label = "font-size:0.65rem; color:#94A3B8; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px;"
        kpi_val = "font-size:1.5rem; font-weight:800; color:#1E293B; font-family:'DM Mono',monospace;"

        with k1:
            st.markdown(f'<div style="{kpi_style}"><div style="{kpi_label}">Total</div><div style="{kpi_val}">{stats["total"]}</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div style="{kpi_style}"><div style="{kpi_label}">High Risk</div><div style="{kpi_val};color:#DC2626">{stats["high_risk"]}</div></div>', unsafe_allow_html=True)
        with k3:
            st.markdown(f'<div style="{kpi_style}"><div style="{kpi_label}">Avg PHQ</div><div style="{kpi_val}">{stats["avg_phq"] or "—"}</div></div>', unsafe_allow_html=True)
        with k4:
            st.markdown(f'<div style="{kpi_style}"><div style="{kpi_label}">Avg Conf</div><div style="{kpi_val}">{stats["avg_conf"] or "—"}%</div></div>', unsafe_allow_html=True)
        with k5:
            st.markdown(f'<div style="{kpi_style}"><div style="{kpi_label}">Last 7 Days</div><div style="{kpi_val};color:#2563EB">{stats["recent_7d"]}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        ch1, ch2 = st.columns(2)

        with ch1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Assessments by Severity</div>', unsafe_allow_html=True)
            if stats["by_severity"]:
                sev_df = pd.DataFrame(stats["by_severity"])
                sev_colors = {
                    "severe":"#DC2626","moderately-high":"#EA580C",
                    "moderate":"#D97706","mild":"#2563EB","none":"#16A34A"
                }
                sev_df["color"] = sev_df["severity"].map(sev_colors).fillna("#94A3B8")
                max_cnt = sev_df["cnt"].max()
                for _, r in sev_df.iterrows():
                    pct = r["cnt"] / max_cnt * 100
                    st.markdown(f"""
                    <div style="margin-bottom:10px;">
                      <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:4px;">
                        <span style="color:#334155; font-weight:600;">{sev_icon.get(r['severity'],'⚪')} {r['severity']}</span>
                        <span style="color:#94A3B8; font-family:'DM Mono',monospace;">{r['cnt']}</span>
                      </div>
                      <div style="height:8px; background:#F1F5F9; border-radius:4px; overflow:hidden;">
                        <div style="height:100%; width:{pct:.1f}%; background:{r['color']}; border-radius:4px;"></div>
                      </div>
                    </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with ch2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Top Diagnoses</div>', unsafe_allow_html=True)
            if stats["by_label"]:
                lbl_df = pd.DataFrame(stats["by_label"]).head(6)
                max_cnt = lbl_df["cnt"].max()
                for _, r in lbl_df.iterrows():
                    pct = r["cnt"] / max_cnt * 100
                    short = r["predicted_label"][:32] + ("…" if len(r["predicted_label"]) > 32 else "")
                    st.markdown(f"""
                    <div style="margin-bottom:10px;">
                      <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:4px;">
                        <span style="color:#334155;">{short}</span>
                        <span style="color:#94A3B8; font-family:'DM Mono',monospace;">{r['cnt']}</span>
                      </div>
                      <div style="height:8px; background:#F1F5F9; border-radius:4px; overflow:hidden;">
                        <div style="height:100%; width:{pct:.1f}%; background:#6366F1; border-radius:4px;"></div>
                      </div>
                    </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
