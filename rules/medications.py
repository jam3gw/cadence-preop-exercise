"""Reference data for identifying anticoagulant medications.

Enumerating the common agents here keeps Rule 3's determination deterministic
and reviewable by a clinician without reading a prompt or running a model.
Organised by the pharmacological classes in ATC group B01A, covering agents in
routine use under generic and common US brand names.

Not exhaustive: anything absent falls through to the model-backed fallback in
`rules.classifiers`. Names are matched token-wise after case folding, so
entries must be single lowercase tokens.
"""

from __future__ import annotations

import re

# --- Anticoagulants -------------------------------------------------------

#: Vitamin K antagonists.
VITAMIN_K_ANTAGONISTS = {
    "warfarin",
    "coumadin",
    "jantoven",
    "acenocoumarol",
    "phenprocoumon",
}

#: Direct oral anticoagulants -- direct factor Xa inhibitors.
FACTOR_XA_INHIBITORS = {
    "apixaban",
    "eliquis",
    "rivaroxaban",
    "xarelto",
    "edoxaban",
    "savaysa",
    "lixiana",
    "betrixaban",
    "bevyxxa",
}

#: Direct thrombin inhibitors, oral and parenteral.
DIRECT_THROMBIN_INHIBITORS = {
    "dabigatran",
    "pradaxa",
    "argatroban",
    "bivalirudin",
    "angiomax",
    "desirudin",
}

#: Unfractionated and low molecular weight heparins, and related agents.
HEPARINS = {
    "heparin",
    "enoxaparin",
    "lovenox",
    "dalteparin",
    "fragmin",
    "tinzaparin",
    "innohep",
    "nadroparin",
    "fondaparinux",
    "arixtra",
    "danaparoid",
}

ANTICOAGULANT_NAMES = (
    VITAMIN_K_ANTAGONISTS
    | FACTOR_XA_INHIBITORS
    | DIRECT_THROMBIN_INHIBITORS
    | HEPARINS
)

# --- Documented non-anticoagulants ----------------------------------------

#: Antiplatelet agents. Recorded explicitly because they are the confusable
#: case -- a model asked "is this a blood thinner?" may well say yes -- but the
#: policy speaks of anticoagulants, a distinct class, so they do not trigger
#: Rule 3. To change that, move this set into ANTICOAGULANT_NAMES above.
ANTIPLATELETS = {
    "aspirin",
    "asa",
    "clopidogrel",
    "plavix",
    "ticagrelor",
    "brilinta",
    "prasugrel",
    "effient",
    "ticlopidine",
    "dipyridamole",
    "aggrenox",
    "cilostazol",
    "pletal",
}

#: Thrombolytics. Not anticoagulants, and not plausibly present in an elective
#: pre-op medication list, but named here so the distinction is explicit.
THROMBOLYTICS = {
    "alteplase",
    "tenecteplase",
    "reteplase",
    "streptokinase",
    "urokinase",
}

#: Common chronic-disease medications that are unambiguously not
#: anticoagulants. Purely an optimisation -- every name here is one that never
#: needs a model call. Positive identification always wins over this set (see
#: `lookup`), so adding a name here can never mask an anticoagulant.
CARDIOVASCULAR = {
    "lisinopril", "enalapril", "ramipril", "benazepril", "captopril",
    "losartan", "valsartan", "olmesartan", "irbesartan", "candesartan",
    "amlodipine", "nifedipine", "felodipine", "diltiazem", "verapamil",
    "metoprolol", "atenolol", "carvedilol", "bisoprolol", "propranolol",
    "labetalol", "nebivolol",
    "hydrochlorothiazide", "chlorthalidone", "furosemide", "torsemide",
    "bumetanide", "spironolactone", "eplerenone", "triamterene",
    "digoxin", "amiodarone", "sotalol", "flecainide", "ranolazine",
    "isosorbide", "nitroglycerin", "hydralazine", "clonidine", "doxazosin",
}
ANTIDIABETICS = {
    "metformin", "glipizide", "glyburide", "glimepiride",
    "sitagliptin", "januvia", "linagliptin", "tradjenta", "saxagliptin",
    "empagliflozin", "jardiance", "dapagliflozin", "farxiga", "canagliflozin",
    "liraglutide", "victoza", "semaglutide", "ozempic", "dulaglutide", "trulicity",
    "pioglitazone", "acarbose", "repaglinide",
    "insulin", "lantus", "levemir", "humalog", "novolog", "tresiba",
}
LIPID_AGENTS = {
    "atorvastatin", "lipitor", "simvastatin", "zocor", "rosuvastatin", "crestor",
    "pravastatin", "lovastatin", "pitavastatin",
    "ezetimibe", "zetia", "fenofibrate", "gemfibrozil", "niacin",
    "evolocumab", "repatha", "alirocumab",
}
GASTROINTESTINAL = {
    "omeprazole", "prilosec", "pantoprazole", "protonix", "esomeprazole",
    "nexium", "lansoprazole", "rabeprazole",
    "famotidine", "pepcid", "ranitidine", "sucralfate",
    "ondansetron", "zofran", "docusate", "senna", "polyethylene", "bisacodyl",
}
ENDOCRINE_AND_BONE = {
    "levothyroxine", "synthroid", "liothyronine", "methimazole",
    "propylthiouracil", "prednisone", "prednisolone", "hydrocortisone",
    "dexamethasone", "alendronate", "fosamax", "risedronate", "zoledronic",
    "denosumab", "prolia", "raloxifene", "estradiol", "testosterone",
}
RESPIRATORY = {
    "albuterol", "ventolin", "proair", "levalbuterol", "ipratropium",
    "tiotropium", "spiriva", "fluticasone", "flovent", "budesonide",
    "montelukast", "singulair", "salmeterol", "formoterol", "theophylline",
}
NEUROPSYCHIATRIC = {
    "sertraline", "zoloft", "fluoxetine", "prozac", "escitalopram", "lexapro",
    "citalopram", "celexa", "paroxetine", "duloxetine", "cymbalta",
    "venlafaxine", "bupropion", "wellbutrin", "mirtazapine", "trazodone",
    "amitriptyline", "nortriptyline", "buspirone",
    "alprazolam", "lorazepam", "clonazepam", "diazepam", "zolpidem", "ambien",
    "gabapentin", "neurontin", "pregabalin", "lyrica", "levetiracetam",
    "keppra", "lamotrigine", "topiramate", "phenytoin", "valproate",
    "carbidopa", "levodopa", "donepezil", "memantine", "sumatriptan",
    "quetiapine", "risperidone", "aripiprazole", "olanzapine", "lithium",
}
ANALGESICS = {
    "acetaminophen", "tylenol", "paracetamol",
    "ibuprofen", "advil", "motrin", "naproxen", "aleve", "meloxicam",
    "celecoxib", "celebrex", "indomethacin", "diclofenac", "ketorolac",
    "tramadol", "oxycodone", "hydrocodone", "morphine", "hydromorphone",
    "codeine", "methadone", "buprenorphine", "cyclobenzaprine", "baclofen",
    "tizanidine", "methocarbamol",
}
OTHER_COMMON = {
    "allopurinol", "colchicine", "febuxostat",
    "methotrexate", "hydroxychloroquine", "sulfasalazine", "azathioprine",
    "tamsulosin", "flomax", "finasteride", "oxybutynin", "tolterodine",
    "sildenafil", "tadalafil",
    "loratadine", "cetirizine", "fexofenadine", "diphenhydramine", "benadryl",
    "amoxicillin", "azithromycin", "cephalexin", "doxycycline",
    "ciprofloxacin", "nitrofurantoin", "valacyclovir", "fluconazole",
    "vitamin", "cholecalciferol", "ergocalciferol", "calcium", "magnesium",
    "folic", "ferrous", "cyanocobalamin", "thiamine", "potassium",
    "melatonin", "probiotic", "multivitamin",
}

COMMON_NON_ANTICOAGULANTS = (
    CARDIOVASCULAR
    | ANTIDIABETICS
    | LIPID_AGENTS
    | GASTROINTESTINAL
    | ENDOCRINE_AND_BONE
    | RESPIRATORY
    | NEUROPSYCHIATRIC
    | ANALGESICS
    | OTHER_COMMON
)

NON_ANTICOAGULANT_NAMES = ANTIPLATELETS | THROMBOLYTICS | COMMON_NON_ANTICOAGULANTS

_TOKEN_RE = re.compile(r"[a-z]+")


def tokenize(name: str | None) -> list[str]:
    """Split a medication name into comparable lowercase word tokens.

    Strips dosage and formulation noise: "Eliquis 5mg BID" -> ["eliquis", "mg",
    "bid"].
    """

    if not name:
        return []
    return _TOKEN_RE.findall(name.casefold())


def lookup(name: str | None) -> bool | None:
    """Classify a name against the reference lists.

    True for a known anticoagulant, False for a name explicitly known not to be
    one, None when it is in neither -- the signal to fall back to the model.
    """

    tokens = tokenize(name)
    if not tokens:
        return None

    # A combination product naming an anticoagulant is still an anticoagulant.
    if any(token in ANTICOAGULANT_NAMES for token in tokens):
        return True
    if any(token in NON_ANTICOAGULANT_NAMES for token in tokens):
        return False
    return None
