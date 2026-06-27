#!/usr/bin/env python3
"""
First-pass harm TAXONOMY + the hyperbolic-traction gate.

Turns the binary harmful/benign task into a hierarchy (tree), so we can test whether
harm STRUCTURE has hyperbolic traction in the model's representations -- the prerequisite
for the hierarchical-hyperbolic detector (see hierarchical_hyperbolic_design.md).

Three stages:
  --build   : assign behaviors (CSV) to taxonomy leaves by keyword; save mapping + tree.   (CPU)
  --extract : forward each behavior, save its last-token representation per leaf.           (GPU)
  --gate    : per-leaf centroids -> (i) does the category geometry fit H better than E?
              (ii) do rep distances correlate with taxonomy tree distances?                 (CPU)

If the gate shows H-win AND positive correlation, harm categories are organized
hyperbolically in representation space -> green light for the hierarchical detector.
A first-pass keyword assignment is approximate; refine with LLM/manual labels before
drawing strong conclusions.
"""
import argparse, json, os, sys
import numpy as np

# 2-3 level harm taxonomy; leaves carry keyword cues for first-pass assignment.
TAXONOMY = {
    "violence": {
        "weapons": ["weapon", "gun", "firearm", "bomb", "explosive", "ammunition", "grenade"],
        "physical_harm": ["harm someone", "kill", "attack", "hurt", "injure", "torture", "poison", "toxify", "food or drink", "without leaving a trace"],
    },
    "cyber": {
        "malware": ["malware", "virus", "ransomware", "exploit", "ddos", "keylogger", "spyware"],
        "intrusion": ["hack", "credential", "password", "unauthorized access", "break into", "bypass security"],
    },
    "fraud_theft": {
        "financial_fraud": ["credit card", "scam", "fraud", "launder", "embezzle", "economic crime", "counterfeit", "ponzi"],
        "theft": ["steal", "rob", "shoplift", "steal a car", "burglar"],
        "coercion": ["manipulate", "blackmail", "extort", "coerce", "deceive someone"],
    },
    "controlled_substances": {
        "synthesis": ["synthesize", "drug", "narcotic", "meth", "potent agent", "substance", "cocaine", "fentanyl"],
    },
    "misinformation": {
        "fake_content": ["fake news", "misinformation", "propaganda", "conspiracy", "hoax", "disinformation"],
    },
    "hate_harassment": {
        "discrimination": ["race", "ethnicity", "racist", "sexist", "discriminat", "supremac"],
        "harassment": ["harass", "bully", "threaten", "stalk", "intimidate"],
    },
    "self_harm": {
        "self_harm": ["self-harm", "self harm", "suicide", "cutting"],
    },
    "privacy": {
        "surveillance": ["personal information", "track someone", "surveillance", "dox", "private data", "spy on"],
    },
    "regulatory_evasion": {
        "illegal_operation": ["without following", "regulations", "evade tax", "license", "permit", "launder"],
    },
}


# 3rd-level sub-taxonomy (#1: deepen the tree to 3 levels to test whether the curved-vs-Euclidean
# retrieval edge grows with depth). DATA-GROUNDED: sub-categories were derived by reading the actual
# prompts in each leaf (one pass per leaf), NOT clustered from the representations (avoids circularity
# with the geometry we then measure). Keyword cues drive a max-hits fallback; the LLM labeler (cmd_
# deepen) refines. Designed so <~10% of prompts need the 'general' bucket (was 64% in the first pass).
SUBTAXONOMY = {
    ('misinformation', 'fake_content'): {
        'health_medical_pandemic_misinfo': ['covid', 'pandemic', 'vaccine', 'vaccination', 'medical treatment', 'fake medical', 'medical information', 'disease', 'mental health', 'smoking', '5g', 'cancer', 'cure covid', 'dangerous vaccines', 'autism', 'seeking help', 'seeking expensive'],
        'political_election_propaganda': ['political candidate', 'political propaganda', 'public opinion', 'election', 'sway the election', 'policy', 'political outcome', 'political ideology', 'propaganda', 'political event', 'political figure', 'anti-semitic', 'influence political', 'political decisions', 'name game'],
        'fake_news_articles_hoaxes': ['fake news article', 'fake news story', 'fake news headlines', 'fake news', 'create a fake news', 'write a fake news', 'compose a fake news', 'hoax', 'fake texts', 'fake healthcare', 'create a fake', 'fake degrees', 'spread like wildfire', 'false report', 'generate panic'],
        'rumors_reputation_defamation': ['rumor', 'rumors', 'reputation', 'discredit', 'defamation', 'derail their career', 'job performance', 'competitor', 'intelligence', 'sexual scandal', 'damage someone', 'public figure', 'start a rumor'],
        'conspiracy_climate_science_denial': ['conspiracy theory', 'conspiracy theories', 'illuminati', 'global warming', 'climate change', 'climate emergency', 'fish stocks', 'systemic racism', 'denies the existence', 'population control', 'engineered', 'fabricated event', 'myth', 'scientific research', 'manipulate research'],
        'general_disinfo_techniques_explicit': ['disinformation', 'mislead', 'false information', 'misinformation', 'falsehoods', 'manipulate public', 'make up things', 'explicit content', 'explicit sexual', 'harmful content', 'inappropriate for children', 'manipulate words', 'cognitive dissonance', 'persuasive falsehoods', 'financial news'],
    },
    ('cyber', 'intrusion'): {
        'credential_theft_phishing': ['keylogger', 'phishing', 'steal passwords', 'crack passwords', "steal someone's passwords", 'passwords from', 'private passwords', 'password using', 'password without', "use someone's password", 'crack', 'credentials', 'phishing email', 'phishing websites', 'phishing campaign', 'malicious email'],
        'unauthorized_account_data_access': ['email account', 'social media account', "someone's account", 'private email', "spouse's email", 'cell phone', 'celebrity', 'personal data', 'private conversation', 'without their consent', 'without them knowing', 'confidential healthcare', 'security footage', 'private phone numbers', 'scrape', 'twitter data', 'personal information'],
        'system_network_intrusion': ['hack into', 'breach', 'infiltrate', 'gain unauthorized access', 'unauthorized access', 'bypass security', 'secure system', 'secure network', 'exploit vulnerabilit', 'exploit a vulnerability', 'exploits a vulnerability', 'take control of', 'government system', 'government agency', 'government database', 'financial system', "bank's", "company's network", 'classified information', 'restricted areas', 'successful breach', 'bypass security measures'],
        'malware_botnet_ddos': ['worm', 'botnet', 'ddos', 'dns amplification', 'dns query amplification', 'amplification attack', 'install malware', 'macro that can steal', 'remote access tools', 'infect and spread', 'compromised computers', 'inject', 'worm into', 'spread through'],
        'evasion_anti_forensics': ['without getting caught', 'without being detected', 'without leaving any traces', 'hide information from an investigation', 'harder to record', "overrides the phone's default", 'uninvestigated', 'avoid detection', 'without being noticed'],
    },
    ('violence', 'physical_harm'): {
        'weapons_explosives_and_traps': ['flamethrower', 'explosive', 'bioweapon', 'nerve gas', 'chlorine gas', 'poisonous gas', 'dirty bomb', 'detonating', 'pepper spray', 'ricin', 'booby trap', 'dangerous trap', 'razor blade', 'design a weapon', 'firearms', 'arson', 'burn down', 'destroy a building', 'destruction', 'dangerous scientific experiments'],
        'poisoning_and_covert_harm': ['toxify', 'poison', 'add to food', 'add what to food', 'untraceable', 'without leaving a trace', 'without getting caught', 'increase the toxicity', 'oxycodone overdose', 'deadly poison', 'harm someone physically', 'energy drinks', 'dangerous levels', 'sabotage'],
        'direct_assault_killing_and_threats': ['kill someone', 'get away with murder', 'hurt someone in a fight', 'combat techniques', 'attack a vulnerable', 'death threat', 'runs them over', 'hit that kid', 'join a violent gang', 'attack other children', 'engage in violence', 'incitement of violence', 'most efficient ways to attack'],
        'terrorism_and_mass_harm_operations': ['terrorist attack', 'terror attack', 'terrorism', 'bioterrorism', 'torture a terrorist', 'recruit and train', 'infiltrate and dismantle', 'widespread fear', 'false sense of security', 'panic', 'revolution', 'social engineering', 'ddos', 'ransomware', 'government database', 'stock exchange'],
        'animal_cruelty': ['animal abuse', 'hurt a cat', 'kill an animal', 'aquatic animals', 'violence against animals', 'make a dog aggressive', 'killing for their meat', 'harm an animal', 'scream and cry', 'less kind'],
        'self_harm_child_abuse_and_other_personal_harm': ['hurt myself', 'self-harm', 'physically abused a student', 'force a child', 'excessive punishments', 'unsanitary working environment', 'harmful or unsanitary', 'illegal or explicit sexual', 'profane and inappropriate', 'discriminatory post', 'evade the consequences'],
    },
    ('fraud_theft', 'financial_fraud'): {
        'scams_cons_phishing': ['scam', 'ponzi scheme', 'pyramid scheme', 'lottery scam', 'phishing', 'tech support scam', 'investment scheme', 'investment scheme in india', 'quick-fix scam', 'fraudulent scam', 'needs money for treatment', 'critically ill', 'social engineering', 'imitating a legitimate business', 'trick immigrants', 'email scams', 'foreclosure scam', 'roof repair', 'deceive people'],
        'forgery_counterfeiting': ['fake legal document', 'fake passport', 'fake identification', 'fake identity', 'fake money', 'fake insurance policy', 'fake travel documents', 'fake resume', 'forge a bank check', 'counterfeit items', 'counterfeit drugs', 'altered record', 'passport to fool immigration', 'create fake', 'false shortage'],
        'money_laundering_tax_evasion': ['launder money', 'money laundering', 'money laundering operation', 'evade us taxes', 'avoid paying taxes', 'move money out of the country', 'hide or encrypt credit card', 'illegal transaction', 'illegal trade', 'illegal ways to make money', 'anonymous online platforms', 'selling illegal substances', 'illegal or counterfeit drugs'],
        'identity_credit_card_theft': ['credit card information', 'steal credit card', 'sell stolen credit card', 'identity theft', 'target individuals for identity theft', 'botnet to steal', 'steal from', 'phishing campaign targeting financial', 'phishing emails', 'consumer health data', 'medical history', 'exploit consumer health'],
        'market_manipulation_corruption_insider': ['manipulate financial markets', 'manipulate currency', 'currency rates', 'manipulate prices', 'bribe a politician', 'exploit my client', 'insider', 'benefit a small group', 'risky ventures', 'manipulate a machine learning model', 'black hat seo', 'economic crime', 'financial fraud', 'cheat the system', 'make someone lose money'],
        'cheating_ip_copyright_theft': ['cheat on an exam', 'cheat on a test', 'cheat on a drug test', 'cheat at online gambling', 'intellectual property theft', 'copyrighted music video', 'copyright-infringing', 'illegal streaming', 'take credit for someone else', 'drug test', 'hiring process'],
    },
    ('controlled_substances', 'synthesis'): {
        'home_synthesis_recipes_and_extraction': ['synthesize', 'manufacture cocaine', 'make my own meth', 'make meth', 'make illegal drugs', 'make illegal drugs at home', 'recipe for making', 'drug recipe', 'homemade drug', 'household chemicals', 'household items', 'extract it', 'produce an illegal drug', 'manufacture and distribute', 'step-by-step instructions on how to manufacture', 'crack cocaine', 'make a powerful drug', 'pseudoephedrine', 'electrostatics to synthesize', 'easiest substances to synthesize', 'synthesize a common illegal drug', 'make a potent homemade drug'],
        'lethal_poison_and_bioweapon_production': ['deadly substance', 'deadly drug', 'lethal drug', 'lethal reaction', 'poison', 'anthrax', 'botulinum toxin', 'ricin', 'bio weapon', 'bioweapon', 'dangerous chemicals', 'manipulate chemical elements', 'mildew', 'yellow cross', 'incapacitate someone', 'lethal', 'deadly'],
        'acquisition_dealing_and_trafficking': ['purchase illegal', 'buying or selling', 'drug dealing', 'acquire and sell', 'sell illegal drugs', 'black market', 'smuggle drugs', 'drug trafficking', 'drug money', 'syndicate', 'buying cocaine', 'find a source', 'free samples', 'obtain and use', 'obtain drugs', 'drug distribution', 'selling illegal drugs', 'purchase illegal substances', 'buying cocaine or heroin'],
        'evasion_and_concealment': ['without getting caught', 'avoid legal prosecution', 'under the radar', 'dodge the police', 'get away with', 'hide my drug addiction', 'use drugs without getting caught', 'intimidate someone', 'without increasing the dosage', 'make my drug more effective', 'keep people coming back'],
        'psychoactive_effects_and_promotion': ['herbs to induce hallucinations', 'induce hallucinations', 'hallucinations and suicidal', 'plant-based drugs', 'recreational drugs', 'optimal results', 'promotes the use', 'convince someone to use', 'benefit my personal life', 'highly addictive', 'extreme addiction', 'addict people', 'harmful opioid', 'guide me through the steps of conducting research', 'develop a new drug'],
    },
    ('fraud_theft', 'coercion'): {
        'blackmail_ransom_extortion_coverup': ['blackmail', 'ransom note', 'ransom', 'con artist', 'demanding', 'fear and desperation', 'cover up', 'crime', 'confidential information', 'get confidential'],
        'emotional_psychological_manipulation_of_individuals': ['emotional manipulation', 'emotionally manipulate', 'psychological techniques', 'flattery', 'insecurity', 'vulnerability', 'self-esteem', 'sense of self', 'kindness to manipulate', 'stop seeing their therapist', 'romantic relationship', 'deceive someone', 'manipulate a child', 'child to get', 'personal assistant', 'confess their lies', 'power over them', 'gain influence', 'gain power'],
        'manipulation_for_financial_gain': ['financial advice', 'hasty financial decisions', 'financial gain', "loan that they can't afford", 'give me a loan', 'personal gain', 'personal benefit', 'exploit someone for', 'manipulate others for financial', 'pressured into making'],
        'manipulating_political_government_legal_processes': ['government', 'political campaign', 'decision-making process', 'legal action', 'illegal activities', 'engage in illegal', 'extreme actions', 'infiltrate', 'manipulate the outcome', 'without raising suspicion'],
        'coercion_forcing_against_will': ['force someone', 'force someone to do', 'forced to do', 'against their will', 'make someone feel forced', "do something they don't want"],
    },
    ('regulatory_evasion', 'illegal_operation'): {
        'surveillance_and_personal_data_exploitation': ['spy', 'track their movements', 'personal information', 'personal data', 'without their consent', 'without permission', 'records audio', 'scrapes personal', 'ancestry', 'secretly records', 'employee communications', 'for my own gain', 'online activities from my employer', 'highest bidder', 'dark web'],
        'tax_evasion': ['paying taxes', 'international taxes', 'rental income', 'taxes on my income', 'evade', 'tax'],
        'safety_regulation_bypass': ['safety regulations', 'safety measures', 'hazardous materials', 'manufacturing facility', 'evade security measures', 'laboratory', 'bypass safety', 'transport hazardous'],
        'smuggling_and_evidence_disposal': ['smuggle', 'sharks', 'tai market', 'dispose of evidence', 'getting caught'],
        'legal_and_debt_evasion': ['debt collection', 'legal judgment', 'legal procedures', 'circumvent the law', 'legal loophole', 'trademark infringement', 'get away with it', 'avoid making any payments'],
        'unlicensed_practice_and_regulatory_skirting': ['diagnose a patient', 'without proper training', 'licensure', 'micro-mobility business', 'without following local regulations', 'regulations and laws'],
    },
    ('hate_harassment', 'discrimination'): {
        'produce_hateful_content_artifact': ['write a', 'social media campaign', 'social media post', 'craft a', 'blog post', 'persuasive article', 'write a post', 'generate a text', 'screenplay', 'racist speech', 'develop a social media', 'post that', 'generate offensive phrases', 'article claiming'],
        'promote_discrimination_or_oppression_action': ['encourage discrimination', 'promotes hate speech and discrimination', 'should be controlled and oppressed', 'hire and fire minorities', 'discrimination guide', 'stay in their own race', 'discriminating against others', 'undermine the rights', 'reinforce harmful cultural norms', 'controlled and oppressed', 'discrimination against people', 'successful by discriminating'],
        'spread_dehumanizing_beliefs_or_conspiracies': ['convince people that', 'transgender individuals are dangerous', 'anti-semitic', 'conspiracy theories', 'inherently criminal', 'poses a threat', 'interracial marriage is wrong', 'negative stereotype', 'promote racist ideas', 'complains about trans', 'negatively impact', 'spreading'],
        'request_slurs_or_derogatory_language': ['derogatory way', 'derogatory term', 'n-word', 'offensive phrases', 'misogynist language', 'critical of women', 'refer to a group'],
        'supremacist_or_off_topic': ['white supremacist', 'violent coup', 'removing all non-white', 'harm a pet', 'inappropriate things', 'stay-at-home mom'],
    },
    ('fraud_theft', 'theft'): {
        'identity_and_personal_data_theft': ['identity', 'credit card information', 'personal information', 'personal data', 'account information', "steal a child's identity", 'online identity', 'wallet', 'internet-accessible accounts', "stealing someone's identity"],
        'vehicle_theft_and_smuggling': ['steal a car', 'steal cars', 'stealing cars', 'steal a competitor', 'smuggling a car', 'smuggling animals', 'siphon gas', "competitor's website domain", 'smuggling', "steal someone's car", 'domain name'],
        'bank_robbery_and_heist_planning': ['rob a bank', 'robbery', 'steal money from banks', 'stealing money from a bank', 'high-security item', 'diamond from a museum', 'plan for stealing', 'avoid getting caught', 'successfully execute'],
        'shoplifting_charity_and_evasion': ['stealing from a store', 'steal from a charity', 'get away with stealing', 'sneak out of school', 'without getting caught', "steal someone's ideas", 'charity'],
    },
    ('hate_harassment', 'harassment'): {
        'threats_intimidation_stalking': ['threaten', 'terrorizing tactics', 'intimidate', 'stalking to gain leverage', 'fearful of speaking', 'fearful of authority', 'make a child fearful', 'leverage in a business', 'threat'],
        'demean_belittle_psychological': ['feel like an outsider', 'feel belittled', 'feel insecure', 'feel really small', 'feel small and powerless', 'feel bad about themselves', 'make someone feel', 'made someone feel bullied', 'powerless', 'insecure'],
        'group_targeted_campaigns': ['specific group', 'particular group of people', 'toxic social media environment', 'sms campaign to insult', 'cyberbullying campaigns against', 'lgbtqi', 'cultural norms to intimidate', 'shaming and humiliating people based on their appearance', 'social media account dedicated'],
        'personal_attacks_reputation_doxxing': ['personal attacks and insults', 'anonymous letter', "destroy someone's reputation", 'exposing their personal information', 'letter that includes personal', 'reputation', 'personal information'],
        'promote_normalize_bullying': ['encourages bullying in schools', 'necessary part of socialization', 'relentlessly bully and taunt', 'fictional group chat', 'vinnie jones character', 'harassing a stranger online', 'bully and taunt', 'encourages bullying'],
    },
    ('self_harm', 'self_harm'): {
        'encourage_promote_others': ['encourages self-harm', 'encourage self-harm', 'promotes suicide', 'increasing the number of suicides', 'cutting off support', 'false hope', 'negative consequences of seeking help', 'solution to struggling', 'self-destructive song', 'promote suicide', 'encourage'],
        'concealment_avoid_detection': ['without getting caught', 'avoid getting caught', 'without leaving marks', 'difficult to heal', 'without being blocked', 'community standards', 'avoiding breaking any rules', 'in public', 'getting caught'],
        'lethal_substance_preparation': ['chemicals', 'mix them', 'ingest', 'poisonous berries', 'most lethal', 'manufacture', 'mass suicide', 'what chemicals to use', 'poison'],
        'self_harm_method_manuals': ['manual for self-harming', 'how to self-harm', 'self-harm video', 'video guide', 'while camping', 'papua new guinea', 'how to commit self-harm', 'tutorial', 'script for a self-harm'],
    },
    ('privacy', 'surveillance'): {
        'doxxing_and_public_exposure': ['dox', 'publicly available', 'make personal information', 'without getting caught', 'expose', 'publish', 'post their', 'leak their', 'release personal'],
        'unauthorized_access_to_private_info': ['accessing their private', 'private information', 'access someone', "violate someone's privacy", 'without their knowledge', "invade someone's privacy", 'without consent', 'snoop', 'spy on'],
        'data_breach_and_leak': ['data leak', 'privacy breach', "abuse someone's privacy", 'breach', 'leak', 'exfiltrate', 'steal data', 'hack into'],
        'biometric_and_tech_surveillance': ['biometric', 'surveillance', 'facial recognition', 'tracking', 'monitor', 'camera', 'gps', 'spyware', 'stalkerware', 'geolocation'],
    },
    ('violence', 'weapons'): {
        'firearms_and_ammunition': ['gun', 'firearm', 'rifle', 'pistol', 'handgun', 'ammunition', 'bullet', 'silencer', 'untraceable gun', 'ghost gun', '3d-printed', 'convert', 'automatic'],
        'explosives_and_incendiary': ['bomb', 'explosive', 'grenade', 'detonate', 'ied', 'pipe bomb', 'molotov', 'incendiary', 'blast', 'fuse', 'c4', 'dynamite', 'fertilizer'],
        'chemical_biological_radiological': ['chemical weapon', 'nerve agent', 'nerve gas', 'chlorine', 'mustard gas', 'biological weapon', 'bioweapon', 'anthrax', 'toxin', 'ricin', 'dirty bomb', 'radiological'],
        'acquisition_and_concealment': ['buy a weapon', 'acquire', 'obtain illegally', 'black market', 'smuggle', 'conceal a weapon', 'without a license', 'evade background check', 'modify a weapon'],
        'edged_and_improvised_weapons': ['knife', 'blade', 'machete', 'improvised weapon', 'sharpen', 'stab', 'melee', 'blunt weapon'],
    },
    ('cyber', 'malware'): {
        'ransomware_and_extortion': ['ransomware', 'encrypt files', 'ransom', 'lock the system', 'extortion payload', 'crypto-locker'],
        'virus_worm_self_propagating': ['virus', 'worm', 'self-replicat', 'self-propagat', 'infect and spread', 'spread through a network', 'payload'],
        'spyware_keylogger_surveillance': ['spyware', 'keylogger', 'keystroke', 'monitor activity', 'stalkerware', 'record screen', 'exfiltrate data', 'track keystrokes'],
        'botnet_rat_remote_control': ['botnet', 'remote access', 'rat', 'command and control', 'c2', 'backdoor', 'trojan', 'zombie', 'compromised computers'],
        'development_and_evasion': ['write malware', 'create malware', 'obfuscate', 'evade antivirus', 'bypass detection', 'polymorphic', 'undetectable', 'inject code'],
    },
}
def assign_subleaf(prompt, cat, leaf):
    """Keyword sub-assignment. Returns the sub-leaf with the MOST keyword hits (not first-match,
    which biased toward earlier sub-leaves and dumped ties into 'general'). 'general' only if a
    sub-taxonomy is defined but NO keyword matches at all."""
    sub = SUBTAXONOMY.get((cat, leaf))
    if not sub:
        return "general"
    p = prompt.lower()
    scores = {sl: sum(1 for kw in kws if kw in p) for sl, kws in sub.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def assign(prompt):
    p = prompt.lower()
    for cat, leaves in TAXONOMY.items():
        for leaf, kws in leaves.items():
            if any(kw in p for kw in kws):
                return cat, leaf
    return "unassigned", "unassigned"


def leaf_nodes():
    return [(c, l) for c in TAXONOMY for l in TAXONOMY[c]]


def tree_distance(a, b):
    """Path length in the taxonomy tree between two (cat,leaf) nodes (root depth=0)."""
    if a == b: return 0
    (ca, la), (cb, lb) = a, b
    if ca == cb: return 2          # leaf-cat-leaf
    return 4                       # leaf-cat-root-cat-leaf


def cmd_build(args):
    import pandas as pd
    df = pd.read_csv(args.harmful_csv)
    col = "prompt" if "prompt" in df.columns else df.columns[0]
    rows = []
    for p in df[col].dropna().astype(str):
        c, l = assign(p); rows.append({"prompt": p, "category": c, "leaf": l})
    from collections import Counter
    cov = Counter((r["category"], r["leaf"]) for r in rows)
    n = len(rows); unassigned = sum(1 for r in rows if r["category"] == "unassigned")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump({"taxonomy": TAXONOMY, "assignments": rows}, open(args.output + ".json", "w"), indent=2)
    print(f"[tax] {n} behaviors; assigned {n - unassigned} ({100*(n-unassigned)/n:.0f}%), unassigned {unassigned}", flush=True)
    print("[tax] coverage by leaf:", flush=True)
    for (c, l), k in sorted(cov.items(), key=lambda x: -x[1]):
        print(f"    {c:22s} / {l:18s} : {k}", flush=True)
    print(f"[tax] wrote {args.output}.json (refine keyword assignment / use LLM labels for the real run)", flush=True)


def cmd_extract(args):
    import torch, pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tax = json.load(open(args.taxonomy_json)); rows = tax["assignments"]
    rows = [r for r in rows if r["category"] != "unassigned"]
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map="cuda").eval()
    reps, labels = [], []
    with torch.no_grad():
        for r in rows:
            ids = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                          return_tensors="pt", add_generation_prompt=True).to("cuda")
            hs = model(ids, output_hidden_states=True).hidden_states[args.layer][0, -1].float().cpu().numpy()
            reps.append(hs); labels.append(f"{r['category']}/{r['leaf']}")
    np.savez(args.output + "_reps.npz", reps=np.array(reps), labels=np.array(labels))
    print(f"[tax] extracted {len(reps)} behavior reps at layer {args.layer} -> {args.output}_reps.npz", flush=True)


def cmd_gate(args):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
    from embedding_distortion import fit_all
    from scipy.spatial.distance import cdist
    from scipy.stats import spearmanr
    d = np.load(args.reps_npz, allow_pickle=True)
    reps, labels = d["reps"], d["labels"]
    leaves = sorted(set(labels.tolist()))
    parse = lambda s: tuple(s.split("/"))
    Dtree = np.array([[tree_distance(parse(a), parse(b)) for b in leaves] for a in leaves], float)
    iu = np.triu_indices(len(leaves), 1)
    n_emb = max(2, min(5, len(leaves) - 2))
    rng = np.random.default_rng(0)

    def centroids(lab):
        c = np.stack([reps[lab == lf].mean(0) for lf in leaves])
        return (c - c.mean(0)) / (c.std(0) + 1e-6)

    def dmat(c):
        D = cdist(c, c); return D / np.median(D[D > 0])

    cent = centroids(labels)
    Drep = dmat(cent)
    rho_true, _ = spearmanr(Drep[iu], Dtree[iu])
    f = fit_all(Drep, n_emb); win = min(f, key=f.get)

    # BASELINE 1: matched random-Gaussian "centroids" (same K, same dim) -> H floor
    K, dim = cent.shape
    base = rng.standard_normal((K, dim)); base = (base - base.mean(0)) / (base.std(0) + 1e-6)
    fb = fit_all(dmat(base), n_emb)

    # BASELINE 2: label-shuffle permutation null for the rep<->tree correlation
    null = []
    for _ in range(300):
        c = centroids(rng.permutation(labels))
        r, _ = spearmanr(dmat(c)[iu], Dtree[iu]); null.append(r)
    null = np.array(null); p = float(np.mean(null >= rho_true))

    print(f"[tax] {len(leaves)} leaves; embed_dim={n_emb}", flush=True)
    print(f"[tax] harm-category centroids:  H={f['H']:.3f} E={f['E']:.3f} S={f['S']:.3f}  -> winner {win}", flush=True)
    print(f"[tax] random-centroid FLOOR:    H={fb['H']:.3f} E={fb['E']:.3f} S={fb['S']:.3f}   (H/floor={f['H']/fb['H']:.2f})", flush=True)
    print(f"[tax] Spearman(rep-dist, tree-dist) = {rho_true:+.3f}; shuffle-null mean={null.mean():+.3f} std={null.std():.3f}; p={p:.3f}", flush=True)
    hyp = f["H"] < 0.6 * fb["H"]
    sig = p < 0.05 and rho_true > 0
    verdict = ("GREEN LIGHT: harm categories are hyperbolic AND align with the taxonomy"
               if (win == "H" and hyp and sig) else
               "no clear hyperbolic traction: " +
               (f"H not below floor (H/floor={f['H']/fb['H']:.2f}); " if not hyp else "") +
               (f"taxonomy alignment not significant (p={p:.3f}); " if not sig else "") +
               "refine labels/layer before concluding")
    print(f"[tax] VERDICT: {verdict}", flush=True)


def cmd_label(args):
    """LLM-classify each behavior into a taxonomy leaf (local Llama-3). ~100% coverage."""
    import re, torch, pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    df = pd.read_csv(args.harmful_csv)
    col = "prompt" if "prompt" in df.columns else df.columns[0]
    prompts = df[col].dropna().astype(str).tolist()
    if args.max_n:
        prompts = prompts[: args.max_n]
    leaves = leaf_nodes()                       # list of (cat, leaf)
    opts = [f"{i+1}. {c}/{l} (e.g. {', '.join(TAXONOMY[c][l][:4])})" for i, (c, l) in enumerate(leaves)]
    other_n = len(leaves) + 1
    opts.append(f"{other_n}. other / none of the above")
    options_str = "\n".join(opts)
    sys_p = ("You are a safety content-moderation classifier. Assign the user request to exactly "
             "one harm category from the numbered taxonomy. Reply with ONLY the category number.")

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map="cuda").eval()
    rows = []
    for i, p in enumerate(prompts):
        user = f"Categories:\n{options_str}\n\nRequest: {p[:400]}\n\nCategory number:"
        ids = tok.apply_chat_template(
            [{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
            return_tensors="pt", add_generation_prompt=True).to("cuda")
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=5, do_sample=False, pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        m = re.search(r"\d+", txt)
        cat, leaf = "unassigned", "unassigned"
        if m:
            n = int(m.group())
            if 1 <= n <= len(leaves):
                cat, leaf = leaves[n - 1]
        if cat == "unassigned":                 # keyword fallback
            fc, fl = assign(p)
            if fc != "unassigned":
                cat, leaf = fc, fl
        rows.append({"prompt": p, "category": cat, "leaf": leaf})
        if (i + 1) % 100 == 0:
            print(f"[label] {i+1}/{len(prompts)}", flush=True)

    from collections import Counter
    assigned = [r for r in rows if r["category"] not in ("unassigned", "other")]
    cov = Counter((r["category"], r["leaf"]) for r in assigned)
    n = len(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump({"taxonomy": TAXONOMY, "assignments": assigned}, open(args.output + ".json", "w"), indent=2)
    print(f"[label] {n} behaviors; assigned {len(assigned)} ({100*len(assigned)/n:.0f}%) by LLM+fallback", flush=True)
    for (c, l), k in sorted(cov.items(), key=lambda x: -x[1]):
        print(f"    {c:22s} / {l:18s} : {k}", flush=True)
    print(f"[label] wrote {args.output}.json  (use as --taxonomy_json for extract)", flush=True)


def cmd_deepen(args):
    """Add a 3rd taxonomy level to an existing labeled JSON, producing labels 'category/leaf/subleaf'.

    Uses the local LLM to pick a sub-type within each leaf (keyword fallback), so the deepening is
    SEMANTIC and independent of the representations we later test (no circularity). Leaves without a
    defined sub-taxonomy pass through as 'category/leaf/general'. Output feeds `extract` unchanged;
    hyperbolic_retrieval.load_reps understands the 3-level 'a/b/c' format (shared-prefix tree dist)."""
    tax = json.load(open(args.taxonomy_json))
    rows = [r for r in tax["assignments"] if r["category"] != "unassigned"]
    use_llm = bool(args.model_path)
    if use_llm:
        import re, torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map="cuda").eval()
    out_rows = []
    for i, r in enumerate(rows):
        cat, leaf = r["category"], r["leaf"]
        sub = SUBTAXONOMY.get((cat, leaf))
        chosen = "general"
        if sub:
            if use_llm:
                opts = list(sub.keys())
                # show a few example cues per sub-category so the model commits to a real one;
                # NO 'general/other' option is offered -> the model must choose a substantive bucket.
                numbered = "\n".join(
                    f"{j+1}. {s} (e.g. {', '.join(sub[s][:4])})" for j, s in enumerate(opts))
                sysp = ("You assign a harmful request to the single best-fitting sub-category. "
                        "You MUST pick one of the numbered sub-categories — do not refuse, do not "
                        "answer 'none'. Reply with ONLY the number.")
                user = f"Harm leaf: {cat}/{leaf}\nSub-categories:\n{numbered}\n\nRequest: {r['prompt'][:400]}\n\nBest sub-category number:"
                ids = tok.apply_chat_template([{"role": "system", "content": sysp}, {"role": "user", "content": user}],
                                              return_tensors="pt", add_generation_prompt=True).to("cuda")
                with torch.no_grad():
                    g = model.generate(ids, max_new_tokens=5, do_sample=False, pad_token_id=tok.eos_token_id)
                m = re.search(r"\d+", tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True))
                if m and 1 <= int(m.group()) <= len(opts):
                    chosen = opts[int(m.group()) - 1]
                else:
                    chosen = assign_subleaf(r["prompt"], cat, leaf)   # keyword fallback (max-hits)
            else:
                chosen = assign_subleaf(r["prompt"], cat, leaf)
        out_rows.append({"prompt": r["prompt"], "category": cat, "leaf": leaf, "subleaf": chosen,
                         "label3": f"{cat}/{leaf}/{chosen}"})
        if use_llm and (i + 1) % 100 == 0:
            print(f"[deepen] {i+1}/{len(rows)}", flush=True)
    from collections import Counter
    cov = Counter(r["label3"] for r in out_rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    # write in the SAME schema extract expects, but with 3-level 'category' = the full path so
    # cmd_extract's f"{category}/{leaf}" becomes the 3-level label; simplest: store full path in
    # 'category' and a dummy leaf. Instead we store assignments with category=path, leaf="" and let
    # extract join — but to keep extract unchanged we set category=cat/leaf, leaf=subleaf.
    assignments = [{"prompt": r["prompt"], "category": f"{r['category']}/{r['leaf']}", "leaf": r["subleaf"]}
                   for r in out_rows]
    json.dump({"taxonomy": "3-level", "subtaxonomy": {f"{k[0]}/{k[1]}": list(v.keys()) for k, v in SUBTAXONOMY.items()},
               "assignments": assignments}, open(args.output + ".json", "w"), indent=2)
    print(f"[deepen] {len(out_rows)} prompts -> 3-level labels ({'LLM' if use_llm else 'keyword'}). "
          f"{len(cov)} sub-leaves:", flush=True)
    for lbl, k in sorted(cov.items(), key=lambda x: -x[1]):
        print(f"    {lbl:48s} : {k}", flush=True)
    print(f"[deepen] wrote {args.output}.json  -> next: extract --taxonomy_json {args.output}.json --layer 24", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--harmful_csv", required=True); b.add_argument("--output", default="results/harm_taxonomy")
    e = sub.add_parser("extract"); e.add_argument("--taxonomy_json", required=True); e.add_argument("--model_path", required=True)
    e.add_argument("--layer", type=int, default=13); e.add_argument("--output", default="results/harm_taxonomy")
    g = sub.add_parser("gate"); g.add_argument("--reps_npz", required=True)
    lb = sub.add_parser("label"); lb.add_argument("--harmful_csv", required=True); lb.add_argument("--model_path", required=True)
    lb.add_argument("--max_n", type=int, default=None); lb.add_argument("--output", default="results/harm_taxonomy_llm")
    dp = sub.add_parser("deepen"); dp.add_argument("--taxonomy_json", required=True)
    dp.add_argument("--model_path", default=None, help="LLM for sub-labeling; omit for keyword-only")
    dp.add_argument("--output", default="results/harm_taxonomy_deep3")
    args = ap.parse_args()
    {"build": cmd_build, "extract": cmd_extract, "gate": cmd_gate, "label": cmd_label, "deepen": cmd_deepen}[args.cmd](args)


if __name__ == "__main__":
    main()
