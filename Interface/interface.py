import gradio as gr
import pandas as pd

from analyse import (
    NutriScoreCalculator,
    ElectreTRI,
    calculer_supernutriscore
)

custom_css = """
<style>
* {
    margin: 0;
    padding: 0;
}

.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    height: auto;
    background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
}

.main {
    height: auto;
}

/* Section headers with icons */
.section-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 16px 20px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    letter-spacing: 0.3px;
}

/* Result cards */
.result-card {
    background: white;
    border: none;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    transition: all 0.3s ease;
}

.result-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.15);
}

.result-title {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 20px;
    font-weight: 800;
    margin-bottom: 20px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Enhanced Nutri-Score bubbles */
.nutriscore-container {
    display: flex;
    gap: 12px;
    margin: 20px 0;
    align-items: center;
    justify-content: center;
    padding: 20px;
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-radius: 16px;
}

.nutriscore-bubble {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
    font-weight: 900;
    color: white;
    transition: all 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
    opacity: 0.3;
    border: 3px solid transparent;
    cursor: pointer;
}

.nutriscore-bubble.active {
    width: 90px;
    height: 90px;
    font-size: 42px;
    opacity: 1;
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    transform: scale(1.1);
    border: 4px solid white;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1.1); }
    50% { transform: scale(1.15); }
}

.bubble-A { background: linear-gradient(135deg, #038141 0%, #02a357 100%); }
.bubble-B { background: linear-gradient(135deg, #85bb2f 0%, #a0d444 100%); }
.bubble-C { background: linear-gradient(135deg, #fecb02 0%, #ffdb4d 100%); color: #333; }
.bubble-D { background: linear-gradient(135deg, #ee8100 0%, #ff9922 100%); }
.bubble-E { background: linear-gradient(135deg, #e63e11 0%, #ff5533 100%); }

.score-large {
    font-size: 22px;
    font-weight: 800;
    margin: 16px 0;
    text-align: center;
    color: #1e293b;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Components with icons */
.composantes-container {
    display: flex;
    gap: 16px;
    margin-top: 20px;
}

.composante-box {
    flex: 1;
    border: none;
    border-radius: 12px;
    padding: 20px;
    font-size: 14px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}

.composante-box:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

.composante-negative {
    background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
}

.composante-positive {
    background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
}

.composante-title {
    font-weight: 800;
    margin-bottom: 12px;
    font-size: 16px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.composante-negative .composante-title {
    color: #dc2626;
}

.composante-positive .composante-title {
    color: #059669;
}

/* Styled ELECTRE parameters */
.params-box {
    background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
    padding: 20px;
    border-radius: 12px;
    margin: 16px 0;
    font-size: 14px;
    border: 2px solid #818cf8;
    box-shadow: 0 4px 15px rgba(129, 140, 248, 0.2);
}

.params-box strong {
    color: #4338ca;
}

/* ELECTRE assignments */
.affectation-container {
    display: flex;
    gap: 16px;
    margin-top: 20px;
}

.affectation-box {
    flex: 1;
    border: none;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    background: linear-gradient(135deg, #ffffff 0%, #f3f4f6 100%);
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
}

.affectation-box:hover {
    transform: translateY(-4px) scale(1.02);
    box-shadow: 0 8px 30px rgba(102, 126, 234, 0.3);
}

.affectation-title {
    color: #667eea;
    font-weight: 800;
    margin-bottom: 12px;
    font-size: 15px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.affectation-grade {
    font-size: 48px;
    font-weight: 900;
    margin: 12px 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Final SuperNutriScore - Premium version */
.super-final {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 32px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 10px 40px rgba(102, 126, 234, 0.5);
    position: relative;
}

.super-final::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    animation: rotate 20s linear infinite;
}

@keyframes rotate {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.super-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    text-transform: uppercase;
    letter-spacing: 2px;
    position: relative;
}

.super-category {
    font-size: 56px;
    font-weight: 900;
    margin: 20px 0;
    text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
    animation: glow 2s ease-in-out infinite;
    position: relative;
}

@keyframes glow {
    0%, 100% { text-shadow: 3px 3px 6px rgba(0,0,0,0.3); }
    50% { text-shadow: 0 0 20px rgba(255,255,255,0.8), 3px 3px 6px rgba(0,0,0,0.3); }
}

.super-description {
    font-size: 16px;
    margin-bottom: 16px;
    font-weight: 500;
    line-height: 1.6;
    position: relative;
}

.method-box {
    background: rgba(255, 255, 255, 0.2);
    padding: 16px;
    border-radius: 12px;
    text-align: left;
    margin-top: 16px;
    font-size: 13px;
    line-height: 1.6;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.3);
    position: relative;
}

/* Styled Gradio inputs */
.gr-form {
    gap: 14px !important;
}

label {
    font-size: 14px !important;
    margin-bottom: 6px !important;
    font-weight: 700 !important;
    color: #334155 !important;
    letter-spacing: 0.3px !important;
}

input[type="number"], input[type="text"], select {
    font-size: 15px !important;
    padding: 12px 16px !important;
    border-radius: 10px !important;
    border: 2px solid #e2e8f0 !important;
    transition: all 0.3s ease !important;
    background: white !important;
}

input[type="number"]:focus, input[type="text"]:focus, select:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.15) !important;
    transform: translateY(-2px);
}

input[type="number"]:hover, input[type="text"]:hover, select:hover {
    border-color: #cbd5e1 !important;
}

/* Styled action button */
button.primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    padding: 16px 32px !important;
    border-radius: 12px !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

button.primary:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6) !important;
}

button.primary:active {
    transform: translateY(-1px) !important;
}

/* Info text */
.gr-block.gr-box span.text-gray-500 {
    font-size: 12px !important;
    color: #64748b !important;
}

/* Accordion styles */
.gr-accordion {
    border: 2px solid #818cf8 !important;
    border-radius: 12px !important;
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important;
    margin-bottom: 16px !important;
}

/* Columns */
.gr-column {
    height: 100vh !important;
    display: flex !important;
    flex-direction: column !important;
    padding: 20px !important;
}

/* Scroll for the right column */
.result-column {
    padding-right: 10px !important;
}

.result-column::-webkit-scrollbar {
    width: 8px;
}

.result-column::-webkit-scrollbar-track {
    background: #e2e8f0;
    border-radius: 10px;
}

.result-column::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
}

.result-column::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
}

/* Entry animation */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.result-card, .super-final {
    animation: fadeInUp 0.6s ease-out;
}

/* Profile table styles */
.profile-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 16px;
    font-size: 13px;
    background: white;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.profile-table th {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 12px 8px;
    text-align: center;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-size: 12px;
}

.profile-table td {
    padding: 10px 8px;
    text-align: center;
    border-bottom: 1px solid #e2e8f0;
}

.profile-table tr:hover {
    background: #f8fafc;
}

.warning-box {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border: 2px solid #f59e0b;
    padding: 16px;
    border-radius: 12px;
    margin-top: 12px;
    font-size: 13px;
    color: #92400e;
    line-height: 1.6;
}
</style>
"""


def analyser_produit(nom, energie, sucres, ag_satures, sodium, proteines,
                     fibres, fruits_legumes, additifs, ecoscore, bio,
                     # Poids
                     poids_en, poids_sa, poids_su, poids_so, poids_pr, poids_fi, poids_fr, poids_ad,
                     # Seuil
                     seuil_majorite,
                     # Profils b1-b6 pour EN
                     b1_en, b2_en, b3_en, b4_en, b5_en, b6_en,
                     # Profils b1-b6 pour SA
                     b1_sa, b2_sa, b3_sa, b4_sa, b5_sa, b6_sa,
                     # Profils b1-b6 pour SU
                     b1_su, b2_su, b3_su, b4_su, b5_su, b6_su,
                     # Profils b1-b6 pour SO
                     b1_so, b2_so, b3_so, b4_so, b5_so, b6_so,
                     # Profils b1-b6 pour PR
                     b1_pr, b2_pr, b3_pr, b4_pr, b5_pr, b6_pr,
                     # Profils b1-b6 pour FI
                     b1_fi, b2_fi, b3_fi, b4_fi, b5_fi, b6_fi,
                     # Profils b1-b6 pour FR
                     b1_fr, b2_fr, b3_fr, b4_fr, b5_fr, b6_fr,
                     # Profils b1-b6 pour AD
                     b1_ad, b2_ad, b3_ad, b4_ad, b5_ad, b6_ad):

    # Handle None values for nutritional data
    energie = float(energie) if energie is not None else 0.0
    sucres = float(sucres) if sucres is not None else 0.0
    ag_satures = float(ag_satures) if ag_satures is not None else 0.0
    sodium = float(sodium) if sodium is not None else 0.0
    proteines = float(proteines) if proteines is not None else 0.0
    fibres = float(fibres) if fibres is not None else 0.0
    fruits_legumes = float(fruits_legumes) if fruits_legumes is not None else 0.0
    additifs = float(additifs) if additifs is not None else 0.0

    # Construire les poids personnalisés
    poids_custom = {
        'en': float(poids_en),
        'sa': float(poids_sa),
        'su': float(poids_su),
        'so': float(poids_so),
        'pr': float(poids_pr),
        'fi': float(poids_fi),
        'fr': float(poids_fr),
        'ad': float(poids_ad)
    }
    
    # Normaliser les poids
    somme_poids = sum(poids_custom.values())
    if somme_poids == 0:
        return "<div class='warning-box'>⚠️ <strong>Error:</strong> The sum of weights cannot be zero.</div>"
    
    poids_custom = {k: v/somme_poids for k, v in poids_custom.items()}
    
    # Construire les profils personnalisés
    profils_custom = pd.DataFrame({
        'en': [b6_en, b5_en, b4_en, b3_en, b2_en, b1_en],
        'sa': [b6_sa, b5_sa, b4_sa, b3_sa, b2_sa, b1_sa],
        'su': [b6_su, b5_su, b4_su, b3_su, b2_su, b1_su],
        'so': [b6_so, b5_so, b4_so, b3_so, b2_so, b1_so],
        'pr': [b6_pr, b5_pr, b4_pr, b3_pr, b2_pr, b1_pr],
        'fi': [b6_fi, b5_fi, b4_fi, b3_fi, b2_fi, b1_fi],
        'fr': [b6_fr, b5_fr, b4_fr, b3_fr, b2_fr, b1_fr],
        'ad': [b6_ad, b5_ad, b4_ad, b3_ad, b2_ad, b1_ad]
    }, index=['b6', 'b5', 'b4', 'b3', 'b2', 'b1'])

    calc = NutriScoreCalculator()

    # NutriScore
    nutri = calc.calculer_score(
        energie_kj=energie,
        sucres_g=sucres,
        ag_satures_g=ag_satures,
        sodium_mg=sodium,
        proteines_g=proteines,
        fibres_g=fibres,
        fruits_legumes_pct=fruits_legumes
    )

    nutri_cat = nutri["categorie"]
    nutri_score = nutri["score"]
    neg = nutri["composante_negative"]
    pos = nutri["composante_positive"]
    prot_ok = nutri["proteines_comptees"]

    # ELECTRE TRI avec paramètres personnalisés
    df = pd.DataFrame([{
        "en": energie,
        "sa": ag_satures,
        "su": sucres,
        "so": sodium,
        "pr": proteines,
        "fi": fibres,
        "fr": fruits_legumes,
        "ad": additifs
    }])

    electre = ElectreTRI(profils_custom, poids_custom, seuil_majorite=float(seuil_majorite))
    res_elec = electre.classifier_base_donnees(df).iloc[0]

    pess = res_elec["Categorie_Pessimiste"]
    opti = res_elec["Categorie_Optimiste"]

    # SuperNutriScore
    ligne = {
        "Categorie_Pessimiste": pess,
        "ecoscore_grade": ecoscore,
        "bio": 1 if bio == "Yes" else 0
    }

    super_cat, super_exp = calculer_supernutriscore(ligne)
    
    # Générer un tableau HTML des profils utilisés
    profils_html = """
    <table class="profile-table">
        <tr>
            <th>Profile</th>
            <th>EN (kJ)</th>
            <th>SA (g)</th>
            <th>SU (g)</th>
            <th>SO (mg)</th>
            <th>PR (g)</th>
            <th>FI (g)</th>
            <th>FR (%)</th>
            <th>AD</th>
        </tr>
    """
    
    for profile in ['b6', 'b5', 'b4', 'b3', 'b2', 'b1']:
        profils_html += f"""
        <tr>
            <td><strong>{profile}</strong></td>
            <td>{profils_custom.loc[profile, 'en']:.0f}</td>
            <td>{profils_custom.loc[profile, 'sa']:.1f}</td>
            <td>{profils_custom.loc[profile, 'su']:.0f}</td>
            <td>{profils_custom.loc[profile, 'so']:.0f}</td>
            <td>{profils_custom.loc[profile, 'pr']:.1f}</td>
            <td>{profils_custom.loc[profile, 'fi']:.1f}</td>
            <td>{profils_custom.loc[profile, 'fr']:.0f}</td>
            <td>{profils_custom.loc[profile, 'ad']:.0f}</td>
        </tr>
        """
    
    profils_html += "</table>"

    # HTML generation
    html = f"""
    <div class="result-card">
        <div class="result-title">🏆 Nutri-Score</div>
        
        <div class="nutriscore-container">
            <div class="nutriscore-bubble bubble-A {'active' if nutri_cat == 'A' else ''}">A</div>
            <div class="nutriscore-bubble bubble-B {'active' if nutri_cat == 'B' else ''}">B</div>
            <div class="nutriscore-bubble bubble-C {'active' if nutri_cat == 'C' else ''}">C</div>
            <div class="nutriscore-bubble bubble-D {'active' if nutri_cat == 'D' else ''}">D</div>
            <div class="nutriscore-bubble bubble-E {'active' if nutri_cat == 'E' else ''}">E</div>
        </div>
        
        <div class="score-large">Computed score: {nutri_score}</div>
        
        <div class="composantes-container">
            <div class="composante-box composante-negative">
                <div class="composante-title">⚠️ Negative Component</div>
                <div style="font-size: 28px; font-weight: 800; margin: 12px 0; color: #dc2626;">{neg} points</div>
                <div style="margin-top:12px; font-size:13px; color: #991b1b; line-height: 1.5;">
                    Energy • Sugars • Saturated fat • Sodium
                </div>
            </div>
            
            <div class="composante-box composante-positive">
                <div class="composante-title">✅ Positive Component</div>
                <div style="font-size: 28px; font-weight: 800; margin: 12px 0; color: #059669;">{pos} points</div>
                <div style="margin-top:12px; font-size:13px; color: #065f46; line-height: 1.5;">
                    Fiber • Protein{" (counted)" if prot_ok else " (not counted)"} • Fruits/Vegetables
                </div>
            </div>
        </div>
    </div>
    
    <div class="result-card">
        <div class="result-title">🎯 ELECTRE TRI</div>
        
        <div class="params-box">
            <strong>⚙️ Parameters used:</strong><br><br>
            <strong>Majority threshold (λ):</strong> {seuil_majorite}<br><br>
            <strong>Normalized criteria weights:</strong><br>
            EN: {poids_custom['en']*100:.1f}% • 
            SA: {poids_custom['sa']*100:.1f}% • 
            SU: {poids_custom['su']*100:.1f}% • 
            SO: {poids_custom['so']*100:.1f}%<br>
            PR: {poids_custom['pr']*100:.1f}% • 
            FI: {poids_custom['fi']*100:.1f}% • 
            FR: {poids_custom['fr']*100:.1f}% • 
            AD: {poids_custom['ad']*100:.1f}%
        </div>
        
        <div style="margin: 20px 0;">
            <strong style="color: #4338ca; font-size: 15px;">📊 Category Profiles:</strong>
            {profils_html}
        </div>
        
        <div class="affectation-container">
            <div class="affectation-box">
                <div class="affectation-title">📉 Pessimistic Assignment</div>
                <div class="affectation-grade">{pess}</div>
            </div>
            
            <div class="affectation-box">
                <div class="affectation-title">📈 Optimistic Assignment</div>
                <div class="affectation-grade">{opti}</div>
            </div>
        </div>
    </div>
    
    <div class="super-final">
        <div class="super-title">⭐ Final SuperNutri-Score</div>
        <div class="super-category">Category {super_cat}</div>
        <div class="super-description">{super_exp}</div>
        
        <div class="method-box">
            <strong>📋 Computation method:</strong><br><br>
            The SuperNutriScore combines ELECTRE TRI (pessimistic),
            the Eco-Score and the organic label to produce an overall evaluation.
            Adjustments are based on environmental impact and organic production practices.
        </div>
    </div>
    """

    return html


def interface():
    with gr.Blocks(title="SuperNutriScore") as demo:
        
        gr.HTML(custom_css)

        with gr.Row():
            # LEFT COLUMN - Inputs
            with gr.Column(scale=45, elem_classes="input-column"):
                gr.HTML("<div class='section-header'>🍎 Nutritional Data</div>")
                
                nom = gr.Textbox(label="Product name", placeholder="e.g. Organic tomato sauce", container=True)
                
                energie = gr.Number(
                    label="Energy (kJ/100g)",
                    value=0,
                    info="Energy per 100g in kilojoules",
                    container=True
                )
                
                sucres = gr.Number(label="Sugars (g/100g)", value=0, container=True)
                
                ag_satures = gr.Number(label="Saturated fat (g/100g)", value=0, container=True)
                
                sodium = gr.Number(
                    label="Sodium (mg/100g)",
                    value=0,
                    info="If salt is given in g, multiply by 400",
                    container=True
                )
                
                proteines = gr.Number(label="Protein (g/100g)", value=0, container=True)
                
                fibres = gr.Number(label="Fiber (g/100g)", value=0, container=True)
                
                fruits = gr.Number(label="Fruits / Vegetables / Nuts (%)", value=0, container=True)
                
                additifs = gr.Number(label="Number of additives", value=0, container=True)
                
                ecoscore = gr.Dropdown(
                    ["A","B","C","D","E"],
                    label="Eco-Score / Green-Score",
                    value="C",
                    container=True
                )
                
                bio = gr.Dropdown(
                    ["Yes","No"],
                    label="Organic label",
                    value="No",
                    container=True
                )
                
                gr.HTML("<div class='section-header' style='margin-top: 30px;'>⚙️ ELECTRE TRI Parameters</div>")
                
                with gr.Accordion("🎚️ Criteria Weights", open=False):
                    gr.Markdown("**Define weights for each criterion (will be normalized automatically)**")
                    
                    with gr.Row():
                        poids_en = gr.Number(label="Weight EN (Energy)", value=0.15, minimum=0, maximum=1, step=0.01)
                        poids_sa = gr.Number(label="Weight SA (Saturated fat)", value=0.15, minimum=0, maximum=1, step=0.01)
                    
                    with gr.Row():
                        poids_su = gr.Number(label="Weight SU (Sugars)", value=0.15, minimum=0, maximum=1, step=0.01)
                        poids_so = gr.Number(label="Weight SO (Sodium)", value=0.20, minimum=0, maximum=1, step=0.01)
                    
                    with gr.Row():
                        poids_pr = gr.Number(label="Weight PR (Protein)", value=0.10, minimum=0, maximum=1, step=0.01)
                        poids_fi = gr.Number(label="Weight FI (Fiber)", value=0.10, minimum=0, maximum=1, step=0.01)
                    
                    with gr.Row():
                        poids_fr = gr.Number(label="Weight FR (Fruits/Veg)", value=0.10, minimum=0, maximum=1, step=0.01)
                        poids_ad = gr.Number(label="Weight AD (Additives)", value=0.05, minimum=0, maximum=1, step=0.01)
                
                with gr.Accordion("🎯 Majority Threshold", open=False):
                    gr.Markdown("**Define the majority threshold (λ) for concordance**")
                    seuil_majorite = gr.Slider(
                        label="Threshold λ",
                        minimum=0.5,
                        maximum=1.0,
                        value=0.6,
                        step=0.05,
                        info="Value between 0.5 and 1.0"
                    )
                
                with gr.Accordion("📊 Category Profiles (b1 to b6)", open=False):
                    gr.Markdown("""
                    **Define the limit profiles for each category:**
                    - **b1**: Limit between A' and B'
                    - **b2**: Limit between B' and C'
                    - **b3**: Limit between C' and D'
                    - **b4**: Limit between D' and E'
                    - **b5**: Very poor quality threshold
                    - **b6**: Excellent quality threshold
                    
                    *For criteria to minimize (EN, SA, SU, SO, AD): lower values are better*  
                    *For criteria to maximize (PR, FI, FR): higher values are better*
                    """)
                    
                    # Profile b1
                    gr.Markdown("### **Profile b1** (A'/B' limit)")
                    with gr.Row():
                        b1_en = gr.Number(label="b1 - EN (kJ)", value=500)
                        b1_sa = gr.Number(label="b1 - SA (g)", value=0.5)
                        b1_su = gr.Number(label="b1 - SU (g)", value=3)
                        b1_so = gr.Number(label="b1 - SO (mg)", value=200)
                    with gr.Row():
                        b1_pr = gr.Number(label="b1 - PR (g)", value=3)
                        b1_fi = gr.Number(label="b1 - FI (g)", value=2.5)
                        b1_fr = gr.Number(label="b1 - FR (%)", value=80)
                        b1_ad = gr.Number(label="b1 - AD", value=3)
                    
                    # Profile b2
                    gr.Markdown("### **Profile b2** (B'/C' limit)")
                    with gr.Row():
                        b2_en = gr.Number(label="b2 - EN (kJ)", value=800)
                        b2_sa = gr.Number(label="b2 - SA (g)", value=1.5)
                        b2_su = gr.Number(label="b2 - SU (g)", value=6)
                        b2_so = gr.Number(label="b2 - SO (mg)", value=400)
                    with gr.Row():
                        b2_pr = gr.Number(label="b2 - PR (g)", value=2)
                        b2_fi = gr.Number(label="b2 - FI (g)", value=1.5)
                        b2_fr = gr.Number(label="b2 - FR (%)", value=60)
                        b2_ad = gr.Number(label="b2 - AD", value=6)
                    
                    # Profile b3
                    gr.Markdown("### **Profile b3** (C'/D' limit)")
                    with gr.Row():
                        b3_en = gr.Number(label="b3 - EN (kJ)", value=1200)
                        b3_sa = gr.Number(label="b3 - SA (g)", value=3.0)
                        b3_su = gr.Number(label="b3 - SU (g)", value=10)
                        b3_so = gr.Number(label="b3 - SO (mg)", value=600)
                    with gr.Row():
                        b3_pr = gr.Number(label="b3 - PR (g)", value=1)
                        b3_fi = gr.Number(label="b3 - FI (g)", value=0.8)
                        b3_fr = gr.Number(label="b3 - FR (%)", value=40)
                        b3_ad = gr.Number(label="b3 - AD", value=10)
                    
                    # Profile b4
                    gr.Markdown("### **Profile b4** (D'/E' limit)")
                    with gr.Row():
                        b4_en = gr.Number(label="b4 - EN (kJ)", value=1800)
                        b4_sa = gr.Number(label="b4 - SA (g)", value=5.0)
                        b4_su = gr.Number(label="b4 - SU (g)", value=15)
                        b4_so = gr.Number(label="b4 - SO (mg)", value=900)
                    with gr.Row():
                        b4_pr = gr.Number(label="b4 - PR (g)", value=0.5)
                        b4_fi = gr.Number(label="b4 - FI (g)", value=0.3)
                        b4_fr = gr.Number(label="b4 - FR (%)", value=20)
                        b4_ad = gr.Number(label="b4 - AD", value=15)
                    
                    # Profile b5
                    gr.Markdown("### **Profile b5** (Very poor quality)")
                    with gr.Row():
                        b5_en = gr.Number(label="b5 - EN (kJ)", value=4000)
                        b5_sa = gr.Number(label="b5 - SA (g)", value=20)
                        b5_su = gr.Number(label="b5 - SU (g)", value=50)
                        b5_so = gr.Number(label="b5 - SO (mg)", value=2000)
                    with gr.Row():
                        b5_pr = gr.Number(label="b5 - PR (g)", value=0)
                        b5_fi = gr.Number(label="b5 - FI (g)", value=0)
                        b5_fr = gr.Number(label="b5 - FR (%)", value=0)
                        b5_ad = gr.Number(label="b5 - AD", value=30)
                    
                    # Profile b6
                    gr.Markdown("### **Profile b6** (Excellent quality)")
                    with gr.Row():
                        b6_en = gr.Number(label="b6 - EN (kJ)", value=0)
                        b6_sa = gr.Number(label="b6 - SA (g)", value=0)
                        b6_su = gr.Number(label="b6 - SU (g)", value=0)
                        b6_so = gr.Number(label="b6 - SO (mg)", value=0)
                    with gr.Row():
                        b6_pr = gr.Number(label="b6 - PR (g)", value=10)
                        b6_fi = gr.Number(label="b6 - FI (g)", value=10)
                        b6_fr = gr.Number(label="b6 - FR (%)", value=100)
                        b6_ad = gr.Number(label="b6 - AD", value=0)

                btn = gr.Button("🚀 Run analysis", variant="primary", size="lg")

            # RIGHT COLUMN - Results
            with gr.Column(scale=55, elem_classes="result-column"):
                gr.HTML("<div class='section-header'>📊 Analysis Results</div>")
                resultat = gr.HTML()

        btn.click(
            analyser_produit,
            inputs=[
                # Nutritional data
                nom, energie, sucres, ag_satures, sodium,
                proteines, fibres, fruits, additifs, ecoscore, bio,
                # Weights
                poids_en, poids_sa, poids_su, poids_so, poids_pr, poids_fi, poids_fr, poids_ad,
                # Threshold
                seuil_majorite,
                # Profiles
                b1_en, b2_en, b3_en, b4_en, b5_en, b6_en,
                b1_sa, b2_sa, b3_sa, b4_sa, b5_sa, b6_sa,
                b1_su, b2_su, b3_su, b4_su, b5_su, b6_su,
                b1_so, b2_so, b3_so, b4_so, b5_so, b6_so,
                b1_pr, b2_pr, b3_pr, b4_pr, b5_pr, b6_pr,
                b1_fi, b2_fi, b3_fi, b4_fi, b5_fi, b6_fi,
                b1_fr, b2_fr, b3_fr, b4_fr, b5_fr, b6_fr,
                b1_ad, b2_ad, b3_ad, b4_ad, b5_ad, b6_ad
            ],
            outputs=resultat
        )

    return demo


app = interface()
app.launch(inbrowser=True)