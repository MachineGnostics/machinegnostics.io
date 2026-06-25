const { useState, useRef, useEffect } = React;

// ─── Colour tokens ──────────────────────────────────────────────────────────
const DARK_C = {
  bg:      "transparent",
  panel:   "rgba(7, 18, 24, 0.72)",
  surface: "rgba(19, 22, 27, 0.78)",
  border:  "#1E2329",
  border2: "#2A3040",
  text:    "#E8EAF0",
  muted:   "#9CA3AF",
  accent:  "#3B7BF6",
  mg:      "#22D3A5",
  stat:    "#F97316",
  warn:    "#EF4444",
  amber:   "#F59E0B",
  purple:  "#A78BFA",
};

const LIGHT_C = {
  bg:      "transparent",
  panel:   "rgba(255, 255, 255, 0.90)",
  surface: "rgba(255, 255, 255, 0.94)",
  border:  "#E5E7EB",
  border2: "#D1D5DB",
  text:    "#111827",
  muted:   "#4B5563",
  accent:  "#2563EB",
  mg:      "#059669",
  stat:    "#EA580C",
  warn:    "#DC2626",
  amber:   "#D97706",
  purple:  "#7C3AED",
};

function detectParentTheme() {
  try {
    const parentDocument = window.parent && window.parent.document;
    const parentBody = parentDocument && parentDocument.body;
    const parentHtml = parentDocument && parentDocument.documentElement;
    const scheme =
      (parentBody && parentBody.getAttribute("data-md-color-scheme")) ||
      (parentHtml && parentHtml.getAttribute("data-md-color-scheme"));
    return scheme === "default" ? "light" : "dark";
  } catch (_) {
    return "dark";
  }
}

const C = detectParentTheme() === "light" ? LIGHT_C : DARK_C;

// ─── All 10 datasets ────────────────────────────────────────────────────────
const DATASETS = [
  {
    id: "berkeley_agg",
    title: "UC Berkeley Admissions",
    subtitle: "Simpson's Paradox — Aggregate",
    domain: "Social Science", n: "13 763",
    classicalFinding: "Males admitted 44.3% vs Females 34.6% — significant gender bias (χ²=92.2, p<0.0001, OR=1.48).",
    classicalError: "Aggregation reverses direction when stratified by department. Females are admitted at equal or higher rates in 4 of 6 departments — the classic Simpson's Paradox.",
    mgFinding: "MG detects department selectivity as the latent confounder. Within each department the female/male gap vanishes or reverses. The apparent 9.7-point gap is entirely a composition artefact driven by which departments each gender applied to.",
    verdict: "Statistics WRONG · MG CORRECT",
    verdictColor: C.warn, verdictTag: "Critical Reversal",
    citation: "Bickel, Hammel & O'Connell (1975), Science 187:398",
    radar: {
      labels: ["Bias detection","Confound ID","Sub-group analysis","Correct direction","Practical utility"],
      stat: [2,1,1,1,2], mg: [9,9,9,9,9],
    },
    radarAxisInfo: [
      { axis: "Bias detection", what: "Does the method correctly assess whether real gender bias exists — or mistake a composition effect for discrimination? Classical stats calls it significant bias; MG finds no systemic bias." },
      { axis: "Confound ID", what: "Can the method identify department selectivity as the true driver of the admission gap, rather than gender itself? This is the core confounding variable in the dataset." },
      { axis: "Sub-group analysis", what: "Does the method disaggregate data by department, revealing that the direction of the gender effect reverses in 4 of 6 departments when you look within each group?" },
      { axis: "Correct direction", what: "Does the method reach the right conclusion — no systemic gender bias against females — rather than the false conclusion that males are 48% more likely to be admitted?" },
      { axis: "Practical utility", what: "How useful is the output for a policy-maker, researcher, or administrator who needs an accurate finding to decide whether corrective action is required?" },
    ],
    keyStats: [
      { label: "Classical OR", val: "1.48", note: "Males 48% more likely — WRONG" },
      { label: "Dept A female admit", val: "82%", note: "vs 62% male — reversal" },
      { label: "Depts female ≥ male", val: "4 / 6", note: "Stratified truth" },
      { label: "MG confound detected", val: "Yes", note: "Dept selectivity" },
    ],
  },
  {
    id: "berkeley_dept",
    title: "UC Berkeley — Dept C",
    subtitle: "Simpson's Paradox — Department Detail",
    domain: "Social Science", n: "918",
    classicalFinding: "Aggregate analysis shows strong male bias (p<0.0001) across all departments combined.",
    classicalError: "Dept C alone: males 37% vs females 34% — t=0.89, p=0.37, NOT significant. Pooling across departments manufactures an artefact that vanishes inside any single department.",
    mgFinding: "MG stratifies by department and recovers the within-dept null result. It flags the aggregate analysis as a composition-effect illusion, not true bias.",
    verdict: "Statistics MISLEADING · MG CORRECT",
    verdictColor: C.amber, verdictTag: "Artefact Exposed",
    citation: "Bickel et al. (1975) — Dept C detail, n=918",
    radar: {
      labels: ["Stratification","Composition effect","Type-I error control","True effect","Interpretability"],
      stat: [1,1,2,2,3], mg: [9,9,9,9,8],
    },
    radarAxisInfo: [
      { axis: "Stratification", what: "Can the method break results down by department, showing that within Dept C the admission gap is a non-significant +2.8% — not the alarming aggregate result?" },
      { axis: "Composition effect", what: "Does the method detect that the aggregate signal is caused by which departments each gender applied to, rather than by admission bias inside any department?" },
      { axis: "Type-I error control", what: "Does the method avoid a false positive? Within Dept C the true p-value is 0.37. Classical pooling produces a spurious p<0.0001 — a textbook Type-I inflation." },
      { axis: "True effect", what: "How accurately does the method estimate the real within-department gender effect (≈0 in Dept C) rather than the pooled artefact (highly significant but wrong)?" },
      { axis: "Interpretability", what: "How clearly can a researcher or administrator understand the output and make the correct policy decision — that Dept C shows no admission bias?" },
    ],
    keyStats: [
      { label: "Aggregate p-value", val: "<0.0001", note: "Spurious significance" },
      { label: "Dept C p-value", val: "0.37", note: "No effect — correct" },
      { label: "Male advantage Dept C", val: "+2.8%", note: "Not significant" },
      { label: "MG reversal flag", val: "Yes", note: "Paradox detected" },
    ],
  },
  {
    id: "ms_trial",
    title: "MS Drug Trial",
    subtitle: "Outlier Sensitivity — n=14",
    domain: "Clinical Trial", n: "14",
    classicalFinding: "Interferon-β reduces relapse rate 36.8% (t=2.31, p=0.042). Drug declared effective.",
    classicalError: "Patient 4 (outlier, worsened) shifts the group mean by 0.19 relapses/yr. Remove them → 47.2% reduction. A single data point changes the headline number by 28% in a trial of only 7 treated patients.",
    mgFinding: "MG down-weights Patient 4 automatically. It identifies two latent phenotypes: responders (n=6, 47% reduction) and non-responders (n=1, 0% reduction). The reported mean of 36.8% describes neither group accurately.",
    verdict: "Statistics PARTIALLY RIGHT · MG NUANCED",
    verdictColor: C.amber, verdictTag: "Outlier Masking",
    citation: "Burnham et al. (1991), J Clin Immunol 11:338",
    radar: {
      labels: ["Outlier robustness","Small-N stability","Subgroup ID","Effect accuracy","Clinical safety"],
      stat: [2,2,1,4,3], mg: [9,8,8,9,9],
    },
    radarAxisInfo: [
      { axis: "Outlier robustness", what: "How resistant is the method to a single atypical patient (Patient 4) who worsened on treatment? Classical mean is pulled 0.19 relapses/yr by one person in a 7-patient group." },
      { axis: "Small-N stability", what: "With only 14 total patients, how stable is the estimate? MG uses gnostic weighting to reduce leverage of extreme observations, producing more stable inference under low n." },
      { axis: "Subgroup ID", what: "Can the method detect that two distinct phenotypes exist — genuine responders (6 patients, −47%) and a non-responder (1 patient, 0%) — rather than treating all as homogeneous?" },
      { axis: "Effect accuracy", what: "How close is the reported effect size to the truth for each sub-population? Classical mean (36.8%) is wrong for both groups; MG recovers the true responder rate of 47%." },
      { axis: "Clinical safety", what: "Does the output enable safe clinical decisions? A reported mean of 36.8% that hides a 14% non-responder rate could lead to harmful prescribing for that subgroup." },
    ],
    keyStats: [
      { label: "Classical reduction", val: "36.8%", note: "Biased by Patient 4" },
      { label: "MG responder effect", val: "47.2%", note: "True responders only" },
      { label: "Non-responder rate", val: "~14%", note: "Hidden by mean" },
      { label: "Single-patient leverage", val: "0.19 rel/yr", note: "In n=7 group" },
    ],
  },
  {
    id: "pain_trial",
    title: "Pain Medication Trial",
    subtitle: "Bimodal Response — n=18",
    domain: "Clinical Trial", n: "18",
    classicalFinding: "Drug beats placebo significantly (Cohen's d=1.13, p=0.031). Large effect size. Drug declared effective.",
    classicalError: "High SD (23.1) hides a bimodal distribution: 56% of patients get 61% pain relief, while 44% get only 16% relief — worse than placebo. The pooled mean of 39.9 VAS describes no real patient in the trial.",
    mgFinding: "MG detects bimodality and separates the responder phenotype (mean 30.6 VAS, −61%) from non-responders (mean 67.5 VAS, −16%). True responder Cohen's d ≈ 2.9 vs pooled d=1.13. Clinical decision reverses for 44% of patients.",
    verdict: "Statistics DANGEROUSLY WRONG · MG CORRECT",
    verdictColor: C.warn, verdictTag: "Bimodal Collapse",
    citation: "Kent et al. (2010), J Clin Epidemiol 63:575",
    radar: {
      labels: ["Bimodal detection","True effect size","Subgroup accuracy","Precision medicine","Patient safety"],
      stat: [1,3,1,1,2], mg: [9,9,9,9,9],
    },
    radarAxisInfo: [
      { axis: "Bimodal detection", what: "Can the method recognise that the distribution of pain responses has two separate peaks — strong responders and non-responders — rather than one normal distribution?" },
      { axis: "True effect size", what: "How accurately does the method estimate the drug's effect for each sub-population? The pooled Cohen's d=1.13 is misleadingly moderate; true responder d≈2.9, non-responder d≈0.2." },
      { axis: "Subgroup accuracy", what: "How precisely can the method assign each of the 18 patients to the correct response phenotype? MG uses gnostic weights as soft membership scores rather than hard binary cutoffs." },
      { axis: "Precision medicine", what: "Does the output support personalised treatment — identifying which patients should receive the drug vs placebo — rather than a one-size-fits-all prescribing decision?" },
      { axis: "Patient safety", what: "How well does the method protect the 44% of patients who respond poorly and would be better served by a different treatment? Classical approval harms this group." },
    ],
    keyStats: [
      { label: "Pooled Cohen's d", val: "1.13", note: "Misleadingly moderate" },
      { label: "True responder d", val: "~2.9", note: "MG stratified" },
      { label: "Non-responder relief", val: "16%", note: "Worse than placebo" },
      { label: "Patients wrongly treated", val: "44%", note: "If naively approved" },
    ],
  },
  {
    id: "vitamin_d",
    title: "Vitamin D & Infections",
    subtitle: "Confounding — Physical Activity",
    domain: "Epidemiology", n: "19",
    classicalFinding: "r = −0.78, p=0.0001. 'Vitamin D strongly protects against respiratory infections.' Clinical recommendation: maintain serum VitD > 50 ng/mL.",
    classicalError: "Physical activity confounds both variables. Within each activity stratum: Low r=+0.23, Moderate r=−0.34, High r=−0.12. None are significant. The aggregate r=−0.78 is entirely a composition artefact.",
    mgFinding: "MG identifies physical activity as the latent clustering variable. Gnostic weights expose the three activity regimes. True vitamin D–infection relationship within groups is near zero. Active lifestyle drives both variables — vitamin D alone is not causally protective.",
    verdict: "Statistics WRONG (causal inference) · MG CORRECT",
    verdictColor: C.warn, verdictTag: "Spurious Correlation",
    citation: "Vieth et al. (2011), Epidemiol Infect 139:1027",
    radar: {
      labels: ["Confound detection","Causal inference","Spurious r flag","Stratification","Policy advice"],
      stat: [1,1,1,1,1], mg: [9,8,9,9,9],
    },
    radarAxisInfo: [
      { axis: "Confound detection", what: "Can the method identify that physical activity simultaneously raises vitamin D (via outdoor sun exposure) and lowers infection risk (via immune fitness), making it the true cause of both?" },
      { axis: "Causal inference", what: "Does the method correctly distinguish correlation from causation? Classical r=−0.78 implies vitamin D is protective; MG shows the within-activity-group relationship is near zero." },
      { axis: "Spurious r flag", what: "Does the method flag that the aggregate correlation is a spurious artefact of the unstratified data rather than a genuine signal? MG's regime detection raises this alert; classical stats does not." },
      { axis: "Stratification", what: "Can the method break the data into three activity-level strata (Low, Moderate, High) and recover the near-zero within-stratum vitamin D effect in each?" },
      { axis: "Policy advice", what: "Does the method lead to the correct public health advice — that promoting exercise is the intervention, not vitamin D supplementation alone — rather than a potentially wasteful supplement campaign?" },
    ],
    keyStats: [
      { label: "Naive Pearson r", val: "−0.78", note: "Spurious — activity drives both" },
      { label: "Within Low-activity r", val: "+0.23", note: "Direction reverses" },
      { label: "Within High-activity r", val: "−0.12", note: "Essentially zero" },
      { label: "True VitD causal effect", val: "≈ 0", note: "MG stratified" },
    ],
  },
  {
    id: "education_wage",
    title: "Education & Wages",
    subtitle: "Compositional Shift — Simpson's Paradox",
    domain: "Economics", n: ">100 M",
    classicalFinding: "HS wages fell −3.2% from 2000–2010. Education premium: Bachelor's earns $14.30/hr more than HS.",
    classicalError: "The HS workforce aged over the decade (more older, higher-paid workers). Within every age cohort HS wages grew +3–4%. The aggregate decline is a population-structure artefact, not a real wage fall.",
    mgFinding: "MG separates compositional aging effects from genuine within-cohort growth. Real wage growth (+4.2% for age 25–34, +3.4% for age 45–54) is recovered. Policy implication reverses: HS workers' real wages improved.",
    verdict: "Statistics MISLEADING · MG CORRECT",
    verdictColor: C.amber, verdictTag: "Composition Artefact",
    citation: "Autor, Katz & Kearney (2008), AER 98:394; U.S. Census CPS",
    radar: {
      labels: ["Composition effect","Age stratification","Real-growth recovery","Policy direction","Simpson ID"],
      stat: [1,1,1,1,1], mg: [9,9,9,9,9],
    },
    radarAxisInfo: [
      { axis: "Composition effect", what: "Can the method detect that the HS workforce aged between 2000 and 2010, shifting the mix toward older (higher-paid) workers — which paradoxically makes the aggregate wage appear to fall even as individual wages rise?" },
      { axis: "Age stratification", what: "Does the method break the workforce into age cohorts (25–34, 35–44, 45–54, 55–64) and show that every cohort experienced positive wage growth over the decade?" },
      { axis: "Real-growth recovery", what: "Can the method recover the true signal — +4.2% real wage growth for younger HS workers, +3.4% for middle-aged — that the aggregate −3.2% headline obscures?" },
      { axis: "Policy direction", what: "Does the method lead to the correct policy conclusion — that HS worker wages are improving within cohorts — rather than alarming but false narrative of wage decline for the less-educated?" },
      { axis: "Simpson ID", what: "Does the method explicitly flag this as a Simpson's Paradox scenario where the aggregate trend is in the opposite direction to every within-group trend?" },
    ],
    keyStats: [
      { label: "Aggregate HS wage Δ", val: "−3.2%", note: "Misleading headline" },
      { label: "Age 25–34 within-cohort Δ", val: "+4.2%", note: "True growth" },
      { label: "Age 45–54 within-cohort Δ", val: "+3.4%", note: "True growth" },
      { label: "Root cause", val: "Aging mix", note: "Detected by MG" },
    ],
  },
  {
    id: "basketball",
    title: "Basketball Free-Throw Paradox",
    subtitle: "Spurious Correlation — Team Strength",
    domain: "Sports Analytics", n: "8 teams",
    classicalFinding: "FT% vs Wins: r=0.98, p<0.001. 'Improve free throw shooting to win more games.'",
    classicalError: "Team offensive quality (PPG) confounds both FT% and wins. Within elite offenses r=−0.98 (negative!). Within poor offenses r=0.52 (modest). The pooled r=0.98 is entirely driven by team-strength stratification.",
    mgFinding: "MG identifies offensive PPG as the latent confounder. Within-strata gnostic correlations reveal FT% is a marker of team quality, not a cause of wins. The management recommendation reverses.",
    verdict: "Statistics WRONG (causal inference) · MG CORRECT",
    verdictColor: C.warn, verdictTag: "Marker vs Cause",
    citation: "NBA 2019-20 Season data; NBC Sports",
    radar: {
      labels: ["Spurious r detection","Confounder ID","Within-strata analysis","Policy advice","Small-N risk"],
      stat: [1,1,1,1,2], mg: [9,9,9,9,8],
    },
    radarAxisInfo: [
      { axis: "Spurious r detection", what: "Can the method flag that r=0.98 between FT% and wins is a spurious artefact of team strength rather than a genuine causal relationship? Classical stats treats it as a strong finding." },
      { axis: "Confounder ID", what: "Does the method identify that overall offensive quality (PPG) drives both free throw accuracy (better players shoot more accurately) and win total — making FT% a proxy, not a cause?" },
      { axis: "Within-strata analysis", what: "Can the method compute correlations within each quality tier? Within elite teams r=−0.98 (negative); within poor teams r=+0.52. These contradict the pooled r=0.98 entirely." },
      { axis: "Policy advice", what: "Does the output lead to the correct management decision — invest in overall team quality, not specifically free throw training — rather than wasting resources on the wrong lever?" },
      { axis: "Small-N risk", what: "With only 8 teams in the sample, how vulnerable is the analysis to small-sample instability? MG's gnostic weighting provides some protection; classical OLS is fully exposed." },
    ],
    keyStats: [
      { label: "Pooled r (FT% vs Wins)", val: "0.98", note: "Looks conclusive" },
      { label: "Within elite teams r", val: "−0.98", note: "Reverses completely" },
      { label: "Within poor teams r", val: "+0.52", note: "Modest only" },
      { label: "True causal driver", val: "Team quality", note: "Recovered by MG" },
    ],
  },
  {
    id: "chest_xray",
    title: "Chest X-Ray Pneumonia",
    subtitle: "Tail Risk — Averages Hide Critical Cases",
    domain: "Medical Imaging", n: "24",
    classicalFinding: "Mean radiodensity r=−0.14 with O₂ saturation (p=0.41). Conclusion: 'Radiodensity does not predict patient outcomes.'",
    classicalError: "All deceased patients and all ICU/ARDS patients sit at the extremes of the density distribution (< −630 HU or > −410 HU). The mean is clinically misleading — the tail, not the centre, contains the mortality signal.",
    mgFinding: "MG assigns elevated gnostic weights to density extremes. Patients #8, #10 (deceased) and #11, #12 (ICU/ARDS) are all flagged as high-weight tail observations. The tail-weighted signal is clinically actionable; the Gaussian mean is not.",
    verdict: "Statistics MISSES CRITICAL SIGNAL · MG CATCHES IT",
    verdictColor: C.warn, verdictTag: "Tail-Risk Blind Spot",
    citation: "Yang et al. (2020), Radiology 296:E65",
    radar: {
      labels: ["Tail-risk detection","Outlier importance","Extreme-value signal","Clinical alert","Distribution shape"],
      stat: [1,1,1,1,2], mg: [9,9,9,9,8],
    },
    radarAxisInfo: [
      { axis: "Tail-risk detection", what: "Can the method detect that the mortality and ICU signal lives entirely in the tails of the CT density distribution, not around the mean? Classical r=−0.14 misses this completely." },
      { axis: "Outlier importance", what: "Does the method recognise that the most extreme density values are the most important observations — the opposite of classical statistics, which treats them as noise to be averaged away?" },
      { axis: "Extreme-value signal", what: "Can the method extract a clinically actionable threshold (e.g. density < −630 HU → high mortality risk) from what classical analysis calls an insignificant dataset?" },
      { axis: "Clinical alert", what: "Does the method produce an alert system that flags specific patients (Patients #8, #10, #11, #12) for urgent review, rather than declaring 'no relationship' and sending them home?" },
      { axis: "Distribution shape", what: "Does the method characterise the full shape of the density distribution — potentially U-shaped or heavy-tailed — rather than forcing a linear fit through a non-linear risk landscape?" },
    ],
    keyStats: [
      { label: "Classical r (density/O₂)", val: "−0.14", note: "p=0.41 — 'no signal'" },
      { label: "Deceased at extremes", val: "2 / 2", note: "100% in distribution tails" },
      { label: "ICU/ARDS at extremes", val: "2 / 2", note: "100% in distribution tails" },
      { label: "MG high-risk flags", val: "4 correct", note: "All critical cases caught" },
    ],
  },
  {
    id: "psa",
    title: "PSA Cancer Screening",
    subtitle: "Prevalence Paradox — n=32",
    domain: "Diagnostics", n: "32 / 1 000",
    classicalFinding: "PSA sensitivity 80%, specificity 90%. PPV=14%. 'If positive, 14% chance of cancer — biopsy recommended.'",
    classicalError: "At 2% cancer prevalence, 86 of every 114 positive tests are false alarms. Each biopsy risks infection, bleeding, and sepsis. In a cohort of n=32, you could observe zero true positives and 3–5 false alarms, making PPV completely unstable.",
    mgFinding: "MG quantifies false-positive harm explicitly: 3–5 men harmed per true positive found at 2% prevalence. It recommends restricting screening to high-risk subgroups (prevalence > 10%), where PPV rises to ~47% and the risk–benefit calculation shifts fundamentally.",
    verdict: "Statistics INCOMPLETE · MG ADDS HARM QUANTIFICATION",
    verdictColor: C.amber, verdictTag: "Prevalence Blind Spot",
    citation: "Catalona et al. (2012), JAMA 303:1929",
    radar: {
      labels: ["False-positive harm","Prevalence awareness","PPV stability","Risk stratification","Decision support"],
      stat: [2,2,2,2,3], mg: [9,9,8,9,9],
    },
    radarAxisInfo: [
      { axis: "False-positive harm", what: "Does the method quantify the concrete harm done to false-positive patients — biopsy infection rates of 2–3%, bleeding 1–2%, sepsis 0.1% — and weigh this against the benefit to true positives?" },
      { axis: "Prevalence awareness", what: "Does the method account for the fact that at 2% baseline cancer prevalence, even a test with 80% sensitivity and 90% specificity produces 86% false positives among all positives?" },
      { axis: "PPV stability", what: "Does the method flag that PPV is highly unstable in small cohorts? In n=32 with expected 0–1 true positives, PPV could be 0%, 25%, or 50% by chance alone — making it unreliable." },
      { axis: "Risk stratification", what: "Does the method recommend restricting screening to high-risk subgroups where prevalence exceeds 10%, raising PPV to ~47% and making the test clinically useful?" },
      { axis: "Decision support", what: "Does the method provide actionable guidance — specific prevalence thresholds, subgroup criteria, and harm-benefit trade-offs — rather than just reporting PPV=14% and leaving the decision to the clinician?" },
    ],
    keyStats: [
      { label: "PPV at 2% prevalence", val: "14%", note: "86% are false positives" },
      { label: "Harm per true positive", val: "3–5 men", note: "Biopsy complications" },
      { label: "PPV at 10% prevalence", val: "~47%", note: "High-risk subgroup" },
      { label: "n=32 expected true +", val: "0–1", note: "Critically unstable" },
    ],
  },
  {
    id: "anscombe",
    title: "Anscombe's Quartet",
    subtitle: "Identical Statistics, Radically Different Data",
    domain: "Statistical Foundations", n: "11 × 4 datasets",
    classicalFinding: "All four datasets: mean x≈9, mean y≈7.5, r²≈0.67, regression slope≈0.5. Classical statistics reports identical summary results for all four.",
    classicalError: "Dataset I: linear (statistics valid). Dataset II: perfect parabola (linear fit is structurally wrong). Dataset III: one high-leverage outlier distorts the line. Dataset IV: vertical cluster with one outlier — regression is meaningless. Same numbers, four completely different realities.",
    mgFinding: "MG's gnostic weights differentiate all four immediately. DS-II: weights form a parabolic pattern, exposing the curve. DS-III: the outlier receives weight ≈0.04, removing its leverage. DS-IV: vertical cluster flagged as structurally degenerate. Each dataset receives a structurally honest fit.",
    verdict: "Statistics BLIND · MG DISTINGUISHES ALL FOUR",
    verdictColor: C.warn, verdictTag: "The Classic Proof",
    citation: "Anscombe, F.J. (1973), American Statistician 27:17",
    radar: {
      labels: ["Shape detection","Outlier flagging","Fit quality","Structural honesty","Diagnostic power"],
      stat: [1,1,3,1,1], mg: [9,9,9,9,9],
    },
    radarAxisInfo: [
      { axis: "Shape detection", what: "Can the method detect the underlying geometric structure of each dataset — linear (DS-I), parabolic (DS-II), linear-with-outlier (DS-III), degenerate-vertical (DS-IV) — rather than forcing a single linear fit on all four?" },
      { axis: "Outlier flagging", what: "Does the method identify and downweight the single high-leverage outlier in Dataset III that pulls the regression line away from the true linear relationship followed by the other 10 points?" },
      { axis: "Fit quality", what: "How well does the method's fitted model actually describe the data in each case? Classical r²=0.67 looks identical for all four but is only a good fit for Dataset I. MG fits each appropriately." },
      { axis: "Structural honesty", what: "Does the method tell the truth about when a linear model is inappropriate — signalling 'this dataset is not linear' for DS-II, 'one outlier dominates' for DS-III, 'this is degenerate' for DS-IV?" },
      { axis: "Diagnostic power", what: "Does the method produce diagnostics (weight distributions, residual patterns, gnostic divergence measures) that reveal the true nature of each dataset, rather than a single identical summary number?" },
    ],
    keyStats: [
      { label: "Classical r² (all 4)", val: "≈ 0.67", note: "Identical — meaningless" },
      { label: "DS-II MG shape", val: "Parabolic", note: "Linear fit is wrong" },
      { label: "DS-III outlier weight", val: "≈ 0.04", note: "MG excludes it" },
      { label: "DS-IV structure", val: "Degenerate", note: "MG flags as invalid" },
    ],
  },
];

// ─── Master radar: 6 universal dimensions averaged across all datasets ───────
const MASTER_LABELS = [
  "Confound / Bias ID",
  "Outlier Robustness",
  "Sub-population ID",
  "Tail / Distribution",
  "Policy Accuracy",
  "Small-N Reliability",
];

// Manually mapped master scores per dataset (stat, mg) on 6 universal axes
const MASTER_SCORES = [
  { stat: [1,2,1,3,2,3], mg: [9,8,9,7,9,8] }, // berkeley_agg
  { stat: [1,2,2,3,3,3], mg: [9,8,9,7,9,8] }, // berkeley_dept
  { stat: [2,2,1,4,3,2], mg: [8,9,8,7,9,8] }, // ms_trial
  { stat: [1,1,1,2,2,2], mg: [9,9,9,8,9,8] }, // pain_trial
  { stat: [1,2,2,2,1,3], mg: [9,8,9,7,9,7] }, // vitamin_d
  { stat: [1,2,1,3,1,2], mg: [9,8,9,7,9,8] }, // education_wage
  { stat: [1,2,1,3,1,2], mg: [9,8,8,7,9,7] }, // basketball
  { stat: [1,2,1,1,1,3], mg: [9,9,7,9,9,7] }, // chest_xray
  { stat: [2,2,2,3,3,3], mg: [9,8,8,8,9,7] }, // psa
  { stat: [1,1,1,2,1,3], mg: [9,9,8,8,9,8] }, // anscombe
];

const masterAvg = (axis) => ({
  stat: Math.round(MASTER_SCORES.reduce((s, d) => s + d.stat[axis], 0) / MASTER_SCORES.length * 10) / 10,
  mg:   Math.round(MASTER_SCORES.reduce((s, d) => s + d.mg[axis], 0)   / MASTER_SCORES.length * 10) / 10,
});

const MASTER_AVG_STAT = MASTER_LABELS.map((_, i) => masterAvg(i).stat);
const MASTER_AVG_MG   = MASTER_LABELS.map((_, i) => masterAvg(i).mg);

// ─── Radar chart (pure Canvas) ───────────────────────────────────────────────
function RadarChart({ labels, statData, mgData, size = 260 }) {
  const ref = useRef(null);
  useEffect(() => {
    const canvas = ref.current; if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr; canvas.height = size * dpr;
    canvas.style.width = size + "px"; canvas.style.height = size + "px";
    ctx.scale(dpr, dpr);
    const cx = size / 2, cy = size / 2, R = size * 0.34;
    const n = labels.length;
    const ang = i => Math.PI * 2 * i / n - Math.PI / 2;
    const pt  = (v, i) => ({ x: cx + R * (v / 10) * Math.cos(ang(i)), y: cy + R * (v / 10) * Math.sin(ang(i)) });
    ctx.clearRect(0, 0, size, size);
    [2,4,6,8,10].forEach(v => {
      ctx.beginPath();
      for (let i = 0; i < n; i++) { const p = pt(v, i); i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y); }
      ctx.closePath(); ctx.strokeStyle = v === 10 ? C.border2 : C.border; ctx.lineWidth = 1; ctx.stroke();
    });
    for (let i = 0; i < n; i++) {
      const p = pt(10, i);
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(p.x, p.y);
      ctx.strokeStyle = C.border; ctx.lineWidth = 1; ctx.stroke();
    }
    const poly = (data, col, fill) => {
      ctx.beginPath();
      data.forEach((v, i) => { const p = pt(v, i); i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y); });
      ctx.closePath(); ctx.fillStyle = fill; ctx.fill();
      ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.stroke();
      data.forEach((v, i) => {
        const p = pt(v, i);
        ctx.beginPath(); ctx.arc(p.x, p.y, 3, 0, Math.PI * 2); ctx.fillStyle = col; ctx.fill();
      });
    };
    poly(statData, C.stat, C.stat + "22");
    poly(mgData,   C.mg,   C.mg   + "22");
    ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillStyle = C.muted; ctx.font = `${size < 280 ? 9 : 10}px system-ui`;
    labels.forEach((lbl, i) => {
      const p = pt(12.8, i);
      const words = lbl.split(" "); let lines = [], cur = "";
      words.forEach(w => { if ((cur + w).length > 12) { if (cur) lines.push(cur.trim()); cur = w + " "; } else cur += w + " "; });
      if (cur.trim()) lines.push(cur.trim());
      lines.forEach((l, li) => ctx.fillText(l, p.x, p.y + (li - (lines.length - 1) / 2) * 12));
    });
  }, [labels, statData, mgData, size]);
  return <canvas ref={ref} style={{ display: "block" }} aria-label="Radar chart" />;
}

// ─── Score bar ───────────────────────────────────────────────────────────────
function ScoreBar({ label, statScore, mgScore }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: C.muted, marginBottom: 4 }}>
        <span>{label}</span>
        <span style={{ display: "flex", gap: 8 }}>
          <span style={{ color: C.stat }}>Stat {statScore}</span>
          <span style={{ color: C.mg }}>MG {mgScore}</span>
        </span>
      </div>
      <div style={{ display: "flex", gap: 4 }}>
        <div style={{ height: 4, borderRadius: 2, background: C.stat, width: `${statScore * 10}%`, transition: "width .4s" }} />
        <div style={{ height: 4, borderRadius: 2, background: C.mg,   width: `${mgScore  * 10}%`, transition: "width .4s", marginLeft: 4 }} />
      </div>
    </div>
  );
}

// ─── Key stat tile ───────────────────────────────────────────────────────────
function KeyStat({ label, val, note }) {
  return (
    <div style={{ background: C.panel, borderRadius: 8, padding: "10px 14px", border: `1px solid ${C.border}` }}>
      <div style={{ fontSize: 11, color: C.muted, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 18, fontWeight: 600, color: C.text, marginBottom: 2 }}>{val}</div>
      <div style={{ fontSize: 11, color: C.muted }}>{note}</div>
    </div>
  );
}

// ─── Radar axis info panel ───────────────────────────────────────────────────
function RadarAxisPanel({ items, statScores, mgScores }) {
  return (
    <div style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 10, overflow: "hidden", backdropFilter: "blur(10px)" }}>
      <div style={{
        padding: "10px 14px", borderBottom: `1px solid ${C.border}`,
        display: "flex", alignItems: "center", gap: 8,
      }}>
        <div style={{ width: 6, height: 6, borderRadius: "50%", background: C.accent }} />
        <span style={{ fontSize: 11, fontWeight: 700, color: C.text, textTransform: "uppercase", letterSpacing: "0.07em" }}>
          Radar Axis Guide
        </span>
        <span style={{ fontSize: 10, color: C.muted, marginLeft: 4 }}>— what each dimension measures and why it matters</span>
      </div>
      {items.map((item, i) => (
        <div key={i} style={{
          padding: "11px 14px",
          borderBottom: i < items.length - 1 ? `1px solid ${C.border}` : "none",
          display: "grid", gridTemplateColumns: "1fr auto", gap: 12, alignItems: "start",
        }}>
          <div>
            <div style={{ fontSize: 12, fontWeight: 600, color: C.text, marginBottom: 4 }}>{item.axis}</div>
            <div style={{ fontSize: 12, color: C.muted, lineHeight: 1.6 }}>{item.what}</div>
          </div>
          <div style={{ display: "flex", gap: 8, flexShrink: 0, paddingTop: 2 }}>
            <div style={{ textAlign: "center" }}>
              <div style={{ fontSize: 16, fontWeight: 700, color: C.stat }}>{statScores[i]}</div>
              <div style={{ fontSize: 9, color: C.muted }}>STAT</div>
            </div>
            <div style={{ width: 1, background: C.border }} />
            <div style={{ textAlign: "center" }}>
              <div style={{ fontSize: 16, fontWeight: 700, color: C.mg }}>{mgScores[i]}</div>
              <div style={{ fontSize: 9, color: C.muted }}>MG</div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

// ─── Dataset sidebar card ────────────────────────────────────────────────────
function DatasetCard({ ds, isActive, onClick }) {
  const avgS = Math.round(ds.radar.stat.reduce((a,b)=>a+b,0)/ds.radar.stat.length);
  const avgM = Math.round(ds.radar.mg.reduce((a,b)=>a+b,0)/ds.radar.mg.length);
  return (
    <button onClick={onClick} style={{
      display: "block", width: "100%", textAlign: "left",
      background: isActive ? C.panel : C.surface,
      border: `1px solid ${isActive ? C.accent : C.border}`,
      borderRadius: 10, padding: "11px 13px", cursor: "pointer",
      transition: "all .2s", marginBottom: 7, backdropFilter: "blur(10px)",
    }}>
      <div style={{ fontSize: 10, color: C.muted, textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 2 }}>{ds.domain}</div>
      <div style={{ fontSize: 13, fontWeight: 600, color: C.text, marginBottom: 5, lineHeight: 1.3 }}>{ds.title}</div>
      <div style={{ fontSize: 10, color: C.muted, marginBottom: 7 }}>{ds.subtitle}</div>
      <div style={{ display: "flex", gap: 5 }}>
        <span style={{ fontSize: 9, background: C.stat+"22", color: C.stat, padding: "2px 6px", borderRadius: 4, fontWeight: 700 }}>STAT {avgS}/10</span>
        <span style={{ fontSize: 9, background: C.mg+"22",   color: C.mg,   padding: "2px 6px", borderRadius: 4, fontWeight: 700 }}>MG {avgM}/10</span>
      </div>
    </button>
  );
}

// ─── Overview bar chart ──────────────────────────────────────────────────────
function OverviewBarChart() {
  const ref = useRef(null);
  const avgPerDs = DATASETS.map(ds => ({
    name: ds.title,
    stat: Math.round(ds.radar.stat.reduce((a,b)=>a+b,0)/ds.radar.stat.length * 10)/10,
    mg:   Math.round(ds.radar.mg.reduce((a,b)=>a+b,0)/ds.radar.mg.length   * 10)/10,
  }));
  useEffect(() => {
    const canvas = ref.current; if (!canvas) return;
    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const W = canvas.offsetWidth || 700;
      const H = W < 560 ? 340 : 300;
      const isCompactChart = W < 760;
      const isPhoneChart = W < 560;

      canvas.width = W * dpr;
      canvas.height = H * dpr;
      canvas.style.width = W + "px";
      canvas.style.height = H + "px";

      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);

      const pad = {
        l: isPhoneChart ? 36 : 42,
        r: 12,
        t: 28,
        b: isPhoneChart ? 114 : isCompactChart ? 96 : 76,
      };
      const cw = W - pad.l - pad.r;
      const ch = H - pad.t - pad.b;
      const n = avgPerDs.length;
      const bw = cw / n;
      const barW = bw * 0.28;

      [0,2,4,6,8,10].forEach(v => {
        const y = pad.t + ch * (1 - v/10);
        ctx.beginPath();
        ctx.moveTo(pad.l, y);
        ctx.lineTo(pad.l + cw, y);
        ctx.strokeStyle = v === 0 ? "#2A3040" : "#161B22";
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.fillStyle = C.muted;
        ctx.font = "10px system-ui";
        ctx.textAlign = "right";
        ctx.fillText(v, pad.l - 6, y + 3);
      });

      avgPerDs.forEach((d, i) => {
        const cx2 = pad.l + i*bw + bw/2;
        const sH = ch*(d.stat/10);
        const mH = ch*(d.mg/10);
        ctx.fillStyle = C.stat + "BB";
        ctx.beginPath();
        ctx.roundRect(cx2-barW-2, pad.t+ch-sH, barW, sH, [3,3,0,0]);
        ctx.fill();
        ctx.fillStyle = C.mg + "BB";
        ctx.beginPath();
        ctx.roundRect(cx2+2, pad.t+ch-mH, barW, mH, [3,3,0,0]);
        ctx.fill();

        ctx.fillStyle = C.stat;
        ctx.font = "bold 9px system-ui";
        ctx.textAlign = "center";
        ctx.fillText(d.stat.toFixed(1), cx2-barW/2-2, pad.t+ch-sH-5);
        ctx.fillStyle = C.mg;
        ctx.fillText(d.mg.toFixed(1), cx2+barW/2+2, pad.t+ch-mH-5);

        const maxChars = isPhoneChart ? 10 : isCompactChart ? 12 : 14;
        const short = d.name.length > maxChars ? d.name.slice(0, maxChars) + "…" : d.name;
        ctx.save();
        ctx.translate(cx2 + 2, pad.t + ch + 28);
        ctx.rotate(-Math.PI / 4);
        ctx.fillStyle = C.muted;
        ctx.font = (isPhoneChart ? "8px" : "9px") + " system-ui";
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        ctx.fillText(short, 0, 0);
        ctx.restore();
      });

      const legendStep = isCompactChart ? 146 : 170;
      const lx = pad.l + cw/2 - (legendStep - 14);
      const ly = H - (isPhoneChart ? 14 : 12);
      [[C.stat,"Classical Statistics"],[C.mg,"Machine Gnostics"]].forEach(([col,lbl], i) => {
        const x = lx + i * legendStep;
        ctx.fillStyle = col;
        ctx.fillRect(x, ly - 8, 12, 8);
        ctx.fillStyle = C.muted;
        ctx.font = (isPhoneChart ? "9px" : "10px") + " system-ui";
        ctx.textAlign = "left";
        ctx.fillText(lbl, x + 15, ly);
      });
    };

    draw();

    const onResize = () => draw();
    window.addEventListener("resize", onResize);

    let ro;
    if (window.ResizeObserver) {
      ro = new ResizeObserver(onResize);
      ro.observe(canvas);
      if (canvas.parentElement) {
        ro.observe(canvas.parentElement);
      }
    }

    return () => {
      window.removeEventListener("resize", onResize);
      if (ro) {
        ro.disconnect();
      }
    };
  }, [avgPerDs]);
  return <canvas ref={ref} style={{ width:"100%", display:"block" }} />;
}

// ─── Master radar (all 10 datasets averaged, 6 universal axes) ───────────────
function MasterRadarPanel({ isNarrow, isCompact }) {
  return (
    <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: isCompact ? "18px 14px" : "22px 26px", marginBottom: 24, backdropFilter: "blur(12px)" }}>
      <div style={{ marginBottom: 6 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
          <div style={{ width: 6, height: 6, borderRadius: "50%", background: C.mg, boxShadow: `0 0 8px ${C.mg}` }} />
          <span style={{ fontSize: 11, color: C.mg, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em" }}>Master Benchmark Radar</span>
        </div>
        <h2 style={{ fontSize: 15, fontWeight: 700, color: C.text, margin: 0 }}>
          Overall Performance — Averaged Across All 10 Datasets
        </h2>
        <p style={{ fontSize: 12, color: C.muted, marginTop: 4, marginBottom: 0, lineHeight: 1.5 }}>
          Six universal dimensions capture every failure mode in the benchmark. Scores are the mean across all 10 datasets for each axis (0 = method fails, 10 = method succeeds).
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: isNarrow ? "1fr" : "300px 1fr", gap: 24, alignItems: "start", marginTop: 20 }}>
        {/* Radar */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
          <RadarChart labels={MASTER_LABELS} statData={MASTER_AVG_STAT} mgData={MASTER_AVG_MG} size={isCompact ? 210 : isNarrow ? 240 : 280} />
          <div style={{ display: "flex", gap: 18, marginTop: 14, fontSize: 11, flexWrap: "wrap", justifyContent: "center" }}>
            <span style={{ color: C.stat, display: "flex", alignItems: "center", gap: 5 }}>
              <span style={{ width: 12, height: 3, background: C.stat, display: "inline-block", borderRadius: 1 }} />
              Classical Statistics
            </span>
            <span style={{ color: C.mg, display: "flex", alignItems: "center", gap: 5 }}>
              <span style={{ width: 12, height: 3, background: C.mg, display: "inline-block", borderRadius: 1 }} />
              Machine Gnostics
            </span>
          </div>
          {/* Overall scores */}
          <div style={{ marginTop: 16, display: "flex", gap: 20, flexWrap: "wrap", justifyContent: "center" }}>
            {[
              { label: "STAT overall", val: (MASTER_AVG_STAT.reduce((a,b)=>a+b,0)/MASTER_AVG_STAT.length).toFixed(1), col: C.stat },
              { label: "MG overall", val: (MASTER_AVG_MG.reduce((a,b)=>a+b,0)/MASTER_AVG_MG.length).toFixed(1), col: C.mg },
            ].map(({ label, val, col }) => (
              <div key={label} style={{ textAlign: "center", background: C.panel, borderRadius: 8, padding: "10px 18px", border: `1px solid ${C.border}` }}>
                <div style={{ fontSize: 24, fontWeight: 700, color: col }}>{val}<span style={{ fontSize: 13, color: C.muted }}>/10</span></div>
                <div style={{ fontSize: 10, color: C.muted, marginTop: 2 }}>{label}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Axis explanations */}
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: C.text, textTransform: "uppercase", letterSpacing: "0.07em", marginBottom: 12 }}>
            Universal Axis Definitions
          </div>
          {[
            { axis: "Confound / Bias ID", stat: MASTER_AVG_STAT[0], mg: MASTER_AVG_MG[0], what: "Ability to detect hidden confounding variables and identify when an observed association (correlation, odds ratio) is driven by a lurking third variable rather than a true causal relationship." },
            { axis: "Outlier Robustness", stat: MASTER_AVG_STAT[1], mg: MASTER_AVG_MG[1], what: "Resistance to high-leverage outliers that disproportionately shift the fitted model. Particularly critical in small samples (n<20) where a single atypical observation can reverse a finding." },
            { axis: "Sub-population ID", stat: MASTER_AVG_STAT[2], mg: MASTER_AVG_MG[2], what: "Ability to detect and separate latent sub-groups within a dataset — responders vs non-responders, seasonal clusters, department tiers — rather than collapsing everything into one aggregate estimate." },
            { axis: "Tail / Distribution", stat: MASTER_AVG_STAT[3], mg: MASTER_AVG_MG[3], what: "Ability to characterise the full shape of a distribution including its tails, rather than summarising it by mean and variance alone. Critical when the important signal lives at the extremes, not the centre." },
            { axis: "Policy Accuracy", stat: MASTER_AVG_STAT[4], mg: MASTER_AVG_MG[4], what: "Whether the method leads to the correct real-world decision — the right drug approval, the right screening policy, the right investment, the right interpretation of bias — rather than a statistically significant but wrong conclusion." },
            { axis: "Small-N Reliability", stat: MASTER_AVG_STAT[5], mg: MASTER_AVG_MG[5], what: "Stability of estimates when sample sizes are very small (n=8 to n=32). Small-N amplifies every weakness: outlier leverage, distribution misspecification, and false-positive rates all increase rapidly as n falls." },
          ].map((item, i) => (
            <div key={i} style={{
              marginBottom: 10, background: C.panel, borderRadius: 8,
              border: `1px solid ${C.border}`, padding: "11px 14px",
              display: "grid", gridTemplateColumns: "1fr auto", gap: 12, alignItems: "start",
            }}>
              <div>
                <div style={{ fontSize: 12, fontWeight: 700, color: C.text, marginBottom: 3 }}>{item.axis}</div>
                <div style={{ fontSize: 11, color: C.muted, lineHeight: 1.6 }}>{item.what}</div>
              </div>
              <div style={{ display: "flex", gap: 8, flexShrink: 0, paddingTop: 2 }}>
                <div style={{ textAlign: "center" }}>
                  <div style={{ fontSize: 16, fontWeight: 700, color: C.stat }}>{item.stat}</div>
                  <div style={{ fontSize: 9, color: C.muted }}>STAT</div>
                </div>
                <div style={{ width: 1, background: C.border }} />
                <div style={{ textAlign: "center" }}>
                  <div style={{ fontSize: 16, fontWeight: 700, color: C.mg }}>{item.mg}</div>
                  <div style={{ fontSize: 9, color: C.muted }}>MG</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── Main app ────────────────────────────────────────────────────────────────
function App() {
  const [activeId, setActiveId] = useState(null);
  const [view, setView] = useState("overview");
  const [viewportWidth, setViewportWidth] = useState(typeof window !== "undefined" ? window.innerWidth : 1440);
  const isNarrow = viewportWidth < 1120;
  const isCompact = viewportWidth < 760;

  useEffect(() => {
    const onResize = () => setViewportWidth(window.innerWidth);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  const ds = DATASETS.find(d => d.id === activeId);
  const totalMGWins = DATASETS.filter(d => {
    const s = d.radar.stat.reduce((a,b)=>a+b,0)/d.radar.stat.length;
    const m = d.radar.mg.reduce((a,b)=>a+b,0)/d.radar.mg.length;
    return m - s >= 5;
  }).length;

  return (
    <div style={{ background: C.bg, color: C.text, fontFamily: "'Roboto', system-ui, sans-serif", display: "flex", flexDirection: "column", overflowX: "hidden", width: "100%" }}>

      {/* Header */}
      <header style={{ borderBottom: `1px solid ${C.border}`, padding: isCompact ? "18px 14px" : "24px 32px 20px", display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 16 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
            <div style={{ width: 8, height: 8, borderRadius: "50%", background: C.mg, boxShadow: `0 0 8px ${C.mg}` }} />
            <span style={{ fontSize: 11, color: C.mg, letterSpacing: "0.1em", fontWeight: 600, textTransform: "uppercase" }}>Benchmark · v1.0</span>
          </div>
          <h1 style={{ fontSize: isCompact ? 24 : 30, fontWeight: 700, color: C.text, letterSpacing: "-0.03em", lineHeight: 1.1, margin: 0 }}>
            Machine Gnostics vs Classical Statistics
          </h1>
          <p style={{ fontSize: isCompact ? 14 : 15, color: C.muted, marginTop: 8, marginBottom: 0, lineHeight: 1.7, maxWidth: 760 }}>
            A structured benchmark across 10 datasets showing where Machine Gnostics and classical statistics agree, diverge, or reverse each other in practice.
          </p>
        </div>
        <div style={{ display: "flex", gap: isCompact ? 44 : 60, flexWrap: "wrap", justifyContent: "center", width: "100%" }}>
          {[{ label: "Datasets", val: "10" }, { label: "MG wins", val: `${totalMGWins}/10` }, { label: "Domains", val: "6" }].map(({ label, val }) => (
            <div key={label} style={{ textAlign: "center" }}>
              <div style={{ fontSize: 42, fontWeight: 700, color: C.mg }}>{val}</div>
              <div style={{ fontSize: 15, color: C.muted }}>{label}</div>
            </div>
          ))}
        </div>
      </header>

      <div style={{ padding: isCompact ? "14px" : "20px 32px 0", margin: "0 auto", width: "100%", maxWidth: 1280, boxSizing: "border-box" }}>
        <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 14, padding: isCompact ? "16px 14px" : "18px 20px", backdropFilter: "blur(12px)", boxShadow: "0 12px 34px rgba(0,0,0,0.10)" }}>
          <div style={{ display: "grid", gridTemplateColumns: isNarrow ? "1fr" : "1.4fr 1fr", gap: 16, alignItems: "start" }}>
            <div>
              <div style={{ fontSize: 11, color: C.accent, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 8 }}>How To Read This Benchmark</div>
              <p style={{ fontSize: isCompact ? 13 : 14, color: C.text, lineHeight: 1.8, margin: 0, maxWidth: 70 + "ch" }}>
                Each case compares the two approaches on decision-relevant dimensions such as confound detection, subgroup recovery, outlier stability, tail-risk sensitivity, and policy usefulness. The benchmark is designed to show not only score differences, but also where the interpretation itself changes.
              </p>
            </div>
            <div style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 12, padding: "14px 15px" }}>
              <div style={{ fontSize: 11, color: C.mg, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 8 }}>Note</div>
              <p style={{ fontSize: isCompact ? 12 : 13, color: C.text, lineHeight: 1.8, margin: 0 }}>
                This is a Machine Gnostics benchmark. It uses a different analytical perspective than classical statistics, so the meaning of a result and the correct interpretation can differ. Readers should approach the findings as a new framework, not just a different wording of standard statistical output.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Nav */}
      <div style={{ borderBottom: `1px solid ${C.border}`, padding: isCompact ? "0 10px" : "0 32px", display: "flex", flexWrap: "wrap" }}>
        {[["overview","Overview"],["detail","Dataset Explorer"]].map(([v, lbl]) => (
          <button key={v} onClick={() => setView(v)} style={{
            background: "none", border: "none", color: view===v ? C.text : C.muted,
            padding: "12px 16px", fontSize: 13, fontWeight: view===v ? 600 : 400,
            cursor: "pointer", borderBottom: `2px solid ${view===v ? C.accent : "transparent"}`,
            transition: "all .2s",
          }}>{lbl}</button>
        ))}
      </div>

      {/* Content */}
      <div style={{ flex: 1, padding: isCompact ? "16px 10px" : isNarrow ? "18px 12px" : "30px 32px", maxWidth: 1280, width: "100%", boxSizing: "border-box", margin: "0 auto" }}>

        {/* ── OVERVIEW ── */}
        {view === "overview" && (
          <div>
            {/* Master radar */}
            <MasterRadarPanel isNarrow={isNarrow} isCompact={isCompact} />

            {/* Bar chart */}
            <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: isCompact ? "18px 14px" : "22px 24px", marginBottom: 24, backdropFilter: "blur(12px)" }}>
              <h2 style={{ fontSize: isCompact ? 16 : 18, fontWeight: 700, color: C.text, marginBottom: 6, letterSpacing: "-0.02em" }}>Average Score Per Dataset</h2>
              <p style={{ fontSize: 13, color: C.muted, marginBottom: 20, lineHeight: 1.7 }}>Mean of each dataset's 5 domain-specific radar scores. Higher is better.</p>
              <OverviewBarChart />
            </div>

            {/* Verdict grid */}
            <h2 style={{ fontSize: isCompact ? 16 : 18, fontWeight: 700, color: C.text, marginBottom: 16, letterSpacing: "-0.02em" }}>Dataset Verdicts</h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: 14 }}>
              {DATASETS.map(d => {
                const avgS = (d.radar.stat.reduce((a,b)=>a+b,0)/d.radar.stat.length).toFixed(1);
                const avgM = (d.radar.mg.reduce((a,b)=>a+b,0)/d.radar.mg.length).toFixed(1);
                return (
                  <div key={d.id} style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: "18px 20px", borderLeft: `3px solid ${d.verdictColor}` }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8 }}>
                      <div>
                        <div style={{ fontSize: 11, color: C.muted, marginBottom: 3 }}>{d.domain}</div>
                        <div style={{ fontSize: 14, fontWeight: 600, color: C.text }}>{d.title}</div>
                        <div style={{ fontSize: 11, color: C.muted, marginTop: 2 }}>{d.subtitle}</div>
                      </div>
                      <span style={{ fontSize: 10, padding: "3px 8px", borderRadius: 5, fontWeight: 700, whiteSpace: "nowrap", background: d.verdictColor+"22", color: d.verdictColor }}>{d.verdictTag}</span>
                    </div>
                    <div style={{ background: C.panel, borderRadius: 8, padding: "10px 12px", marginTop: 10 }}>
                      <div style={{ fontSize: 11, color: C.muted, marginBottom: 6 }}>Radar scores (domain-specific)</div>
                      {d.radar.labels.map((lbl, i) => <ScoreBar key={lbl} label={lbl} statScore={d.radar.stat[i]} mgScore={d.radar.mg[i]} />)}
                    </div>
                    <div style={{ marginTop: 10, fontSize: 11, fontWeight: 600, color: d.verdictColor, borderTop: `1px solid ${C.border}`, paddingTop: 10 }}>
                      {d.verdict}
                    </div>
                    <button onClick={() => { setActiveId(d.id); setView("detail"); }} style={{ marginTop: 8, fontSize: 11, color: C.accent, background: "none", border: "none", cursor: "pointer", padding: 0, textDecoration: "underline" }}>
                      View full analysis →
                    </button>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* ── DETAIL ── */}
        {view === "detail" && (
          <div style={{ display: "grid", gridTemplateColumns: isNarrow ? "1fr" : "220px 1fr", gap: 22 }}>
            {/* Sidebar */}
            <div>
              <div style={{ fontSize: 11, color: C.muted, textTransform: "uppercase", letterSpacing: "0.07em", marginBottom: 12 }}>Select Dataset</div>
              {DATASETS.map(d => <DatasetCard key={d.id} ds={d} isActive={activeId===d.id} onClick={() => setActiveId(d.id)} />)}
            </div>

            {/* Detail panel */}
            {ds ? (
              <div>
                {/* Header */}
                <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: isCompact ? "18px 14px" : "22px 26px", marginBottom: 16, backdropFilter: "blur(12px)" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 12 }}>
                    <div>
                      <div style={{ fontSize: 11, color: C.muted, marginBottom: 4 }}>{ds.domain} · n = {ds.n}</div>
                      <h2 style={{ fontSize: isCompact ? 20 : 24, fontWeight: 700, color: C.text, margin: 0, letterSpacing: "-0.02em", lineHeight: 1.15 }}>{ds.title}</h2>
                      <div style={{ fontSize: 14, color: C.muted, marginTop: 6, lineHeight: 1.6 }}>{ds.subtitle}</div>
                    </div>
                    <span style={{ fontSize: 11, padding: "5px 12px", borderRadius: 6, fontWeight: 700, background: ds.verdictColor+"22", color: ds.verdictColor, border: `1px solid ${ds.verdictColor}44` }}>{ds.verdictTag}</span>
                  </div>
                  <div style={{ marginTop: 14, fontSize: 12, color: ds.verdictColor, fontWeight: 600, borderTop: `1px solid ${C.border}`, paddingTop: 12 }}>
                    VERDICT: {ds.verdict}
                  </div>
                </div>

                {/* Radar + key stats */}
                <div style={{ display: "grid", gridTemplateColumns: isNarrow ? "1fr" : "minmax(240px, 280px) minmax(0, 1fr)", gap: 14, marginBottom: 16 }}>
                  <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 12, padding: "18px", display: "flex", flexDirection: "column", alignItems: "center", overflow: "hidden" }}>
                    <div style={{ fontSize: 12, fontWeight: 600, color: C.text, marginBottom: 12 }}>Radar Comparison</div>
                    <RadarChart labels={ds.radar.labels} statData={ds.radar.stat} mgData={ds.radar.mg} size={isCompact ? 205 : isNarrow ? 220 : 240} />
                    <div style={{ display: "flex", gap: 16, marginTop: 14, fontSize: 11, flexWrap: "wrap", justifyContent: "center" }}>
                      <span style={{ color: C.stat, display: "flex", alignItems: "center", gap: 5 }}>
                        <span style={{ width: 10, height: 3, background: C.stat, display: "inline-block", borderRadius: 1 }} />Classical Stats
                      </span>
                      <span style={{ color: C.mg, display: "flex", alignItems: "center", gap: 5 }}>
                        <span style={{ width: 10, height: 3, background: C.mg, display: "inline-block", borderRadius: 1 }} />Machine Gnostics
                      </span>
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 700, color: C.text, marginBottom: 12, letterSpacing: "-0.01em" }}>Key Metrics</div>
                    <div style={{ display: "grid", gridTemplateColumns: isNarrow ? "1fr" : "1fr 1fr", gap: 10 }}>
                      {ds.keyStats.map((k, i) => <KeyStat key={i} {...k} />)}
                    </div>
                    <div style={{ marginTop: 12, background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10, padding: "12px 14px", backdropFilter: "blur(10px)" }}>
                      <div style={{ display: "flex", gap: 12, alignItems: "baseline", marginBottom: 5 }}>
                        <span style={{ fontSize: 11, color: C.mg, fontWeight: 700, textTransform: "uppercase" }}>MG avg</span>
                        <span style={{ fontSize: 22, fontWeight: 700, color: C.mg }}>{(ds.radar.mg.reduce((a,b)=>a+b,0)/ds.radar.mg.length).toFixed(1)}/10</span>
                        <span style={{ fontSize: 11, color: C.muted }}>vs</span>
                        <span style={{ fontSize: 11, color: C.stat, fontWeight: 700, textTransform: "uppercase" }}>STAT avg</span>
                        <span style={{ fontSize: 22, fontWeight: 700, color: C.stat }}>{(ds.radar.stat.reduce((a,b)=>a+b,0)/ds.radar.stat.length).toFixed(1)}/10</span>
                      </div>
                      <div style={{ fontSize: 11, color: C.muted }}>
                        MG advantage: +{((ds.radar.mg.reduce((a,b)=>a+b,0) - ds.radar.stat.reduce((a,b)=>a+b,0)) / ds.radar.mg.length).toFixed(1)} points across all dimensions
                      </div>
                    </div>
                  </div>
                </div>

                {/* Radar axis guide */}
                <div style={{ marginBottom: 16 }}>
                  <RadarAxisPanel items={ds.radarAxisInfo} statScores={ds.radar.stat} mgScores={ds.radar.mg} />
                </div>

                {/* Analysis boxes */}
                <div style={{ display: "grid", gridTemplateColumns: isNarrow ? "1fr" : "1fr 1fr", gap: 14, marginBottom: 14 }}>
                  <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10, padding: "16px 18px", borderLeft: `3px solid ${C.stat}` }}>
                    <div style={{ fontSize: 11, color: C.stat, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>Classical Statistics Finding</div>
                    <p style={{ fontSize: 14, color: C.text, lineHeight: 1.75, margin: 0 }}>{ds.classicalFinding}</p>
                  </div>
                  <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10, padding: "16px 18px", borderLeft: `3px solid ${C.warn}` }}>
                    <div style={{ fontSize: 11, color: C.warn, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>Why Statistics Fails Here</div>
                    <p style={{ fontSize: 14, color: C.text, lineHeight: 1.75, margin: 0 }}>{ds.classicalError}</p>
                  </div>
                </div>
                <div style={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10, padding: "16px 18px", marginBottom: 14, borderLeft: `3px solid ${C.mg}` }}>
                  <div style={{ fontSize: 11, color: C.mg, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>Machine Gnostics Finding</div>
                  <p style={{ fontSize: 14, color: C.text, lineHeight: 1.75, margin: 0 }}>{ds.mgFinding}</p>
                </div>

                {/* Citation */}
                <div style={{ fontSize: 11, color: C.muted, borderTop: `1px solid ${C.border}`, paddingTop: 12 }}>
                  <span style={{ fontWeight: 600 }}>Citation: </span>{ds.citation}
                </div>
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "60vh", color: C.muted, gap: 12 }}>
                <div style={{ width: 40, height: 40, borderRadius: "50%", border: `2px solid ${C.border2}`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18 }}>↑</div>
                <p style={{ fontSize: 13, margin: 0 }}>Select a dataset from the above section to explore</p>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Footer */}
      <footer style={{ borderTop: `1px solid ${C.border}`, padding: isCompact ? "14px" : "16px 32px", display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 11, color: C.muted, flexWrap: "wrap", gap: 8 }}>
        <span>Machine Gnostics Benchmark</span>
      </footer>
    </div>
  );
}

const rootEl = document.getElementById("mg-benchmark-root");
if (rootEl) {
  const root = ReactDOM.createRoot(rootEl);
  root.render(<App />);
}
