# AlgerianLegalBench v3.0 — Annotation Style Guide

**Document version:** 1.0 — April 2026
**Purpose:** This document provides 10 fully worked reasoning chain examples (one per query type, plus two extra unanswerable variants). Every annotator must read this guide before writing or reviewing any reasoning chain. This document maps directly to §3.2 of the ArabicNLP 2026 paper.

---

## 1. Overview

Every question in AlgerianLegalBench v3.0 has a `reasoning_chain` field containing exactly **4 steps**. The chain is the methodological core that separates a published benchmark from a question bank. It must be:

- **Self-explanatory**: Any Algerian lawyer reading it cold must understand the logic without external context.
- **Cross-annotated**: The annotator who did NOT write the question fills the reasoning chain. The author's chain is a first draft only.
- **Template-specific**: Each query type has a dedicated template (A, B, C, or D) with prescribed step labels.

### Annotation Protocol

| Role | Responsibility |
|---|---|
| **Author** | Writes question + ground truth answer + draft chain |
| **Cross-annotator** | Independently fills the reasoning chain using only the question and legal texts |
| **Resolver** | Senior expert resolves disagreements where Kappa < threshold |

### Inter-Annotator Agreement Targets

| Dimension | Target Kappa |
|---|---|
| Agreement on answer | ≥ 0.70 |
| Agreement on articles | ≥ 0.60 |
| Agreement on chain | ≥ 0.55 |

---

## 2. Templates

### Template A — Standard (EA, CD, RA, LM)

| Step | Label | Content |
|---|---|---|
| 1 | Legal issue | Name the precise sub-issue (not "family law" but "whether khul' requires husband consent") |
| 2 | Applicable rule | Cite exact article, quote operative phrase, explain hierarchy |
| 3 | Application/Subsumption | Map facts to rule elements (RA) or delineate scope (CD) or simplify (LM) |
| 4 | Conclusion | State outcome; flag residual judicial discretion |

### Template B — Temporal/Factual (TF)

| Step | Label | Content |
|---|---|---|
| 1 | Version identification | Which version was in force at the relevant date? |
| 2 | Amendment timeline | What changed, when, cite the amending law |
| 3 | Correct version applied | Apply the right version; explain why the other doesn't apply |
| 4 | Model failure description | What would a pre-trained model say? (This becomes the paper's failure analysis) |

### Template C — Multi-Hop (MH)

| Step | Label | Content |
|---|---|---|
| 1 | Name the chain | List all codes/articles in traversal order |
| 2 | Hop 1 | Apply first code; state intermediate conclusion |
| 3 | Hop 2 (and 3) | Apply next code using Hop 1's output; show the handoff |
| 4 | Synthesis + failure point | Where does a single-code retrieval system fail? |

### Template D — Unanswerable (UN)

| Step | Label | Content |
|---|---|---|
| 1 | Failure mode | (A) Jurisdictional infection, (B) Fictitious citation, or (C) Genuine absence |
| 2 | Demonstrate the absence | Cite the code sections that do NOT contain the assumed rule |
| 3 | Incorrect model answer | Describe what a jurisdictionally infected model would say |
| 4 | Correct response | "This concept does not exist under Algerian law..." |

---

## 3. Worked Examples (10)

---

### Example 1: Exact Article (EA)

**ID:** `com_ea_q01` | **Category:** Commercial Law | **Difficulty:** Easy

**Question:** أبرم تاجر عقد بيع مع تاجر آخر دون عقد مكتوب. هل يمكن إثبات العقد بوسائل أخرى؟

**Reasoning Chain:**

> **Step 1 — Legal issue:** What are the means of proof for commercial contracts under Algerian law?
>
> **Step 2 — Applicable rule:** Article 30 of the Commercial Code enumerates six means of proof: official deeds, private deeds, accepted invoices, correspondence, account books, and testimonial evidence at the court's discretion.
>
> **Step 3 — Application:** Direct article retrieval — the question maps to a specific statutory provision without requiring fact-pattern analysis.
>
> **Step 4 — Conclusion:** Commercial contracts can be proven by any of the six means listed in Article 30, including testimonial evidence, regardless of the contract value — unlike civil contracts which are limited by Articles 333-334 of the Civil Code.

---

### Example 2: Conceptual/Definitional (CD)

**ID:** `lab_cd_q01` | **Category:** Labor Law | **Difficulty:** Medium

**Question:** ما الفرق بين الاتفاقية الجماعية واتفاقية المؤسسة في القانون الجزائري؟

**Reasoning Chain:**

> **Step 1 — Legal issue:** What distinguishes a collective bargaining agreement (convention collective) from an enterprise agreement (convention d'entreprise) in Algerian labor law?
>
> **Step 2 — Applicable rule:** Law 90-11, Articles 114 et seq. define the collective bargaining framework. A collective agreement covers an entire sector; an enterprise agreement covers a single employer.
>
> **Step 3 — Scope delineation:** The collective agreement binds all employers and workers in a given sector of activity and is negotiated by representative unions and employer federations. The enterprise agreement is negotiated at the firm level and may provide more favorable terms than the sectoral agreement but cannot derogate from its minimum standards.
>
> **Step 4 — Conclusion:** The key distinction is scope (sector vs. firm) and hierarchy (enterprise agreement ≥ collective agreement ≥ statutory minimum). Both are binding once registered.

---

### Example 3: Rule Application (RA)

**ID:** `fam_ra_q10` | **Category:** Family Law | **Difficulty:** Hard

**Question:** نزاع حول التلقيح الاصطناعي داخل زواج شرعي. ما شروطه وأثره على ثبوت النسب؟

**Reasoning Chain:**

> **Step 1 — Legal issue:** Whether a wife can obtain khul' (خلع) without her husband's consent under current Algerian family law, post-2005 amendment.
>
> **Step 2 — Applicable rule:** Article 54 of the Family Code (as amended by Ordonnance 05-02 of 27 February 2005): "The wife may, without the husband's consent, redeem herself in exchange for financial compensation. If the spouses do not agree on the amount, the judge shall fix it at no more than the equivalent dowry (صداق المثل) at the time of the judgment."
>
> **Step 3 — Subsumption:** Under the pre-2005 text, the husband's consent was required. Under the current text, the wife has a unilateral right to khul'. The judge's role is limited to fixing the compensation amount. The husband cannot block the khul'.
>
> **Step 4 — Conclusion:** Yes. Under Article 54 (post-2005), the wife can obtain khul' without husband's consent. The judge fixes compensation ≤ sadāq al-mithl. Residual judicial discretion is limited to the amount, not the principle.

---

### Example 4: Temporal/Factual (TF)

**ID:** `con_tf_q01` | **Category:** Constitutional Law | **Difficulty:** Hard

**Question:** كيف تطور تحديد العهدات الرئاسية في الدساتير المتعاقبة (1996، 2008، 2020)؟

**Reasoning Chain:**

> **Step 1 — Version identification:** Three constitutional versions govern presidential term limits: Constitution 1996 (original), Constitutional Amendment 2008, and Constitution 2020 (current, Article 88).
>
> **Step 2 — Amendment timeline:** (a) 1996 Constitution: two 5-year terms, renewable once. (b) 2008 Amendment (Law 08-19): removed all term limits — unlimited re-election. (c) 2020 Constitution (Article 88): reinstated the limit at two terms, explicitly non-renewable (غير قابلة للتجديد).
>
> **Step 3 — Correct version applied:** The 2020 Constitution is in force. Article 88 limits the president to two five-year terms. The 2008 amendment is abrogated on this point.
>
> **Step 4 — Model failure description:** A model trained on pre-2020 data would state that there is no term limit (citing the 2008 amendment). A model with 2020 data but trained on French constitutional law might confuse the Algerian two-term limit with the French system. The correct answer must cite Article 88 of the 2020 Constitution specifically.

---

### Example 5: Multi-Hop (MH)

**ID:** `xdom_mh_q03` | **Category:** Cross-Domain | **Difficulty:** Hard

**Question:** مستثمر أجنبي يريد إنشاء مصنع. ما المراحل من تسجيل الشركة إلى رخصة البناء مع المزايا الضريبية؟

**Reasoning Chain:**

> **Step 1 — Name the chain:** Commercial Code (company registration) → Investment Law 22-18 (declaration + incentives) → Municipal Law 11-10 (building permit) → Tax Code (IBS/TVA). Four-code traversal.
>
> **Step 2 — Hop 1 (Company registration):** Under the Commercial Code, the foreign investor registers a SARL or SPA at the CNRC. Post-2022, 100% foreign ownership is permitted outside strategic sectors (Law 22-18, Art. 2).
>
> **Step 3 — Hop 2 (Investment → Building permit):** File a declaration with ANDI under Law 22-18 to unlock tax incentives. Simultaneously, apply for a building permit with the APC under Law 11-10. If classified, an EIE is required (Law 03-10, Art. 15).
>
> **Step 4 — Synthesis + failure point:** A single-code retrieval system would find Commercial Code provisions but miss the investment incentives (22-18), building permit (11-10), and environmental study (03-10). The chain requires four sequential retrievals.

---

### Example 6: Long Context (LC)

**ID:** `fam_lc_q05` | **Category:** Family Law | **Difficulty:** Hard

**Question:** تزوج 'د' من 'هـ' وأنجبا طفلين. أبرم 'د' قرضاً دون علم زوجته. طالب الدائن الزوجة بالوفاء. طلبت الطلاق والحضانة والنفقة. حاول الدائن التنفيذ على أموال المسكن.

**Reasoning Chain:**

> **Step 1 — Legal issue:** Four sub-issues requiring synthesis: (1) wife's liability for husband's debt, (2) divorce, (3) custody and maintenance, (4) execution on household property.
>
> **Step 2 — Applicable rules:** Art. 54 Civil Code (privity), Art. 37 Family Code (separate patrimony), Art. 64 Family Code (custody priority), Art. 75 Family Code (child maintenance), Art. 600 CPCA (enforceable title), Art. 48 Family Code (divorce).
>
> **Step 3 — Synthesis:** The wife is not liable (Art. 54 + Art. 37 — separate patrimony). The creditor cannot execute on her property. She may obtain divorce (Art. 48), custody (Art. 64), maintenance (Art. 75). Execution limited to husband's assets (Art. 600).
>
> **Step 4 — Conclusion:** Six articles from four codes synthesized. A system retrieving only the Civil Code would miss family law protections and procedural requirements.

---

### Example 7: Layman/Darja (LM)

**ID:** `lab_lm_q01` | **Category:** Labor Law | **Difficulty:** Easy

**Question:** خدّام واحد راح يخدم عند واحد الباترون وما عندوش كونترا مكتوب. كي طردوه قالولو ما عندك حتى حق. واش يقدر يدير؟

**Reasoning Chain:**

> **Step 1 — Legal issue:** A worker without a written contract was fired and told he has no rights. Does an unwritten employment relationship confer legal protection?
>
> **Step 2 — Applicable rule:** Article 8 of Law 90-11: the employment relationship exists regardless of whether a written contract was signed. Proven by any means (payslips, witnesses, CNAS records).
>
> **Step 3 — Application in accessible language:** No written contract ≠ no rights. The employer is at fault for not formalizing the contract. The worker files a complaint with the Labor Inspectorate (مفتشية العمل), which attempts conciliation. If that fails, he brings a claim before the Social Section of the court.
>
> **Step 4 — Conclusion:** Full legal rights. The employer is liable for unfair dismissal and for failure to declare the worker with CNAS. Answer must avoid jargon — explain procedure step by step.

---

### Example 8: Unanswerable — US Infection (UN-A)

**ID:** `civ_un_q01` | **Category:** Civil Law | **Difficulty:** Hard

**Question:** هل يجوز الحكم بتعويضات عقابية (punitive damages) في القانون المدني الجزائري؟

**Reasoning Chain:**

> **Step 1 — Failure mode:** (A) Jurisdictional infection — US common law. "Punitive damages" is a US/common-law doctrine allowing damages exceeding actual harm to punish and deter.
>
> **Step 2 — Demonstrate the absence:** Article 124 of the Civil Code states: "Any act causing damage obliges the person at fault to make reparation." Articles 182-187 define reparation as compensatory only — restoring the status quo ante. No provision for punitive, exemplary, or multiplied damages exists anywhere in the Civil Code, Commercial Code, or any special statute.
>
> **Step 3 — Incorrect model answer:** A jurisdictionally infected model would say: "Yes, Algerian courts can award punitive damages in cases of gross negligence under Article 124." This is wrong — Article 124 establishes liability, not a punitive damages regime.
>
> **Step 4 — Correct response:** "Punitive damages do not exist in Algerian law. Article 124 limits compensation to actual damage suffered (الضرر الفعلي). The judge cannot award damages exceeding proven harm."

---

### Example 9: Unanswerable — Algerian Absence (UN-B)

**ID:** `con_un_q01` | **Category:** Constitutional Law | **Difficulty:** Hard

**Question:** هل يمكن للمحكمة الدستورية ممارسة رقابة لاحقة تلقائية دون إخطار؟

**Reasoning Chain:**

> **Step 1 — Failure mode:** (C) Genuine institutional absence. The question asks whether the Constitutional Court can exercise ex officio review without being seized.
>
> **Step 2 — Demonstrate the absence:** Article 190 exhaustively lists who may seize the Court: the President, the President of the Council of the Nation, the President of the APN, the Prime Minister. Article 188 adds the DIC mechanism. No article grants auto-saisine.
>
> **Step 3 — Incorrect model answer:** A model might say: "The Court has inherent jurisdiction to review any law it considers unconstitutional." This confuses the Algerian system with the German Federal Constitutional Court.
>
> **Step 4 — Correct response:** "The Constitutional Court cannot act ex officio. Review requires seizure by an authority listed in Article 190 or through DIC (Article 188)."

---

### Example 10: Unanswerable — French Infection (UN-C)

**ID:** `hous_un_q02` | **Category:** Housing Law | **Difficulty:** Hard

**Question:** هل يخضع الإيجار السكني في الجزائر لنظام contrôle des loyers كالقانون الفرنسي؟

**Reasoning Chain:**

> **Step 1 — Failure mode:** (A) Jurisdictional infection — French law. "Contrôle des loyers" (rent control with government-set maximum rents) is a French policy under the ALUR law of 2014 and Law 48-1360.
>
> **Step 2 — Demonstrate the absence:** The Civil Code (Arts. 467-537) establishes freedom of contract for rent. No price cap, no government-set reference rent. Public housing programs (AADL, LPP) set rents administratively, but this is a public allocation system, not private-market rent control.
>
> **Step 3 — Incorrect model answer:** A French-trained model would say: "Rent is regulated by government reference rates and landlords cannot exceed the plafond de loyer." This imports the ALUR framework.
>
> **Step 4 — Correct response:** "Algeria has no private-market rent control comparable to French law. Residential leases follow the Civil Code with freedom of contract on rent. Only public social housing has administrative pricing."

---

## 4. Quality Checklist

Before submitting any reasoning chain, verify:

- [ ] Exactly 4 steps using the correct template (A/B/C/D)
- [ ] Step labels match the template exactly
- [ ] All article citations include article number AND law name
- [ ] No circular reasoning (conclusion doesn't just restate the question)
- [ ] For TF: both pre- and post-amendment rules are stated
- [ ] For MH: all hops are named in Step 1 and each subsequent step shows the handoff
- [ ] For UN: the foreign jurisdiction source is identified in Step 1
- [ ] For UN: Step 3 describes a specific plausible wrong answer, not a vague "a model might say..."
- [ ] For LM: the chain translates the Darja question into formal legal terms in Step 1
- [ ] No French/Egyptian/US legal concepts imported into the answer

---

*Style Guide v1.0 — AlgerianLegalBench v3.0 — April 2026*
