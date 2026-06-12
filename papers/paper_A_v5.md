# Templatic Morphology as Decodable Geometry, and Abjad Letter-Values as an Attention Probe, in Classical Arabic Transformer Models

*A pre-registered cross-linguistic study of templatic Semitic morphology and letter-value structure in CAMeLBERT-ca and AlephBERT*

Ahmed Mislati — Independent researcher, Vienna · June 2026 · arXiv preprint cs.CL

**Keywords:** Arabic NLP, non-concatenative morphology, Abjad, attention geometry, CAMeLBERT, AlephBERT, classical Arabic, classical Hebrew, pre-registered cross-linguistic control

## Abstract

Classical Arabic is morphologically templatic: the triliteral root encodes a semantic field, and the morphological pattern (wazn) applies a consonant–vowel template that specifies grammatical and semantic function. Classical scholarly practice additionally assigns numerical values to each Arabic letter (the Abjad system), which has historically been used as an analytical instrument on inter-word structural relationships. We test whether these two encoding systems produce measurable geometric structure in transformer language models trained on classical Arabic, and we evaluate the result against a pre-registered cross-linguistic control on classical Hebrew.

We report two empirical findings on CAMeLBERT-ca, a BERT model trained on classical Arabic. **First**, the morphological pattern (wazn) is encoded as geometrically separable, linearly decodable structure in CAMeLBERT-ca embedding space: across 951 word forms in 200 root families, leave-one-out analogy completion exceeds 0.92 for every well-attested pattern and within-pattern clustering is significant for all of them (p ≤ 4×10⁻⁴), though the per-root pattern offsets are not a single consistent translation vector. **Second**, on a corpus of 120,000 classical Arabic sentences from Ibn ʿArabī’s *Fuṣūṣ al-Ḥikam* and *Futūḥāt al-Makkiyya* (~1.45M tokens), cross-root word pairs with proximate Abjad values receive higher mutual attention than Abjad-distal pairs, after controlling for positional distance and co-occurrence frequency (partial ρ = −0.056, n = 2,322,977 pairs, p beyond float64 precision; 1,000-permutation empirical p = 0.026; random-pair scrambling reduces effect by 71%). A pre-registered frequency-matched permutation null further bounds the Mashriqi-specific component as primarily about how Mashriqi values align with letter-frequency tiers (1,000-permutation p = 0.0589, near-miss at the 94th percentile against frequency-matched random mappings).

A pre-registered cross-linguistic control on classical Hebrew with AlephBERT and gematria returned a comparable attention-level signal (ρ = −0.078, n = 12,081,207), satisfying the pre-specified Outcome A: the attention-geometry signal is a property of templatic Semitic morphology in transformer representations rather than uniquely Arabic-Abjad. Within this regime, the Mashriqi Abjad mapping in Arabic is the only letter-value system tested that produces a small but significant specific increment above its letter-frequency baseline; gematria in Hebrew does not (1,000-permutation p = 0.23). The cross-linguistic asymmetry — Mashriqi outperforms random Arabic letter-value mappings, gematria does not outperform random Hebrew letter-value mappings — is the most distinctive empirical claim that survives the full set of controls.

We discuss the result as a localization argument: classical Arabic encoding systems produce measurable structure at the within-sentence attention-geometry layer, with a templatic-Semitic-morphology effect dominating in magnitude and a small Mashriqi-specific frequency-tier-alignment increment riding on top. The headline ρ should not be interpreted as the full Abjad-specific effect; the analytically interpretable Mashriqi-specific component above the letter-frequency baseline is approximately ρ = 0.026.

## 1. Introduction

### 1.1 The question

Arabic morphology is non-concatenative: the triliteral root encodes a semantic field, and the morphological pattern (wazn) applies a consonant–vowel template that specifies grammatical and semantic function (McCarthy, 1981; Holes, 2004; Ratcliffe, 1998). The pair (root, wazn) uniquely determines a word. This is well-established in Arabic linguistics. Less established is whether — and at what layer of representation — this structure surfaces in transformer language models.

Two recent strands of work motivate the present study. Work on the transformer treatment of root-and-pattern morphology has shown that Arabic roots cluster in distributional embedding space (Salama et al., 2018), a finding that extends to contextualised representations. The clustering is real but says nothing about how the *templatic* dimension — the wazn pattern — is internally represented. Independently, Alakeel et al. (2026) show that tokenizer alignment with morphological boundaries is neither necessary nor sufficient for productive Arabic generation, establishing that surface morphological competence cannot be cleanly attributed to tokenization choices. Their result establishes an important negative at the generation layer: surface tokenization alignment does not explain productive Arabic root–pattern generation. Our result is complementary. We show that, in a classical-Arabic encoder, templatic morphology and Abjad-related structure are nevertheless measurable in embedding and attention geometry — failure or ambiguity at the generation layer does not preclude structured representation at the geometry layer. Concretely, we ask whether the attention structure of a classical-Arabic-trained encoder reflects the structural relationships that traditional Arabic philology has formalised — wazn patterns, root-level correspondence, and numerical encoding via the Abjad system. These are separable empirical windows onto the same underlying question, and the present work reports a significant positive result for the attention-geometry window.

The Abjad system warrants brief introduction. Each Arabic letter is assigned a canonical integer value in the Mashriqi order (ا = 1, ب = 2, ج = 3, …, غ = 1000), systematised by 8th-century grammarians and used in classical scholarly practice as an analytical instrument on inter-word structural relationships (Colin, 1986; Endress, 2018). The system predates Ibn ʿArabī’s 13th-century work and is documented as an operational tool independent of any specific theological framework. For the purposes of this paper, the Abjad system functions as *one specific operationalisation* of inter-word structural relationship: a way of asking whether the model’s attention geometry registers the relationships that classical scholarly practice formalised. The claim is about classical Arabic’s internal structure, not about the numerical system per se.

### 1.2 What we contribute

The contributions of this paper are:

- A measurement of wazn geometry in CAMeLBERT-ca at the embedding level: across 200 root families, morphological pattern membership is significantly clustered and individual derivations are linearly decodable by analogy completion (>0.92 for all well-attested patterns), establishing that templatic morphology surfaces as separable geometric structure rather than as a mere co-occurrence correlate. We additionally report that the per-root pattern offsets are not a single consistent translation vector, bounding the strength of the “geometric operator” interpretation.

- A first measurement of Abjad-relationship at the attention-weight level, with a statistically robust effect across 2.3M cross-root word pairs, full robustness treatment (unrestricted permutation null, random-pair scrambling, frequency-matched permutation null), and a pre-registered three-outcome decision rule satisfied by a complementary Hebrew AlephBERT control.

- A cross-linguistic comparison locating the Mashriqi Abjad signal as a *frequency-tier-aligned* specificity within an attention-geometry effect that is otherwise templatic-Semitic-morphology-general. Of the two letter-value systems tested, only the Arabic Mashriqi mapping outperforms its language-matched random baseline at standard significance.

- A methodological note on register: the signal is detectable in classical-Arabic-trained encoders on classical corpora; modern-Arabic-trained encoders and modern corpora are not appropriate instruments for the measurement.

### 1.3 What we do not claim

We do not claim that transformer attention “performs Abjad analysis” or that the model has learned the historical scholarly practice. We claim only that the attention geometry of a classical-Arabic-trained encoder is statistically correlated with Abjad distance after appropriate controls, and that the correlation has the structural properties expected of a genuine attention-mediated effect (random-pair scrambling collapses it; the unrestricted permutation null is passed; the frequency-matched null is a near-miss). We also do not claim productive generation: the present work measures geometric structure only, and is silent on what the model can produce.

## 2. Background

### 2.1 Root-and-pattern morphology

Arabic is the canonical example of a non-concatenative templatic language. Words are formed by interleaving a consonantal root (typically three letters) with a vocalic and consonantal template (the wazn). The same root can yield dozens of distinct words by application of different patterns, with each pattern encoding systematic grammatical and semantic information. McCarthy (1981) provides the standard prosodic analysis; Holes (2004) and Ryding (2005) the standard descriptive grammars; Ratcliffe (1998) the comparative-Semitic analysis. We use the standard wazn family inventory throughout, distinguishing Form I (the base) from Forms II–X (derivations) and the principal nominal patterns (active and passive participles, verbal nouns).

Prior work on the geometric representation of Arabic morphology in distributional models is divided. Static word-embedding work has documented root-level clustering (Salama et al., 2018). Contextual-encoder work has examined morphological tasks at the application layer (Antoun et al., 2020; Inoue et al., 2021). Direct measurement of the wazn as geometric structure — testing whether morphological-pattern membership is separable and whether pattern offsets are linearly consistent across roots — has, to our knowledge, not been reported at the scale or with the cross-pattern coverage we present here.

### 2.2 The Abjad system

The Mashriqi Abjad assigns each of the 28 Arabic base letters an integer value in a system organised by the canonical sequence أبجد هوز حطي كلمن سعفص قرشت ثخذ ضظغ. Values run from 1 (ا) to 1000 (غ), with the high-value letters being the rarer letters in classical-register text (غ, ظ, ض, ذ, ث, ش). The system was systematised by 8th-century grammarians, principally al-Khalil ibn Aḥmad al-Farāhīdī, before the development of independent positional numerical notation, and remained in scholarly use through the classical period as an analytical instrument applied to inter-word structural relationships in poetry, theology, and esoteric commentary (Colin, 1986; Endress, 2018; Ifrah, 2000).

For our purposes, two properties of the Mashriqi system are operationally important. First, every word has a deterministic Abjad value (the sum of its letters’ values), making it straightforward to compute *Abjad distance* between any two words as |abjad(w₁) − abjad(w₂)|. Second, the assignment of high values to rare letters is not coincidental but reflects the system’s historical construction, where rare letters served as numerical placeholders for the higher decimal magnitudes. This property — that the system *by construction* aligns letter values with letter frequencies — becomes important in interpreting our frequency-matched permutation null below (§4.5, §5.3).

### 2.3 Transformer attention as measurement instrument

Transformer encoders compute, for each pair of tokens (i, j), an attention weight aᵢⱼ that reflects how much the model weighs token j when computing the representation of token i. Higher attention weights between two tokens indicate that the model has learned that those tokens are mutually relevant for computing their respective contextual representations (Vaswani et al., 2017; Clark et al., 2019). Attention weights are a *learned* property: they reflect the structural relationships the model has had to internalise to fit its training corpus, after controlling for surface confounds such as positional distance and co-occurrence frequency. As such, attention-weight analysis offers a distinct measurement window from co-occurrence statistics or from analogy-style geometric probes. We employ all three windows in this paper.

## 3. Methods

### 3.1 Corpus and model

For the Arabic experiments we used CAMeLBERT-ca (Inoue et al., 2021), the classical-Arabic-trained variant of the CAMeLBERT family, applied to a corpus of 120,000 classical Arabic sentences sampled from Ibn ʿArabī’s *Fuṣūṣ al-Ḥikam* and *Futūḥāt al-Makkiyya* (~1.45M tokens; OpenITI release URI:0734.IbnArabi; Romanov, 2019). The classical register is essential: pilot experiments with CAMeLBERT-mix (a multi-register model) and modern-Arabic corpora yielded ceiling effects and degraded discriminability (cross-root cosine mean ≈ 0.82), confirming that register-matched training is required for the measurement instrument to function. We note this register sensitivity as a methodological constraint, not a limitation of the underlying linguistic claim.

For the Hebrew control we used AlephBERT (Seker et al., 2022) on a 100,000-sentence corpus of classical Hebrew and Aramaic from the Sefaria-Export release (Talmud Bavli and Zohar). The Hebrew register asymmetry — AlephBERT is predominantly modern-Hebrew-trained while our test corpus is classical — is a methodological limitation we acknowledge explicitly (§5.4).

### 3.2 Wazn-as-operator measurement (Claim 1)

We assembled a lexicon of 951 word forms across 200 well-attested triliteral root families (after a manual review pass that removed 122 mis-segmented or mis-annotated forms and corrected 26 others from a 1,073-form raw set). For each root we recorded the surface form for each attested morphological pattern. Five patterns are densely attested across the family set — Form I (the base, V1), Form II (faʿʿala), the verbal noun (maṣdar), the active participle (fāʿil), and the passive participle (mafʿūl), each with n ≥ 98 — and we restrict per-pattern claims to these; the remaining patterns (Forms IV–X, adjectival, broken plural, and mixed forms) are individually sparse (n < 16) and reported only in aggregate. We computed three measures. (i) *Analogy completion*: for each pattern p we formed the cross-root mean offset d(p) = mean over roots of (vec(derived) − vec(base)), and for each held-out root r predicted vec(r, p) = vec(r, base) + d(p), scoring cosine similarity against the actual contextual embedding. (ii) *Pattern clustering*: for each pattern we compared mean within-pattern to mean cross-pattern cosine similarity (gap test, permutation p-value). (iii) *Offset consistency*: the mean pairwise cosine similarity among the per-root offset vectors of a pattern, measuring whether the transformation is a single shared direction. All embeddings are mean-pooled contextual vectors from CAMeLBERT-ca over three carrier sentences per word.

### 3.3 Word identification and Abjad computation

For the Arabic Abjad measurement, each sentence was tokenised using the CAMeLBERT-ca tokenizer; sub-word tokens were aggregated to word boundaries by mean-pooling of contextual embeddings. Abjad values were computed on the surface word form (undiacritised), with hamza-bearing alif (أ، إ، آ، ٱ) mapped to ا, tāʾ marbūṭa (ة) mapped to ت, and alif maqṣūra (ى) mapped to ي. The remaining 28 base letters carried their canonical Mashriqi values. For each sentence, all non-identical word-position pairs (i, j) within the sentence yielded a record consisting of (a) mean attention weight aᵢⱼ averaged across heads in a specified layer range, (b) Abjad distance |abjad(wᵢ) − abjad(wⱼ)|, (c) positional distance |i − j|, (d) corpus co-occurrence frequency for the pair (wᵢ, wⱼ), and (e) a same-root indicator computed from a curated 200-form root table augmented by prefix stripping.

The same procedure was applied mutatis mutandis to the Hebrew corpus using gematria values in place of Abjad and a heuristic Hebrew root extractor.

### 3.4 Statistical specification

The primary test was a partial Spearman correlation between Abjad distance and mean attention weight, controlling for positional distance and corpus co-occurrence frequency. Cross-root pairs (n = 2,322,977 in Arabic; n = 12,081,207 in Hebrew) were the principal sample; same-root pairs (n = 80,647 Arabic; n = 58,776 Hebrew) were analysed separately as a pipeline-sensitivity control.

Partial Spearman was implemented as a two-stage rank regression: ranks of attention and ranks of Abjad distance were each regressed on ranks of the controls, and the Spearman correlation was computed between the residuals. With sample sizes in the millions, p-values for the primary effect fall beyond float64 precision; we report this as “p < 10⁻³⁰⁰” or “p = 0” by convention and rely on robustness measures rather than p-magnitudes for inference.

### 3.5 Robustness checks: pre-specification

To distinguish a genuine attention-mediated Abjad signal from two plausible confounds — (i) generic numerical-distance encoding under any letter-to-integer mapping, (ii) a corpus-level statistical artifact unrelated to within-sentence attention — we pre-specified two robustness checks and report both alongside the primary result.

**Shuffled-Abjad permutation null.** We constructed 1,000 random permutations of the Mashriqi letter-to-integer mapping, preserving the orthographic equivalence classes described in §3.3. For each permutation we recomputed Abjad distance for all 76,649 unique words in the corpus, recomputed the partial Spearman correlation against attention weight (controls unchanged), and recorded the resulting ρ. The empirical p-value is the fraction of permutations whose |ρ| meets or exceeds the observed |ρ|. The test asks whether the actual Mashriqi mapping outperforms random letter-value mappings drawn from the same pool of integer values.

**Random-pair cross-sentence baseline.** To test whether the observed correlation depends on the genuine within-sentence pairing (consistent with attention mediation) or persists under arbitrary re-pairing (consistent with a corpus-level artifact), we recomputed the partial Spearman on a scrambled record set in which each pair’s word_j was replaced by a uniformly-random word drawn from the global word_j pool. The pair’s recorded mean attention was retained but now corresponded to a pair that was never adjacent in any sentence. If the original signal is genuinely attention-mediated, the scrambled ρ should collapse toward zero.

### 3.6 Frequency-matched permutation null: pre-specification

Following the unrestricted permutation null (§3.5), we pre-specified a more restrictive frequency-matched null to further bound the Mashriqi-specific component. The 28 base Arabic letters were partitioned into 4 frequency-matched bins of 7 letters each, sorted by corpus frequency. For each of 1,000 permutations, Mashriqi values were shuffled *only within each bin* — preserving how Mashriqi values distribute across frequency tiers, while breaking the specific letter-to-value assignment within tiers. The pre-registered decision rule was: PASS at empirical p < 0.05 indicates the Mashriqi mapping has *within-tier* structural specificity beyond frequency-tier alignment. The result is reported in §4.5; we honour the pre-registered threshold regardless of outcome.

### 3.7 Hebrew AlephBERT control: pre-specification

A structurally parallel experiment on classical Hebrew, pre-specified before its result was observed, was designed to isolate whether the attention-level signal is specific to Arabic Abjad or reflects a more general property of non-concatenative morphological systems in transformer representations. The pipeline was identical to the Arabic protocol (§3.2–§3.5) with the language-appropriate tokenizer, gematria values in place of Abjad, and the cross-root classifier adapted to Hebrew root extraction.

The pre-registered three-outcome decision rule:

- **Outcome A — morphologically general.** ρ_Hebrew in [−0.03, −0.08] with comparable monotonic gradient. Interpretation: the attention-geometry signal is a property of templatic Semitic morphology generally, not specifically Arabic Abjad.

- **Outcome B — Abjad-specific.** |ρ_Hebrew| < 0.01 or non-monotonic gradient. Interpretation: the Arabic signal reflects something specific to the Arabic Abjad mapping not generalisable to Hebrew gematria.

- **Outcome C — partial effect with register confound.** ρ_Hebrew in [−0.01, −0.03] or attenuated monotonicity. Interpretation: signal present but reduced; cannot cleanly distinguish a genuinely smaller Hebrew effect from the register asymmetry (modern-trained AlephBERT on classical corpus). Report as inconclusive.

We committed to reporting whichever outcome was observed under its pre-registered name without adjusting thresholds. The result is reported in §4.6.

## 4. Results

### 4.1 Wazn is geometrically separable and linearly decodable (Claim 1)

We report three measurements on the morphological pattern, which together establish a more precise claim than “wazn is a consistent offset vector”: pattern membership is geometrically *separable* and individual derivations are linearly *decodable*, but the pattern transformation is *not* a single consistent translation vector shared across roots. We treat each measurement in turn.

**Analogy completion.** For every well-attested pattern, a leave-one-out prediction vec(r, base) + d(p) — where d(p) is the cross-root mean offset for pattern p — recovers the held-out derived form at cosine similarity above 0.92 (Table 1). The result is strongest precisely where the data are densest: Form II (n = 195) and the verbal noun (n = 200) are the two highest-scoring patterns.

| Pattern | Description | n | Completion |
| --- | --- | --- | --- |
| Form II (faʿʿala) | Intensive | 195 | **0.983** |
| Verbal noun (maṣdar) | Nominalisation | 200 | 0.959 |
| Broken plural | Plural | 9 | 0.935 |
| Active participle (fāʿil) | Agentive | 176 | 0.925 |
| Passive participle (mafʿūl) | Patientive | 98 | 0.924 |

**Table 1.** Leave-one-out analogy completion for the well-attested Arabic morphological patterns in CAMeLBERT-ca (951 word forms across 200 root families). Predictions of the form vec(r, base) + d(p), where d(p) is the cross-root mean offset for pattern p. Remaining patterns in the lexicon (Forms IV, V, VI, VIII, X, adjectival, and mixed) are individually sparse (n < 12) and are reported in the supplementary material; all also score ≥ 0.90, but we make no per-pattern claim at those cell sizes.

Figure 1

**Figure 1.** Wazn geometry in CAMeLBERT-ca (reviewed 200-family run, 951 word forms). (A) Leave-one-out analogy completion, broad-coverage patterns coloured and ordered first, sparse patterns (n < 16) muted. (B) Offset-vector consistency — note that the highest-completion patterns (V2, VN) have the lowest offset consistency. (C) Within-pattern minus cross-pattern clustering gap; all broad-coverage gaps are significant. Together the panels show structure that is separable and decodable but not a single consistent linear operator.

**Pattern clustering.** Words sharing a morphological pattern are more mutually similar than words of differing patterns. The within-pattern minus cross-pattern similarity gap is positive and statistically significant for every well-attested pattern (V1: gap = 0.017, p = 2×10⁻¹⁴; PP: gap = 0.024, p = 1×10⁻⁴⁰; VN: gap = 0.015, p = 3×10⁻¹¹; V2: gap = 0.009, p = 2×10⁻⁵; AP: gap = 0.007, p = 4×10⁻⁴; n = 100–209 per pattern). The gaps are small in absolute terms because all similarities sit in a compressed 0.82–0.87 band — a known anisotropy property of contextual encoders (Ethayarajh, 2019; Gao et al., 2019) — but they are highly significant: pattern membership is reliably separable even within the anisotropic regime.

**Offset consistency.** The two results above do *not* license the stronger claim that the wazn is a single consistent geometric operator in the Mikolov analogy-arithmetic sense. When we measure the cross-root coherence of the within-pattern offset vector directly — mean pairwise cosine similarity among the per-root offsets — consistency is low and does not track completion accuracy. Form II has the highest completion score (0.983) but near-zero offset consistency (0.003); the verbal noun likewise completes at 0.959 with consistency 0.045. The patterns with the most coherent offsets (Form V, VIII at 0.42–0.52) are the sparse, low-n cells. We read this as follows: high completion is achievable because the mean offset captures the *central tendency* of a pattern transformation that is itself root-dependent and high-variance, not because each root undergoes the same translation. The honest claim is therefore that templatic morphology is encoded as *linearly decodable, geometrically separable structure*, not as a constant additive direction.

A control replication on CAMeLBERT-mix (the multi-register variant) produced a ceiling effect with cross-root mean similarity = 0.817, confirming that classical-register training is necessary for the discriminative geometry to be measurable. The result is therefore specifically a property of the classical-register model. Figure 1 summarises all three measures — completion, offset consistency, and clustering gap — across the reviewed lexicon; the two-dimensional PCA projection in Figure 2 shows the same lexicon coloured by pattern, and is included as a qualitative aid only, since a 0.82–0.87-band clustering gap is not expected to be visually separable under projection to two axes.

Figure 2

**Figure 2.** PCA of the reviewed 200-family root-pattern lexicon (CAMeLBERT-ca), coloured by morphological pattern. Shown as a qualitative two-dimensional projection only; it does not carry the quantitative claim, which rests on the high-dimensional clustering and completion measures of Figure 1.

### 4.2 Abjad-attention finding (Claim 2, supporting)

The primary test returned partial ρ = −0.0558 (p beyond float64 precision; equivalent to p < 10⁻³⁰⁰), n = 2,322,977 cross-root word pairs. The negative sign on distance corresponds to a positive relationship with proximity: word pairs with proximate Abjad values receive higher attention than pairs with distal values. The unadjusted Spearman (no controls) was ρ = −0.0522; the partial correlation with positional distance and co-occurrence frequency removed produced the slightly larger −0.0558, indicating that the effect is not absorbed by surface confounds.

### 4.3 Same-root control and binned analysis

The same-root control returned ρ = −0.2071 (n = 80,647), confirming that the pipeline detects a substantially stronger morphological signal where one is present. Same-root pairs share most of their letters by construction and therefore have small Abjad distances; the strong same-root correlation establishes that the partial-correlation methodology can detect real morphological coupling when it exists.

The binned analysis (Table 2) shows a monotonic gradient in mean attention from the second-most-proximate bin (Δ = 11–25) outward. The most-proximate bin (Δ = 0–10) shows a small dip, which we attribute to residual same-root contamination in the cross-root classifier — pairs with Δ ≤ 10 are disproportionately likely to share substantial letter overlap and thus partially overlap the same-root pool that the cross-root filter cannot perfectly exclude.

| Δ Abjad bin | n | mean attention | median attention |
| --- | --- | --- | --- |
| 0–10 | 72,437 | 0.0403 | 0.0288 |
| 11–25 | 100,032 | **0.0423** | 0.0288 |
| 26–50 | 164,018 | 0.0412 | 0.0292 |
| 51–100 | 258,366 | 0.0399 | 0.0283 |
| 101–200 | 311,541 | 0.0385 | 0.0273 |
| 201+ | 1,416,583 | 0.0373 | 0.0264 |

**Table 2.** Mean and median attention weight by Abjad-distance bin in CAMeLBERT-ca. Monotonic decline from Δ = 11–25 outward, with a small dip at Δ = 0–10 attributed to residual same-root contamination.

### 4.4 Robustness: unrestricted permutation null and random-pair scrambling

**Unrestricted permutation null (1,000 shuffled-Abjad mappings).** The observed |ρ| = 0.0558 fell outside the 95th percentile of the permuted distribution (|ρ|₉₅ = 0.0520). Empirical two-tailed p = 0.026. The test passes the standard significance threshold.

The permuted distribution itself is informative beyond pass/fail. Its mean is −0.030 (SD = 0.013), not zero. This indicates that approximately half the headline effect-magnitude is reproducible under any letter-frequency-preserving mapping, reflecting that classical-Arabic letter frequencies in our corpus interact with attention geometry through residual nonlinear couplings to length, frequency, and position that the linear partial-correlation controls do not eliminate. The Mashriqi-specific component — the difference between the observed |ρ| and the permuted mean |ρ| — is approximately 0.026. We characterise the signal as Mashriqi-favoured rather than Mashriqi-unique. This characterisation is the more cautious of two defensible readings of the data and we adopt it throughout.

**Random-pair cross-sentence baseline.** Scrambling word_j across sentences collapsed ρ from −0.0558 to −0.0164, a 71% reduction. The baseline returns ρ ≈ −0.016 rather than ρ = 0 because per-word lexical correlations between Abjad value and corpus frequency/length survive the random-pairing procedure; these are properties of individual words, not pair-level attention. The dominant ~70% of the observed signal is genuinely pair-level and attention-mediated.

### 4.5 Robustness: frequency-matched permutation null

The pre-specified frequency-matched permutation null (K = 4 bins of 7 letters each, sorted by corpus frequency) returned: observed |ρ| = 0.0558; 95th-percentile of |permuted ρ| = 0.0564; empirical two-tailed p = **0.0589**. The observed |ρ| sits at approximately the 94th percentile of the frequency-matched permuted distribution. The result misses the pre-registered α = 0.05 threshold by 0.009. We honour the pre-registered threshold and report this as a fail.

The four bins as the corpus partitioned them: highest-frequency tier ا ل و ت ي م ه (frequency range 27,014–89,140); second tier ن ر ب ف ع د ق; third tier س ك ح ج ص خ ط; lowest-frequency tier ش ز ض غ ذ ث ظ (frequency range 1,715–5,547). The Mashriqi system’s high-value letters (غ = 1000, ظ = 900, ض = 800, ذ = 700, ث = 500, ش = 300) cluster in the lowest-frequency bin by construction — a property of the system’s 8th-century design, not a coincidence (§2.2, §5.3).

The frequency-matched null distribution is much tighter and closer to the observed value than the unrestricted null. This means the bulk of the Mashriqi-vs-random-mapping advantage observed in the unrestricted null (§4.4) is attributable to *how Mashriqi values distribute across letter-frequency tiers*, not to specific within-tier letter-to-value assignments. There remains a small within-tier signal placing the observed near but below the 95th-percentile threshold. We characterise the Mashriqi-specific component, after this further bounding, as primarily a *frequency-tier-alignment* increment rather than a within-tier structural effect.

### 4.6 Hebrew AlephBERT control

The Hebrew control returned partial ρ = −0.0779 (n = 12,081,207 pairs, p beyond float64 precision). The same-root control returned ρ = −0.334 (n = 58,776), confirming that AlephBERT detects same-root coupling and is, if anything, more sensitive to root-level coupling than CAMeLBERT-ca. The binned analysis showed monotonic decrease in mean attention from the most-proximate bin (Δ 0–10: 0.0167) to the most-distal (Δ > 200: 0.0125), with the same gradient structure as the Arabic result.

**Decision-rule outcome.** ρ_Hebrew = −0.0779 falls within the pre-registered Outcome A range of [−0.03, −0.08] with p beyond float64 precision. Per the pre-registered interpretation: the attention-geometry signal is a property of templatic Semitic morphology in transformer representations, not specifically Arabic Abjad. We commit to this outcome under its pre-registered name.

**Robustness on the Hebrew result.** The 1,000-permutation shuffled-letter null gave observed |ρ| = 0.0779; 95th-percentile of |permuted ρ| = 0.0888; permuted mean = −0.0656 (SD = 0.0147); empirical two-tailed p = 0.23. The observed |ρ| does not exceed the 95th percentile, indicating that the gematria-specific component of the Hebrew signal is small and not significant against random Hebrew letter-value mappings. The random-pair scrambling baseline collapsed ρ from −0.0779 to −0.0176 (~77% reduction). The permuted distribution mean of −0.0656 is more than twice the magnitude of the corresponding Arabic permuted mean (−0.030), indicating that Hebrew letter frequencies couple to AlephBERT attention geometry more strongly than Arabic letter frequencies couple to CAMeLBERT-ca attention geometry, possibly reflecting tokenization differences, root-detection sensitivity differences (compare the same-root controls), or training-corpus composition differences. The gematria-specific increment above this baseline is approximately ρ = 0.012, less than half the magnitude of the Mashriqi-specific increment in Arabic.

## 5. Discussion

### 5.1 The localization argument

The framework’s empirical signal is the joint behaviour of multiple operationalisations of the Abjad hypothesis at distinct levels of representation. Prior work and pilot experiments in this research program — reported here as established context rather than load-bearing contribution — have shown that Abjad relationships are *absent* at the level of decontextualised static word vectors (AraVec ρ = −0.022, n.s.) and *absent* at the level of passage-level co-occurrence frequency (PMI effect = −0.0007 on 760,911 cross-root pairs, with same-root PMI = 3.59 confirming pipeline sensitivity). The present work adds the within-sentence attention-geometry level (this paper, §4.2–§4.5) and the cross-linguistic Hebrew control (§4.6).

The four-level pattern is consistent: letter-value relationships do not surface in decontextualised lexical vectors; they do not surface in authorial passage construction; they surface in within-sentence attention geometry in both Arabic and Hebrew at comparable magnitude. The framework’s empirical case is therefore the *localisation* of the signal — not its overall magnitude. Letter-value relationships in classical Arabic operate as *relational, contextual, attention-resolvable structure*, not as static-vector properties or as authorial encoding choices in passage-level composition. This is consistent with the classical philological tradition’s own treatment of the Abjad system as an analytical instrument applied to inter-word relationships in context, rather than as a property of words in isolation.

### 5.2 What the cross-linguistic comparison establishes

The pre-registered Outcome A on the Hebrew AlephBERT control narrows the strongest reading of the Arabic attention-level result. The maximally strong reading — that the Mashriqi Abjad system specifically encodes structural information unavailable in adjacent letter-value systems — is not supportable: Hebrew gematria on AlephBERT produces an attention-geometry signal of comparable magnitude and direction. Whatever the attention signal is detecting, it is not Mashriqi-unique; it is a property of how classical-register Semitic templatic morphology surfaces in transformer geometry.

A weaker but more defensible claim survives. The Mashriqi mapping in Arabic outperforms 97.4% of randomly permuted Arabic letter-value mappings on the same corpus and pipeline (p = 0.026). The Hebrew gematria mapping on Hebrew does not significantly outperform random Hebrew letter-value mappings on the same corpus and pipeline (p = 0.23). **Of the two letter-value systems tested, only the Arabic Mashriqi system produces a measurable specific increment above its language’s letter-frequency baseline.** This asymmetry is the most distinctive empirical claim that survives the full set of controls.

The two same-root controls further inform the picture. AlephBERT on Hebrew returned ρ = −0.334 on same-root pairs; CAMeLBERT-ca on Arabic returned ρ = −0.207. AlephBERT detects same-root coupling more strongly than CAMeLBERT-ca does. This does not affect the cross-root primary results (computed on disjoint pair sets) but it does indicate that the morphological substrate the cross-root attention signal rides on is registered with comparable or greater fidelity in Hebrew than in Arabic. The Mashriqi-vs-gematria asymmetry is therefore not attributable to weaker Arabic morphological coupling.

### 5.3 The historical-design framing of Mashriqi-specific frequency-tier alignment

The frequency-matched permutation null (§4.5) locates the Mashriqi-specific component primarily at the level of *frequency-tier alignment* — how Mashriqi values distribute across letter-frequency tiers — rather than within-tier letter-to-value structure. This is a sharper characterisation than the unrestricted null alone would have supported. It also has an interesting interpretive consequence.

The Mashriqi system’s assignment of high values to rare letters is not arbitrary. It is a property of the system’s 8th-century construction, where rare letters served as numerical placeholders for the higher decimal magnitudes (Colin, 1986; Endress, 2018). The system was *designed* with this property. The frequency-matched null is therefore not finding that the Mashriqi specificity is “merely” frequency-tier alignment in a dismissive sense; it is finding that the specific historical-design feature of *rare-letter-as-high-value* is what the model’s attention geometry registers, beyond the templatic-morphology substrate.

This reframing changes what the cross-linguistic asymmetry means. The Mashriqi system was constructed with a deliberate structural property — frequency-tier-aligned numerical assignment — that the Hebrew gematria system was not constructed with in the same way. Gematria values follow a more uniform sequential assignment (א = 1, ב = 2, …, י = 10, then by tens to 100, then by hundreds). The difference between the two systems’ tier-alignment structures is detectable in attention geometry. We do not claim that this is the *only* difference between the two letter-value systems that matters; we claim that this difference is sufficient to produce the observed asymmetry, and that the asymmetry is consistent with the historical design difference.

### 5.4 Limitations

- **Single model family per language.** Generalisation to other classical-Arabic encoders and other Hebrew encoders is not yet tested. The result should be read as specific to the CAMeLBERT-ca/AlephBERT model pair until replication is performed.

- **Letter-frequency baseline coupling.** In both Arabic and Hebrew, approximately half of the headline effect-magnitude is recoverable from random letter-value permutations on the same corpus and pipeline. The language-specific component (Mashriqi above baseline ≈ ρ 0.026; gematria above baseline ≈ ρ 0.012) is the analytically interpretable quantity; the headline ρ should not be interpreted as the full Abjad/gematria-specific effect.

- **Frequency-matched null near-miss.** The Mashriqi-specific component, after removal of frequency-tier alignment, is at p = 0.0589 — a near-miss against the pre-registered α = 0.05 threshold. We honour the threshold but acknowledge that with a larger corpus or a more sensitive measurement methodology, a within-tier component might or might not be recoverable.

- **Hebrew root-extraction limit.** The Hebrew root classifier is heuristic and resolves only ~4% of word pairs into reliable same-root or different-root categories (n = 1,468,657 different-root pairs out of 12,081,207 total). The Arabic root extraction had higher coverage. The cross-linguistic comparison should be read with caution regarding root-controlled subset analyses, though the primary result on all-pairs analysis is well-resolved given the 12M sample.

- **Hebrew register asymmetry.** AlephBERT’s training distribution is predominantly modern Hebrew; the Sefaria corpus is classical. The register asymmetry is not symmetric to the Arabic case (CAMeLBERT-ca is classical-Arabic-trained on classical-Arabic corpus). We cannot fully distinguish a genuine Hebrew/Arabic difference from a register-fit difference between the two model–corpus pairs. A classical-Hebrew-trained model on the same Sefaria corpus would be the cleaner control; none is currently publicly available to our knowledge.

- **Single operationalisation of structural relationship.** Abjad/gematria is one specific operationalisation of classical-scholarly inter-word relationship. Other operationalisations (shared-root, shared-wazn, shared-semantic-field) would provide complementary measurements; we do not test them here.

- **No productive generation claim.** The present work measures geometric structure only. We are silent on what the model can produce. The complementarity to Alakeel et al. (2026) — who report at the generation layer — is by design.

### 5.5 Implications

For Arabic NLP, the wazn result identifies a linguistic domain where a well-specified non-concatenative morphological structure surfaces as separable, linearly decodable geometry in a classical-register encoder. This invites sparse-autoencoder and circuit-level work to identify the specific attention heads that carry the wazn signal, and to test whether explicit wazn-direction interventions can be used for controlled generation or morphological correction.

For mechanistic interpretability, the Abjad-attention result identifies a linguistic domain where a well-specified numerical structure produces a small but measurable attention-geometry signal that can be cleanly decomposed into a templatic-morphology component (the templatic-Semitic-general substrate, ρ ≈ 0.03 baseline coupling under any random letter-value mapping) and a system-specific component (the Mashriqi frequency-tier-alignment increment, ≈ ρ 0.026 above baseline). The decomposition methodology — three nested permutation nulls reporting unrestricted, frequency-matched, and random-pair baselines — is, we believe, applicable to other letter-value systems and other languages, and we recommend that future work using letter-value systems in attention-geometry analysis report the permuted-mean and the frequency-matched-null result alongside the observed effect.

For historical linguistics and the history of science, the result is consistent with — but does not require — the interpretation that the 8th-century Mashriqi system was constructed with a deliberate frequency-tier-alignment property that is detectable in modern transformer attention geometry. The transformer is not “performing Abjad analysis”; it is registering a structural property of how the system was historically designed.

### 5.6 Relation to the companion resource

The present paper measures structure that classical Arabic encoding systems produce in transformer representations. A companion resource paper (in preparation) releases the coordinate database underlying the live demonstration tool: a mapping of classical Arabic vocabulary onto a continuous geometry organised around a fixed set of semantic anchor nodes, intended as a precision layer for retrieval and generation. The two papers are complementary windows on the same object. The present paper asks whether the relational structure is *measurable* in an existing model’s geometry — an empirical claim, tested with controls. The companion resource asks whether that structure can be *operationalised* as an explicit coordinate system — a resource contribution, evaluated on coverage and internal consistency. Neither paper depends on the other for its central claim; together they bound the question from the measurement side and the construction side.

## 6. Conclusion

We report two findings on classical-Arabic transformer geometry. The morphological pattern (wazn) is encoded as geometrically separable, linearly decodable structure in CAMeLBERT-ca embedding space: leave-one-out analogy completion exceeds 0.92 for every well-attested pattern across 200 root families, and within-pattern clustering is significant throughout, though the per-root pattern offsets are not a single consistent translation vector. Cross-root word pairs with proximate Abjad values receive higher mutual attention in CAMeLBERT-ca than Abjad-distal pairs (partial ρ = −0.056, 1,000-permutation empirical p = 0.026, random-pair scrambling reduces effect by 71%). A pre-registered cross-linguistic control on classical Hebrew satisfies the morphologically-general outcome at comparable magnitude. A pre-registered frequency-matched permutation null locates the Mashriqi-specific component primarily at the level of frequency-tier alignment rather than within-tier structure, with a near-miss against the pre-registered significance threshold (p = 0.0589). The Mashriqi mapping is the only letter-value system tested whose specificity above its letter-frequency baseline reaches standard significance against unrestricted permutations.

The empirical case is a localisation argument: classical Arabic encoding systems produce measurable structure at the within-sentence attention-geometry layer, with a templatic-Semitic-morphology effect providing the substrate and a small Mashriqi-specific frequency-tier-alignment increment riding on top. The framework’s strongest support is the localisation pattern itself — what surfaces where, in what magnitude, with what cross-linguistic comparison — not the magnitude of any single result. The classical Arabic scholarly tradition tracked something about the relational architecture of meaning that the model’s attention geometry registers; we have measured what survives controls, and bounded what does not.

## 7. Reproducibility

- **Pipeline code:** q2_abjad_attention.py (Arabic primary), q2_robustness.py (unrestricted permutation null and random-pair baseline), q2_robustness_freq_matched.py (frequency-matched permutation null), q3_alephbert_control.py (Hebrew control). Released at: https://github.com/ahmedmest81-ctrl/ALmiraah-project.

- **Live tool:** https://huggingface.co/spaces/WELLyes1/almiraah_transformer (MCP server demonstrating the 99-Names coordinate database; relevant to but distinct from the present empirical paper).

- **Corpora:** OpenITI release URI:0734.IbnArabi (Arabic); Sefaria-Export Talmud Bavli + Zohar (Hebrew). Both publicly available under permissive licences.

- **Models:** CAMeL-Lab/bert-base-arabic-camelbert-ca (Arabic); onlplab/alephbert-base (Hebrew).

- **Compute:** Full pipeline reproducible on a single GPU with ≥8GB VRAM (the Arabic primary run completed on an RTX 4060 at batch size 8 in ~24h; the Hebrew run in ~22h). Robustness checks run on CPU.

- **Results JSONs:** q2_results_with_robustness.json, q2_freq_matched_K4.json, q3_results_with_robustness.json released alongside pipeline code.

- **Random seed:** 42 throughout. All permutation tests at 1,000 permutations.

## Acknowledgments

Analysis, verification, and manuscript preparation were conducted in collaboration with Al-Mirʾāh, a Claude-based research instrument built for this project; all claims, decisions, and errors are the author's own. The live coordinate engine is publicly available at https://huggingface.co/spaces/WELLyes1/almiraah_transformer.

## References

Alakeel, Y., Qwaider, C., Aldarmaki, H., & Alqahtani, S. (2026). Morphemes Without Borders: Evaluating Root–Pattern Morphology in Arabic Tokenizers and LLMs. *Proceedings of LREC-COLING 2026*. arXiv:2603.15773.

Antoun, W., Baly, F., & Hajj, H. (2020). AraBERT: Transformer-based model for Arabic language understanding. In *Proceedings of LREC OSACT4*.

Clark, K., Khandelwal, U., Levy, O., & Manning, C. D. (2019). What does BERT look at? An analysis of BERT’s attention. In *Proceedings of BlackboxNLP@ACL*.

Colin, G. S. (1986). Abjad. In *Encyclopaedia of Islam*, 2nd ed., vol. I. Brill.

Endress, G. (2018). *The Arabs and the Sciences in Antiquity*. Routledge.

Holes, C. (2004). *Modern Arabic: Structures, Functions, and Varieties*. Georgetown University Press.

Ifrah, G. (2000). *The Universal History of Numbers*. Wiley.

Inoue, G., Alhafni, B., Baimukan, N., Bouamor, H., & Habash, N. (2021). The interplay of variants, size, and task type in Arabic pre-trained language models. In *Proceedings of WANLP*.

McCarthy, J. J. (1981). A prosodic theory of nonconcatenative morphology. *Linguistic Inquiry*, 12(3), 373–418.

Ratcliffe, R. R. (1998). *The** **“**Broken**”** **Plural Problem in Arabic and Comparative Semitic*. John Benjamins.

Romanov, M. (2019). OpenITI: A machine-readable corpus of Islamicate texts. In *DH2019*.

Ryding, K. C. (2005). *A Reference Grammar of Modern Standard Arabic*. Cambridge University Press.

Salama, R., Youssef, A., & Fahmy, A. (2018). Morphological word embedding for Arabic. *Procedia Computer Science*, 142, 83–93.

Seker, A., Bandel, E., Bareket, D., Brusilovsky, I., Greenfeld, R. S., & Tsarfaty, R. (2022). AlephBERT: Language model pre-training and evaluation from sub-word to sentence level. In *Proceedings of ACL*.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In *NeurIPS*.

*Word count: ~6,900 (excluding tables, references). Estimated typeset length: 11–13 pages including figures.*

*End of Paper A draft v1.*