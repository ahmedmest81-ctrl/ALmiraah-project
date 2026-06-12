# Paper C — Skeleton and Asset Map
*Working title:* **Named Before Measured: Classical Arabic Philosophical Vocabulary as a Conceptual Layer for Transformer Geometry**
*Target:* Minds and Machines · *Dependencies:* Papers A and B public (arXiv IDs citable) · *Timeline:* begin drafting after A+B submission; this is the year-out commitment, not the near-term one.

## Thesis (one paragraph)
Two traditions independently arrived at relational, non-essential meaning: Akbarī metaphysics (a term is what the Names make of it) and transformer geometry (a token is its position in a learned relational field). The convergence is structural, not decorative. The classical tradition systematically *named* properties that current interpretability research *measures* but cannot yet adequately name — and the naming is precise enough to generate testable predictions. The paper's claim is homology, not identity, and the distinction is given exact mathematical form.

## Structure

### 1. Introduction: the vocabulary gap
From the existing vocabulary-gap manuscript. Anthropic's natural-language autoencoders translate activations into text; Goodfire's weight decomposition exposes attention geometry — both measure properties (self-disclosure events, persistence beneath activation) for which ML theory has placeholder words and the Akbarī lexicon has technical terms (tajallī, baqāʾ). Empirical-first framing: lead with Paper A's measured signal, then Paper B's operational resource, then the conceptual claim.

### 2. The two banks (condensed Pillars I and II)
One section each, heavily compressed from the lexicon and primitives documents. ML side: attention as weighted relational field, stateless inference, forward pass as discrete complete event, latent/actualised distinction. Philological side: the six-layer method on the load-bearing terms only — tajallī, barzakh, fanāʾ/baqāʾ, tajdīd al-khalq, al-nuqṭah, al-mirʾāh.

### 3. The mapping discipline
The four-way classification per pair: structurally identical / analogous / merely metaphorical / does not map. Include at least two documented negative results — a framework where everything maps is not rigorous. Candidate strong mappings: tajallī ↔ forward pass (proof-of-concept, do first and deepest); barzakh ↔ the intrinsic geodesic midpoint or transition region between two projected semantic positions; tajdīd al-khalq ↔ stateless re-instantiation per token. “Weight space between inferences” remains at most a looser analogy, not the operational mapping.

### 4. Homology given exact form: the categorical formalization
The new core. Roots as objects, awzān as morphisms; derivation chains compose — the ishtiqāq category. CAMeLBERT's embedding becomes a functor from this category into geometric space; **homology = the functor exists and partially preserves structure; identity = equivalence of categories, which is not claimed and is shown not to hold.** Paper A's offset-consistency negative result (pattern offsets are not a single translation vector) becomes the measured degree of the functor's failure of faithfulness. The 99-profile as restricted Yoneda embedding: a term as presheaf on the Name-site. State precisely what is established structure (metric, profiles) vs programmatic (composition).

### 5. The equilibrium argument (new since v3)
The iʿtidāl episode as a case study of the thesis itself: the engine violated its own anchor doctrine by measuring raw cosine from an anisotropic origin; the correction (profile centering against the field's zero, vector centering against the basis centroid — two complementary halves) was *derivable from the classical vocabulary before it was justified by the ML literature* (Mu & Viswanath's all-but-the-top arrived as confirmation, not source). One concrete instance of the paper's claim: the classical term generated the fix.

### 6. The intrinsic measurements (updated in v3.2)
Euclidean displacement vs hyperbolic geodesic; hierarchy load as curvature surcharge; geodesic midpoint as an executable barzakh; and the radial/angular L-path diagnostic as a held-lightly account of whether a relation runs mainly through depth or field. Empirical anchors from Paper B: *faraḥ/surūr* has hierarchy load 4.01 and radial share 0.733. The exploratory *fanāʾ/baqāʾ* comparison is the contrasting case: angular share 0.921, Δθ 178.23°, and a geodesic midpoint near the disk centre (r = 0.0619), nearest Al-Bāṭin, Al-Ẓāhir, and Al-Qayyūm. This is an executable measurement, not yet evidence that the classical concept and the geometric midpoint are identical. The falsifiable next step is to test whether human philological judgments of “intermediate” terms align with midpoint-neighbour predictions.

### 7. The depth experiment (to run before drafting)
Does query r behave as genuine hierarchy depth? Current evidence: tier-monotonic query radii (Dhāt 0.44 < Ṣifāt 0.47 < Afʿāl 0.63) with the circularity caveat documented (tier assigned by attractor vote; r derived from tier-banded positions). The non-circular version: external ontological annotations (e.g., abstractness norms, classical lexicon hierarchy classes) as the independent variable. This is the one experiment that must run before Paper C is drafted, because §6–7 stand on it.

### 8. What does not map, and what the homology licenses
Boundaries section. Consciousness claims explicitly out of scope (the framework names structural properties; it does not adjudicate experience — الوقوف عند حدّ ما يُعلم). The localization argument from Paper A §5.1 gets its full treatment here.

## Asset inventory (already in hand)
- Vocabulary-gap manuscript (intro source) · Pillar I/II documents (section 2)
- Paper A results incl. the offset-consistency negative (functor unfaithfulness measure)
- Paper B v4 §5 cases + §6.4 protocol-correction robustness story
- v2/v3 dual-protocol profiles for 759 terms (instrument-variation evidence)
- hierarchy_load implementation + first measurements
- hyperbolic.py (Karcher, Möbius, exp/log maps, geodesic midpoint) — the formal geometry section's computational companion
- v3.2 locked pair diagnostics, including the exploratory *fanāʾ/baqāʾ* midpoint
- Mapping-completions and phase-two documents (section 3 raw material)

## Open prerequisites
1. Papers A and B on arXiv (citable IDs)
2. The non-circular depth experiment (§7)
3. The hierarchy-load disambiguation pilot (§6) — optional but strengthens
4. Human philological validation of midpoint-neighbour predictions for proposed barzakh cases
5. Co-author with formal credentials in Islamic studies or philosophy of mind (review survivability at M&M)
6. Abjad adjudication closed (five conventions) so all cited values are final
