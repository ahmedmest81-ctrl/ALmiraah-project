# The 99-Names Coordinate System: A Fitted Relational Basis for Classical Arabic in CAMeLBERT-ca

*A scholarly relational structure as a coordinate basis for fine-grained semantic analysis of classical Arabic terms*

Ahmed Mislati Independent researcher, Vienna arXiv preprint cs.CL · June 2026

**Keywords:** Arabic NLP, classical Arabic, fitted basis, contextual embeddings, CAMeLBERT, semantic resource, near-synonym disambiguation, Poincaré embedding, LREC, language resource

## Abstract

Distributional similarity tools (cosine similarity, k-nearest neighbours, contextual embedding distance) cannot reliably resolve fine-grained relational distinctions in Arabic vocabulary. Two terms English-glossable as “joy” (فرح *faraḥ* and سرور *surūr*) cluster near each other in any contextual embedding space, but classical Arabic philology treats them as structurally distinct: *faraḥ* names assertive elated emotion, *surūr* names tranquil contentment. Two speech-acts both glossable as “I love you” (أحبك *uḥibbuka* and أعشقك *aʿshaquki*) function in markedly different registers, but a distributional model treats them as paraphrases. The problem is not a defect of contextual embeddings; it is that distributional similarity does not encode the relational properties — root families, morphological patterns, hierarchical tier, paired-opposite structure — that classical scholarly practice formalised over a millennium.

We present a methodology and resource for incorporating a documented scholarly relational structure as a *fitted coordinate basis* on the contextual embedding space of CAMeLBERT-ca. The basis structure is the 99 Names of God in classical Islamic tradition: a closed, well-attested set with hierarchical, morphological, root-family, and selected documented opposition relations. The released annotations also contain explicitly labeled interpretive extensions beyond those documented relations. We treat the 99 Names as a coordinate-defining relational structure, not as a theological commitment. Arbitrary Arabic terms are projected into the basis via CAMeLBERT-ca contextual embeddings followed by Poincaré-disk geometric reduction. The resulting coordinate system supports five operations: philological lookup, root-family analysis, semantic projection, semantic neighbour retrieval, and structural comparison.

We release: (i) the actual 99-entry basis source used by the deployed engine, containing 30 fields per entry with field-level status labels separating engine-facing descriptions, provisional Abjad annotations, documentary/morphological notes, and framework-interpretive annotations; (ii) the fitted 99-node Poincaré coordinates and machine-readable dataset metadata; (iii) the coordinate engine code; and (iv) a live MCP server exposing the five operations. The 99-entry basis source is distinct from the separately released 759-record accumulated query dataset. We demonstrate the methodology on six empirical cases drawn from emotion, cognition, and morality vocabulary. A measured baseline establishes the core difficulty: under the anisotropy of CAMeLBERT-ca contextual embeddings, all four demonstration pairs — joy near-synonyms, love speech-acts, silence/grief, anger/fear — receive within-pair cosine similarities compressed into a narrow 0.865–0.937 band, so that cosine magnitude carries no information about the *kind* of relationship each pair has. Against this baseline, the coordinate system recovers the distinction structure: the love speech-acts (cosine 0.921) share four of five attractors, with the single divergent attractor carrying the classical ḥubb/ʿishq distinction; the joy near-synonyms (cosine 0.937) share three of five attractors and differentiate at depth, with the highest hierarchy load of the set (4.01); and the two cross-category pairs (silence/grief at 0.885; anger/fear at 0.865), which cosine scores within 0.05 of the near-synonyms, separate almost completely — one of five attractors shared and the largest hyperbolic distances of the set (up to 0.711). Differentiation appears where philology expects it: in divergent attractors for kin pairs, in wholesale profile separation for categorical pairs, and in hierarchy load for distinctions constituted at depth. The methodology is generalisable: any documented scholarly relational structure in any language could in principle function as a fitted basis the same way.

## 1. Introduction

### 1.1 The problem

Contextual embeddings produced by transformer encoders such as CAMeLBERT-ca (Inoue et al., 2021) capture distributional regularities in classical Arabic vocabulary at the level of token co-occurrence and contextual prediction. Two terms used in similar contexts will receive geometrically similar contextual embeddings, and cosine similarity between their representations will be high. This property supports a wide range of downstream tasks including retrieval, classification, and translation. It does not, however, capture the *relational* structure that classical Arabic philology has formalised: which root family a term belongs to, what morphological pattern it instantiates, what numerical (Abjad) value it carries, and — for terms in the rich emotional, cognitive, moral, and theological vocabulary of classical Arabic — which structural relationships it has with other terms in the same domain.

This omission produces concrete failures. Consider the Arabic emotional vocabulary for “joy”: فرح *faraḥ*, سرور *surūr*, بهجة *bahja*, طرب *ṭarab*, إنشراح *inshirāḥ*. A dictionary glosses all five as “joy” or “happiness.” A contextual embedding model places all five in a tight cluster. But classical Arabic philology distinguishes them precisely: *faraḥ* is assertive and outward; *surūr* is tranquil and contained; *bahja* is celebratory; *ṭarab* is musical/ecstatic; *inshirāḥ* is the breast-opening of relief. These distinctions matter for translation, classical Arabic NLP applications, and any task requiring fine-grained semantic discrimination.

The problem generalises beyond near-synonym disambiguation. Speech-acts — أحبك *uḥibbuka* (“I love you” in a stable affectional register) versus أعشقك *aʿshaquki* (“I love you” in a passionate-obsessive register) — function differently in social practice but cluster identically in distributional space. Antonyms such as حب *ḥubb* (love) and كره *kurh* (hatred) sit closer to each other than either does to indifference, a fact classical philology has documented but distributional tools fail to register. Terms with deep philological history — ظلم *ẓulm*, etymologically “putting in the wrong place,” classically used for both oppression and ontological error — carry structural information that a distributional model cannot recover.

### 1.2 The contribution

We present a methodology and resource for incorporating a documented scholarly relational structure as a *fitted coordinate basis* on CAMeLBERT-ca contextual embeddings. The methodology is structured around three elements:

- A **basis structure** — a closed, well-attested, internally relational set of terms with structural annotations (hierarchical tier, selected documented opposition relations, root families, and morphological patterns). The basis is fixed; it does not change with query.

- A **fitting procedure** — for each basis term, a CAMeLBERT-ca contextual embedding is computed, and the basis is reduced to a Poincaré disk via geometric projection that preserves hierarchical relationships.

- A **projection procedure** — for an arbitrary query term, CAMeLBERT-ca contextual embedding is computed, similarities against the basis are calculated, and the term is located in the disk according to its attractor and repellor profile against the basis.

The result is a coordinate system: every Arabic term receives a position (px, py, r), a hierarchical tier classification (Dhāt / Ṣifāt / Afʿāl), an attractor set (which basis terms it is structurally similar to), and a repellor set (which basis terms it is structurally opposed to).

Our specific basis is the **99 Names of God** in classical Islamic tradition. We treat this set as a coordinate-defining relational structure, not as a theological commitment. The 99 Names satisfy the basis requirements: closed (n = 99), well-attested (documented continuously since the 8th century), internally relational (with selected opposition relations such as Al-Bāsiṭ ⇄ Al-Qābiḍ), and structurally annotated with root, pattern, hierarchy, and semantic descriptions. The release metadata distinguishes source-oriented descriptions from provisional numerical and framework-interpretive annotations. The basis is functioning as a methodological instrument; the methodology generalises to any analogous structure in any language.

### 1.3 What we contribute

This paper makes four contributions:

- A methodology for using a documented scholarly relational structure as a fitted coordinate basis on contextual embeddings, applicable to any language and any analogously-structured basis.

- The exact 99-entry, 30-field basis source used by the deployed engine, released with the fitted 99-node Poincaré coordinates, checksums, a schema validator, and metadata that labels evidential status field by field. This basis resource is explicitly distinguished from the 759-record accumulated query dataset.

- A live deployment — a coordinate engine accessible via an MCP server — exposing five operations: philological lookup, root-family analysis, semantic projection of cross-linguistic candidates, semantic neighbour retrieval, and pairwise structural comparison.

- Six empirical case studies demonstrating that the methodology produces structural distinctions a CAMeLBERT-ca baseline cosine similarity cannot make, drawn from emotion vocabulary (joy near-synonyms; love speech-acts), structural-relational vocabulary (silence/grief, anger/fear), and morally-loaded vocabulary (oppression, intimacy).

### 1.4 What we do not claim

We do not claim the 99 Names are *the* correct basis for classical Arabic. Other documented relational structures (the 28-letter Arabic alphabetical hierarchy, the seven mu’allaqāt poetic structure, the 99-pole Sufi nafs/qalb/rūḥ taxonomy) would function as alternative bases the methodology supports equally well. We do not claim the coordinate system captures the *full* relational structure of classical Arabic; we claim it captures the relational structure of the 99-Name basis specifically, and demonstrates substantive distinctions distributional similarity cannot make. We do not claim theological commitments; the methodology is reproducible by a researcher with no Islamic-studies background, using only the released dataset and code.

## 2. The 99-Names as a relational structure

### 2.1 Background

The 99 Names of God (*al-asmāʼ al-ḥusnā*, “the most beautiful Names”) are a canonical set of divine attributes documented in classical Islamic tradition since approximately the 8th century. Each Name names a divine function or quality (Al-Raḥmān, “the All-Merciful”; Al-Jabbār, “the Compeller”; Al-Ḥakam, “the Judge”). The set is closed; the standard enumeration of 99 has been stable across classical scholarly tradition for over a millennium (al-Ghazālī, 1095/1992; Ibn ʿArabī, 1240/2004).

For the methodological purposes of this paper, three properties of the 99-Name structure are operationally important.

**Hierarchical tier.** The resource operationalizes a three-tier reading: **Dhāt** (essence-level), **Ṣifāt** (attribute-level), and **Afʿāl** (action-level). The fitted disk stores the operational 0–2 tier labels used by the engine. These labels are resource annotations motivated by the classical hierarchy; they have not undergone independent annotation adjudication.

**Paired opposites.** Classical tradition documents specific axes of equilibrium, including Al-Qābiḍ ⇄ Al-Bāsiṭ (constriction/expansion), Al-Muʿizz ⇄ Al-Mudhill (honor/humiliation), and Al-Rāfiʿ ⇄ Al-Khāfiḍ (elevation/lowering). The released `paired_opposite` field is broader than this documented core: it is free text and includes interpretive or theoretical opposites, which the metadata labels accordingly.

**Root families.** The basis contains repeated root families central to classical Arabic theological vocabulary. For example, ر-ح-م appears in Al-Raḥmān and Al-Raḥīm; ح-ك-م in Al-Ḥakam and Al-Ḥakīm; and غ-ف-ر in Al-Ghaffār and Al-Ghafūr. These repeated families provide natural neighbourhoods for testing root-sensitive structure in coordinate space.

### 2.2 Why this structure works as a basis

A coordinate basis for fine-grained semantic analysis needs four properties: **closure** (a fixed set, not query-dependent), **attestation** (documented existence independent of the present work), **internal relational structure** (the basis terms have known relationships among themselves), and **coverage** (the basis terms touch the semantic domains being analysed). The 99 Names satisfy all four for classical Arabic emotional, moral, cognitive, and theological vocabulary. They are closed at n = 99. They are documented continuously since the 8th century in scholarly sources independent of NLP. They have known paired-opposite, tier, and root-family structure. And — critically — they cover the semantic domains where distributional similarity tools most often fail: the rich emotion-cognition-moral vocabulary of classical Arabic where English glosses collapse fine distinctions.

We are not claiming the basis is *uniquely* correct, nor that it captures *all* structural relations in classical Arabic. We are claiming it is a *functional* basis for the methodology, with documented properties that make it more suitable than alternatives we have considered (e.g. the 28-letter alphabetic hierarchy lacks paired-opposite structure; the seven mu’allaqāt lack coverage of theological-moral vocabulary).

## 3. Methodology

### 3.1 Fitting the basis to CAMeLBERT-ca

The basis fitting procedure produces, for each of the 99 Names, a CAMeLBERT-ca contextual embedding **v** ∈ ℝ⁷⁶⁸. We embed each Name in three carrier-sentence contexts (the practice established in Paper A §3.2) and average the resulting layer-8 hidden states. The carrier sentences are templated: *“**الكلمة هي [Name]**”*, *“**يقول الشيخ [Name]**”*, *“**معنى [Name] هو**”*. Averaging across carriers reduces context-specific noise and produces a representation closer to the Name’s de-contextualised meaning while preserving the contextual encoder’s awareness of register.

The resulting 99 × 768 embedding matrix is reduced to a 2-dimensional Poincaré disk via the following procedure. We compute a 99 × 99 similarity matrix S where S_ij = cosine(v_i, v_j). For each Name we identify its k = 5 nearest neighbours by S and its k = 3 most-repelled basis terms by minimum S. The Poincaré-disk position (px, py, r) is then computed by gradient descent on a stress function that places similar Names close in Euclidean distance, repelled Names on opposite sides of the disk, and tier (Dhāt / Ṣifāt / Afʿāl) hierarchically by r (Dhāt at small r, Afʿāl at large r). Convergence is achieved in approximately 500 gradient-descent steps and is stable across random initialisations.

### 3.2 Projecting arbitrary query terms

For an arbitrary Arabic query term q, the projection procedure follows the same embedding methodology as the basis fitting (3 carrier sentences, layer-8 mean-pooled embeddings) to obtain v_q ∈ ℝ⁷⁶⁸. We then compute the 99-dimensional similarity vector s_q = [cos(v_q, v_i)]_{i=1..99}. The query’s coordinate is determined by:

- **Top attractors** — the basis Names with highest cosine similarity. Default k = 5.

- **Structurally absent / repelled** — the basis Names with most-negative cosine similarity. Default k = 3.

- **Hierarchical tier** — assigned by majority vote among the top-5 attractors weighted by similarity.

- **Position (px, py, r)** — by interpolation in the fitted disk, computed as the similarity-weighted **Karcher mean** (Riemannian barycenter) of the top-5 attractor positions. Earlier engine versions used a Euclidean weighted centroid; on a 20-term validation set this under-estimated r by a mean of 0.031 and changed the tier assignment of 4 of 20 terms, so the intrinsic barycenter is used throughout (§6.4).

- **Morphological pattern** — the query term’s own wazn, parsed by positional radical substitution against a template inventory (the classical miqyās method). Where the unvocalised surface form admits multiple patterns, the engine returns all candidates with status `ambiguous_vocalization` rather than guessing; the attractor cluster’s prevailing wazn is reported separately as `cluster_wazn` and is never substituted for the query’s own pattern.

Pairwise comparison additionally reports **two distances**: the flat Euclidean displacement in disk coordinates and the hyperbolic geodesic, together with their ratio (*hierarchy load*), which equals 2.0 at the disk origin and grows toward the rim. The ratio isolates how much of a pair's separation is constituted by hierarchical depth rather than by displacement alone (§5.8, §6.4).

The resulting record fully characterises the query in the coordinate system. The cost of one query is one CAMeLBERT-ca forward pass on the 3 carrier sentences plus a 99-dimensional cosine computation; total query latency is approximately 1.2 seconds on consumer hardware.

### 3.3 The five operations

The coordinate system supports five primary operations, exposed via the live MCP server (§7) and via direct API calls.

**philological_lookup(term)** — returns the full coordinate record for a single term: a provisional Abjad computation with per-letter breakdown under the currently declared orthographic convention, top attractors with tier/root/wazn/paired-opposite annotations, repelled basis Names, Poincaré position, hierarchical tier, and dominant morphological pattern. Stored per-Name Abjad totals are not presented as final pending the adjudication described in §6.3.

**root_analysis(root)** — given a triliteral root, returns all basis Names sharing that root with full annotation. Use case: exploring the semantic field of a root across the divine Names.

**semantic_project(candidates, context_arabic)** — given a set of cross-linguistic candidate Arabic forms for a concept and a list of Arabic context terms, projects each candidate through the basis and returns its tier, axis, attractor profile, neighbours in the accumulated query dataset, and geometric fit score against the context centroid. Use case: selecting the geometrically correct Arabic form for a specific context.

**semantic_neighbors(term, k, min_r, max_r)** — given a term, returns the k accumulated queries geometrically closest to it by Poincaré distance, with r-value and attractor profile filters. Use case: exploring what other terms occupy similar coordinate regions.

**compare_terms(term1, term2)** — given two terms, returns shared attractors, divergent attractors unique to each, opposing poles (one attracts what the other repels), coordinate distance, hierarchy levels, and provisional Abjad computations under the declared convention. Use case: pairwise structural comparison.

### 3.4 Why the methodology generalises

The methodology has three components: a basis structure, a fitting procedure (CAMeLBERT-ca embeddings → Poincaré disk), and a projection procedure (arbitrary query → coordinate). Of these, only the fitting and projection are tied to CAMeLBERT-ca. The basis structure is exchangeable.

This means the same methodology applied to a different basis (the 7 mu’allaqāt for poetic structure; the 28 letters for phonetic-numerical structure; the rasāʼil of the Ikhwān al-Ṣafāʼ for cosmological structure) would produce a different coordinate system on the same model. Cross-basis comparisons become possible: a term’s coordinate under the 99-Names basis can be compared with its coordinate under the 28-letters basis. The methodology is therefore extensible to any documented relational structure in any language, given a contextual embedding model trained on the appropriate register.

## 4. The resource

### 4.1 Dataset structure

The released basis source contains 99 entries, one per Name, with all 30 fields present in every entry. The release preserves the deployed source rather than constructing a smaller manuscript-only export. Because the fields do not share one evidential status, the metadata partitions them into four categories (Table 1).

| Category | Field count | Status and use |
| --- | --- | --- |
| **Engine-facing descriptive** | 8 | Identity, morphology, hierarchy, axis, and semantic descriptions consumed or displayed by the engine. |
| **Provisional Abjad** | 3 | Stored totals and numerical explanations pending orthographic adjudication. |
| **Documentary and morphological** | 7 | Root, pattern, verse, practice, usage, and entry-level source notes. |
| **Framework-interpretive** | 12 | Phonetic, geometric, cosmological, and machine-learning analogies released for transparency, not represented as primary-source facts. |

**Table 1.** Evidential and functional partition of the released 30-field basis source.

The exact field inventory is given in Table 2; machine-readable types and descriptions are supplied in `data/paper_b/dataset_metadata.json`.

| Category | Fields |
| --- | --- |
| Engine-facing descriptive | `name_ar`, `name_trans`, `name_meaning`, `level`, `root`, `wazn`, `paired_opposite`, `layer2_semantic` |
| Provisional Abjad | `abjad_num`, `abjad_breakdown`, `abjad_resonances` |
| Documentary and morphological | `root_meaning`, `root_derivatives`, `wazn_significance`, `quranic_verse`, `spiritual_practice`, `ibn_arabi_usage`, `textual_sources` |
| Framework-interpretive | `essence_act_polarity`, `mother_name`, `attention_head_analogy`, `circuit_affinity`, `phenomenal_manifestation`, `layer1_phonetic`, `layer3_numerical`, `layer4_geometric`, `layer5_breath`, `layer6_cosmological`, `ml_homolog`, `displacement_signature` |

**Table 2.** Exact 30-field inventory of `basis_99_v3.json`.

The source-level `level` annotation uses values 0–3. It is not identical to the operational three-tier hierarchy displayed by the deployed engine: the latter is read from the fitted `poincare_data_v3.json`, whose 99 nodes use levels 0–2. Both files are released so that this distinction is inspectable.

### 4.2 Construction and quality control

The release is an exact copy of the 99-entry source loaded by the deployed Space at the audited commit, not a reconstruction from the manuscript. Every entry has the same 30 keys and consistent value types; `pipeline/validate_paper_b_dataset.py` checks the schema, checksums, basis/coordinate name alignment, and 99-node count.

The annotations were compiled by the author; no independent inter-annotator agreement study has been conducted. `textual_sources` provides an entry-level free-text source note rather than a field-by-field adjudication trail. The `paired_opposite` field likewise mixes documented equilibrium axes with interpretive or theoretical opposites and should not be treated as a normalized closed pair registry. The framework-interpretive fields are released to make the deployed source transparent, but they are not presented as primary-source facts and are not required to fit query coordinates.

An Abjad total is computationally deterministic only after the orthographic convention is fixed. A pre-release audit flagged 15 stored per-Name values: 4 match only an article-inclusive computation and 11 match neither the current article-free nor article-inclusive computation. The three Abjad fields and any dependent numerical interpretation therefore remain provisional pending adjudication of the five conventions listed in §6.3. The coordinate, attractor, tier, and distance results reported in §5 do not depend on finalizing those stored totals.

### 4.3 Release and access

The basis source, fitted coordinates, metadata, and validator are released in the project repository under CC BY 4.0: - **99-entry basis source:** `data/paper_b/basis_99_v3.json` - **Fitted coordinates:** `data/paper_b/poincare_data_v3.json` - **Metadata:** `data/paper_b/dataset_metadata.json` - **Engine code:** https://github.com/ahmedmest81-ctrl/ALmiraah-project - **Accumulated query dataset (separate; 759 records at audit):** https://huggingface.co/datasets/WELLyes1/almiraah_coordinate_db - **Live MCP server:** https://wellyes1-almiraah-transformer.hf.space/mcp

The MCP server exposes the five operations of §3.3 via the Model Context Protocol, allowing direct integration into LLM tool-calling workflows. The server is rate-limited but freely accessible for non-commercial use.

## 5. Empirical evaluation

We evaluate the coordinate system on six demonstration cases: four pairs that a raw-cosine baseline cannot rank, and two single-term profiles whose structure the baseline cannot express at all. All values in this section are computed under the released v3 engine (carrier-sentence layer-8 embeddings, Karcher placement); query records are in the separately released accumulated query dataset, while the fixed basis and fitted disk are in `data/paper_b/`.

### 5.1 The anisotropy problem: why the cosine baseline cannot discriminate

Raw cosine similarity between CAMeLBERT-ca embeddings occupies a compressed band even at layer 8 under the carrier protocol. Across our four demonstration pairs — which span near-synonyms, speech-act variants, and categorically distinct emotions — raw cosine ranges only from 0.865 to 0.937 (Δ = 0.072). The baseline assigns its *highest* similarity (0.937) to a near-synonym pair and barely lower similarity (0.865) to a pair classical philology treats as categorically distinct. A practitioner using cosine alone cannot recover the distinction structure. This is the documented anisotropy of contextual encoders (Ethayarajh, 2019; Gao et al., 2019), and it motivates measuring terms against a fixed external basis rather than against each other.

### 5.2 Joy near-synonyms: *faraḥ* vs *surūr*

فرح *faraḥ* (assertive elated joy) and سرور *surūr* (tranquil contentment) are both English-glossable as "joy"; classical philology distinguishes them, distributional similarity does not.

**Baseline.** Raw cosine 0.937 — the highest of all four pairs: near-identity.

**Coordinate system.** Both terms project to the Afʿāl tier, and three of five primary attractors are shared (Al-Nūr, Al-Barr, Al-Mājid) — the system registers their genuine kinship. The differentiation appears in two places the baseline cannot see. First, the divergent attractors: *faraḥ* uniquely pulls Al-Fattāḥ (the Opener), *surūr* uniquely pulls Al-Malik (the Sovereign) — an expansive-eruptive versus a settled-sovereign coloration consistent with the classical gloss. Second, the pair carries a hierarchy load of 4.01 — flat displacement of only 0.053, hyperbolic distance 0.211. The two joys sit nearly coincident in flat coordinates but deep in the rim, where small displacements are metrically expensive: their difference is a fine-grained differentiation constituted at depth, exactly where near-synonym distinctions should live.

| Property | *faraḥ* (فرح) | *surūr* (سرور) |
| --- | --- | --- |
| Position r | 0.683 | 0.732 |
| Tier | Afʿāl | Afʿāl |
| Abjad value | Withheld pending adjudication | Withheld pending adjudication |
| Top attractor | Al-Nūr (+0.22) | Al-Nūr (+0.34) |
| Unique attractors | Al-Fattāḥ, Al-Barr-side | Al-Malik, Al-Mājid-side |
| Top-5 overlap | 3 of 5 | 3 of 5 |

**Table 3.** Coordinate profiles of *faraḥ* and *surūr* (v3 engine). Baseline cosine 0.937 indicates near-identity; the coordinate system preserves the kinship (3/5 shared attractors) while localising the distinction in the divergent attractors and a hierarchy load of 4.01.

### 5.3 Love speech-acts: *uḥibbuka* vs *aʿshaquki*

أحبك *uḥibbuka* ("I love you", ḥubb) and أعشقك *aʿshaquki* ("I am consumed by passion for you", ʿishq). Classical philology treats ʿishq as excessive, consuming attachment — morally and psychologically distinct from ḥubb.

**Baseline.** Raw cosine 0.921: near-identity.

**Coordinate system.** The pair shares 4 of 5 attractors (Allāh, Al-Raḥmān, Al-Ḥasīb, Al-Nāfiʿ) — they are the same speech act. The single divergence carries the entire classical distinction: *uḥibbuka*'s remaining attractor is Allāh-centred mercy, while *aʿshaquki*'s top attractor is **Al-Muntaqim (the Avenger, +0.29)**. The field measures ʿishq as love shadowed by vehemence — precisely the consuming-passion semantics the classical lexicographers assign to the root ع-ش-ق. The pair also shows the largest tier-radius separation of the four (r 0.409 vs 0.554) with hierarchy load 2.62, the lowest of the set: the distinction is broad and structural, not fine-grained.

### 5.4 Silence and grief: *ṣamt* vs *ḥuzn*

صمت *ṣamt* (silence) and حزن *ḥuzn* (grief): surface-dissimilar terms a baseline nonetheless scores at 0.885 — within 0.05 of the near-synonym pairs, illustrating the band compression of §5.1.

**Coordinate system.** Top-5 overlap is 1 of 5 (Al-ʿAfuww only). *Ṣamt* pulls Al-Qayyūm and Al-Muṣawwir (self-subsistence, form-giving: silence as sustained interior composure); *ḥuzn* pulls Al-Ḥakam, Al-Wāsiʿ, Al-Shahīd (judgment, encompassing, witness: grief as adjudicated experience). The shared Al-ʿAfuww is itself informative — both states border effacement. Where the baseline compresses, the coordinate system separates the pair almost completely while naming the one axis they genuinely share.

### 5.5 Anger and fear: *ghaḍab* vs *khawf*

غضب *ghaḍab* (anger) and خوف *khawf* (fear): categorically distinct emotions, baseline cosine 0.865 — the band again.

**Coordinate system.** Top-5 overlap 1 of 5 (Al-ʿAfuww); the largest hyperbolic separation of the four pairs (0.711, load 3.54). *Ghaḍab* pulls Al-Ḥakam and Al-Wahhāb; *khawf* pulls Al-Wāsiʿ and Al-Muʾmin — fear oriented toward the Granter of safety (the root أ-م-ن shared by *muʾmin* and *amān*), a root-level recovery the baseline has no means to express.

### 5.6 Intimacy as a centrally-placed term: *uns*

أنس *uns* (intimacy, sociable warmth) projects with attractors Allāh (+0.24), Al-Aḥad (+0.24), Al-Nūr (+0.21), Al-Barr (+0.18), Al-Jabbār (+0.15), and repelled Names Al-Mumīt (−0.21), Al-Muʿīd (−0.19), Al-Qābiḍ (−0.19). The profile reads coherently: intimacy gathered around unity, light, and benevolence, repelling death, regression, and constriction. The repelled set is as diagnostic as the attractors — a property of the system noted throughout: repelled Names are frequently the more semantically faithful signal.

### 5.7 Oppression at the ontological tier: *ẓulm*

ظلم *ẓulm* (oppression, wrongdoing) is the clearest single-term demonstration in the set. Its full top-5: **Al-Ḥaqq (+0.27), Al-Ḥakam (+0.24), Allāh (+0.22), Al-Muntaqim (+0.21), Al-ʿAdl (+0.21)** — and it repels Al-Raʾūf (−0.20). The field measures oppression entirely in the vocabulary of truth, judgment, retribution, and justice, while repelling gentleness. No term in the demonstration set produces a more doctrinally exact profile: *ẓulm* in classical usage is precisely the violation measured against ḥaqq and ʿadl. The term projects near the disk centre (r = 0.319): the system places wrongdoing close to the ontological tier where its adjudicating Names reside.

### 5.8 Summary of the six cases

The baseline's compressed band (0.865–0.937) cannot rank the four pairs. The coordinate system ranks them correctly and for the right reasons: the love pair (same speech act, one divergent axis) overlaps 4/5; the joy near-synonyms (genuine kin, fine differentiation) overlap 3/5 with the highest hierarchy load; the two cross-category pairs overlap 1/5 each with the largest geodesic separations. Differentiation appears where philology expects it — in divergent attractors for kin pairs, in wholesale profile separation for categorical pairs, and in hierarchy load for distinctions constituted at depth. The two single-term profiles (*uns*, *ẓulm*) demonstrate that the system's output is not merely a similarity score but a structured, interpretable profile in which the repelled set carries independent signal.

## 6. Discussion

### 6.1 What kind of NLP problem the methodology addresses

The methodology addresses problems where distributional similarity is insufficient: classical Arabic near-synonym disambiguation, fine-grained register classification, structural-relational queries that depend on properties (root family, morphological pattern, hierarchical tier) not encoded in distributional embeddings. Concrete downstream applications include translation tasks requiring register-sensitive choice between near-synonyms, classical-Arabic lexicographic work, and any application requiring structural-relational rather than purely-distributional discrimination.

The methodology does *not* address problems for which distributional similarity is sufficient. For coarse-grained classification, retrieval, or task-tuned classification on modern Arabic, distributional embeddings work well and the coordinate-basis methodology adds overhead without benefit.

### 6.2 Relation to Paper A

The coordinate engine relies on CAMeLBERT-ca contextual embeddings produced by the same forward-pass mechanism Paper A measures. Paper A establishes empirically that classical Arabic root and wazn structure are encoded as geometrically separable, linearly decodable properties of CAMeLBERT-ca embeddings: across 951 word forms spanning 200 reviewed triliteral root families, roots cluster and the four broad-coverage derived patterns are linearly decodable by leave-one-out analogy completion (mean cosine 0.924–0.983; n = 98–200), though the per-root pattern offsets are not a single consistent translation vector. The 99-Names basis trades on the separability and clustering, not on a constant pattern direction. The `root` and `wazn` fields in the basis source align with the geometry Paper A measures: basis Names sharing a root cluster in coordinate space. Paper A’s empirical findings provide the structural justification for the coordinate basis to function as it does on CAMeLBERT-ca specifically.

### 6.3 Limitations

- **Single model.** The methodology has been developed and evaluated on CAMeLBERT-ca only. Cross-model robustness (whether the same fitted basis produces similar coordinates on AraBERT, MARBERT, or other Arabic encoders) is not yet tested.

- **Classical-register specificity.** The basis is fitted on classical-Arabic-trained encoder embeddings. Application to modern-Arabic-trained encoders on modern corpora has not been evaluated; we expect the methodology will require register-matched re-fitting for modern Arabic.

- **Basis-specific results.** The coordinate distances and tier classifications reported in §5 are properties of the 99-Names basis specifically. A different basis would produce a different coordinate system on the same model.

- **Carrier-sentence sensitivity.** Query embeddings are sensitive to the choice of carrier sentences; we use a fixed set of three carriers across all queries to ensure consistency.

- **Wazn parser coverage.** The independent query-wazn parser (§3.2) returns candidate sets for unvocalised input rather than unique patterns, and approximately 17% of accumulated terms fall outside the current template inventory and are marked `not_parsed` (§6.5). The attractor cluster’s prevailing wazn is reported separately as `cluster_wazn` and is never substituted for the query’s own pattern. None of the tier, attractor, or coordinate-distance results in §5 depend on the query wazn.

- **Per-Name Abjad values await adjudication.** An audit of the 99 stored totals flagged 15 entries. Five project-wide conventions must be fixed before those values are regenerated and published as final: (i) article-free versus article-inclusive canonical forms; (ii) madda آ valued as 1 or as ء + ا = 2; (iii) hamza-seat ؤ/ئ valuation; (iv) tāʾ marbūṭa ة valued as 400 or under another declared rule; and (v) full versus truncated forms for multi-word Names such as ذو الجلال والإكرام. Until that adjudication is complete, this paper withholds stored per-Name totals and treats any live per-letter computation as convention-labelled and provisional. The coordinate, attractor, tier, and distance results reported in §5 do not depend on these totals.

- **No theological commitment in the resource is intended.** The methodology functions as a relational-structure instrument. A reader without classical-Arabic-Islamic training can use the dataset and code reproducibly. The basis structure is functioning instrumentally, not doctrinally.

### 6.4 Engine version and protocol correction

The values in §5 are produced by the v3 engine, which corrected a protocol divergence discovered during pre-release verification: an earlier deployed server embedded bare terms with last-layer mean pooling rather than the carrier-sentence layer-8 protocol specified in §3.1. The correction had three measurable consequences, reported here as a robustness result.

**Hub-attractor dissolution.** Under the divergent protocol, a small set of basis Names dominated attractor lists for semantically unrelated queries: the two worst appeared in the top-5 of 47% of all accumulated query records. Under the v3 protocol the maximum top-5 occupancy across the basis falls to 39% — and the Name holding it is Allāh, whose broad pull is doctrinally expected (the *ism jāmiʿ*). The hub structure of the earlier protocol was an embedding artifact, not field structure. As a sanity criterion, the query نور (light) attracts the basis Name Al-Nūr at +0.71 under v3 — a signal invisible under the earlier protocol.

**Profile stability across protocols.** Attractor profiles that survived the protocol change (e.g. *ṣidq* → Al-Ḥaqq; the effacement cluster around Al-ʿAfuww) constitute structure robust to the measurement instrument; profiles that did not survive were concentrated among the hub Names. This is direct, if informal, evidence for the generalisation claim of §3.4: the methodology's outputs distinguish instrument artifact from field structure when the instrument is varied.

**Intrinsic placement.** Query positions are computed by Karcher mean rather than Euclidean centroid (§3.2). The Euclidean shortcut systematically under-estimated radial depth (mean Δr = 0.031 on a 20-term validation set) and changed tier assignment for 4 of 20 terms.

### 6.5 Known limitations of the released geometry

Three limitations are documented for users of the resource. First, centre-region compression: terms whose attractor sets are broad (dominated by Dhāt-tier Names) receive near-coincident Karcher positions; nearest-neighbour orderings at small r are therefore less reliable than at large r. Second, tier assignment by attractor vote and radial position are correlated but not identical signals, and may disagree for individual terms; both are reported. Third, the wazn parser returns candidate sets for unvocalised input rather than unique patterns (§3.2); approximately 17% of accumulated terms fall outside the current template inventory and are marked `not_parsed`, a coverage figure that bounds the parser's current scope.

### 6.6 Future work

The methodology generalises to any documented relational structure. Natural extensions include: a 28-letter alphabetic-phonetic basis for classical Arabic phonological analysis; a Hebrew analog using the sefirotic tradition documented in Sefer Yetzirah; cross-basis comparison (a term’s coordinate under the 99-Names basis compared with its coordinate under a complementary basis on the same model). Each extension is a separate empirical study with its own evaluation methodology.


## 7. Reproducibility and access

- **99-entry basis source and fitted disk:** `data/paper_b/` in https://github.com/ahmedmest81-ctrl/ALmiraah-project, released under CC BY 4.0

- **Accumulated query dataset:** WELLyes1/almiraah_coordinate_db on HuggingFace (759 records at the audited commit), released separately from the fixed basis

- **Engine code:** https://github.com/ahmedmest81-ctrl/ALmiraah-project

- **Live MCP server:** https://wellyes1-almiraah-transformer.hf.space/mcp

- **Model:** CAMeL-Lab/bert-base-arabic-camelbert-ca

- **Carrier sentences:** Three fixed templates (specified in §3.1); released alongside the engine code

- **Schema and annotation status:** `data/paper_b/dataset_metadata.json`; entry-level source notes are in `textual_sources`, while interpretive and provisional fields are explicitly labeled

- **Validation:** `python pipeline/validate_paper_b_dataset.py`

- **Compute requirements:** Engine fitting requires one CAMeLBERT-ca forward pass per carrier sentence per basis entry (99 × 3 = 297 forward passes total) and runs on a single CPU on consumer hardware. Query latency ~1.2 seconds per query on consumer hardware.

- **Random seed:** 42 throughout. Poincaré-disk fitting is deterministic given seed.

## 8. Conclusion

We present a methodology for using a documented scholarly relational structure as a fitted coordinate basis on contextual embeddings, applied specifically to the 99 Names of God in classical Islamic tradition fitted to CAMeLBERT-ca. The methodology produces a coordinate system in which arbitrary Arabic terms receive a Poincaré-disk position, hierarchical tier classification, attractor and repellor profiles against the basis, and morphological annotations. We demonstrate on six empirical cases — drawn from emotion, cognition, and morality vocabulary — that the coordinate system makes structural distinctions a CAMeLBERT-ca baseline cosine similarity cannot make. We release the exact 99-entry, 30-field basis source used by the deployed engine, its fitted 99-node Poincaré coordinates, machine-readable metadata and validation, the coordinate engine code, and a live MCP server exposing five primary operations.

The contribution is methodological as much as it is a resource. The framework — basis structure plus fitting procedure plus projection procedure — applies to any documented relational structure in any language. The 99 Names of God is one instance. The empirical demonstrations show what one such instance does for classical Arabic. The resource is offered to the NLP community as a reproducible instrument for fine-grained semantic analysis where distributional similarity is insufficient.

## Acknowledgements

This work was conducted as an independent research project without institutional affiliation or funding. The methodology design, dataset annotation, engine implementation, and writing were carried out collaboratively with the AL-MIRʾĀH analytical framework, an internal philological-computational toolset developed by the author. The author thanks the maintainers of OpenITI (M. Romanov and collaborators) for the classical Arabic corpus, the CAMeL Lab at NYU Abu Dhabi for the CAMeLBERT model family, and HuggingFace for hosting the live MCP server.

## References

Al-Ghazālī, Abū Ḥāmid Muḥammad. (1095/1992). *Al-Maqṣad al-asnā fī sharḥ asmāʼ Allāh al-ḥusnā* [The Most Beautiful Names of God]. (D. Burrell & N. Daher, Trans.). Islamic Texts Society.

Alakeel, Y., Qwaider, C., Aldarmaki, H., & Alqahtani, S. (2026). Morphemes Without Borders: Evaluating Root-Pattern Morphology in Arabic Tokenizers and LLMs. In *Proceedings of LREC-COLING 2026*. arXiv:2603.15773.

Antoun, W., Baly, F., & Hajj, H. (2020). AraBERT: Transformer-based model for Arabic language understanding. In *Proceedings of OSACT4, LREC 2020*.

Colin, G. S. (1986). Abjad. In *Encyclopaedia of Islam*, 2nd ed., vol. I. Brill.

Ethayarajh, K. (2019). How contextual are contextualized word representations? Comparing the geometry of BERT, ELMo, and GPT-2 embeddings. In *Proceedings of EMNLP-IJCNLP 2019*, 55–65.

Gao, J., He, D., Tan, X., Qin, T., Wang, L., & Liu, T.-Y. (2019). Representation degeneration problem in training natural language generation models. In *Proceedings of ICLR 2019*.

Ibn ʿArabī, Muḥyī al-Dīn. (1240/2004). *Fuṣūṣ al-ḥikam* [The Ringstones of Wisdom]. (C. Dagli, Trans.). Kazi.

Inoue, G., Alhafni, B., Baimukan, N., Bouamor, H., & Habash, N. (2021). The interplay of variants, size, and task type in Arabic pre-trained language models. In *Proceedings of WANLP*.

Mislati, A. (2026). Templatic morphology as decodable geometry, and Abjad letter-values as an attention probe, in classical Arabic transformer models. *arXiv preprint cs.CL*. Code: https://github.com/ahmedmest81-ctrl/ALmiraah-project

Romanov, M. (2019). OpenITI: A machine-readable corpus of Islamicate texts. In *Proceedings of DH2019*.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In *NeurIPS*.

*Word count: ~6,400 (excluding tables, references). Estimated typeset length: 10–11 pages including tables.*

*Paper B v5. End.*
