# The 99-Names Coordinate System: A Fitted Relational Basis for Classical Arabic in CAMeLBERT-ca

*A scholarly relational structure as a coordinate basis for fine-grained semantic analysis of classical Arabic terms*

Ahmed Mislati Independent researcher, Vienna arXiv preprint cs.CL · May 2026

**Keywords:** Arabic NLP, classical Arabic, fitted basis, contextual embeddings, CAMeLBERT, semantic resource, near-synonym disambiguation, Poincaré embedding, LREC, language resource

## Abstract

Distributional similarity tools (cosine similarity, k-nearest neighbours, contextual embedding distance) cannot reliably resolve fine-grained relational distinctions in Arabic vocabulary. Two terms English-glossable as “joy” (فرح *faraḥ* and سرور *surūr*) cluster near each other in any contextual embedding space, but classical Arabic philology treats them as structurally distinct: *faraḥ* names assertive elated emotion, *surūr* names tranquil contentment. Two speech-acts both glossable as “I love you” (أحبك *uḥibbuka* and أعشقك *aʿshaquki*) function in markedly different registers, but a distributional model treats them as paraphrases. The problem is not a defect of contextual embeddings; it is that distributional similarity does not encode the relational properties — root families, morphological patterns, hierarchical tier, paired-opposite structure — that classical scholarly practice formalised over a millennium.

We present a methodology and resource for incorporating a documented scholarly relational structure as a *fitted coordinate basis* on the contextual embedding space of CAMeLBERT-ca. The basis structure is the 99 Names of God in classical Islamic tradition: a closed, well-attested relational system with documented hierarchical tiers, paired opposites, root families, and morphological patterns. We treat the 99 Names as a coordinate-defining relational structure, not as a theological commitment. Arbitrary Arabic terms are projected into the basis via CAMeLBERT-ca contextual embeddings followed by Poincaré-disk geometric reduction. The resulting coordinate system supports five operations: philological lookup, root-family analysis, semantic projection, semantic neighbour retrieval, and structural comparison.

We release: (i) a 99-entry annotated dataset covering 19 fields per entry partitioned into operational (used by the coordinate engine) and documentary (philological metadata) layers; (ii) the coordinate engine code; (iii) a live MCP server exposing the five operations. We demonstrate the methodology on six empirical cases drawn from emotion, cognition, and morality vocabulary. A measured baseline establishes the core difficulty: under the anisotropy of CAMeLBERT-ca contextual embeddings, all four demonstration pairs — joy near-synonyms, love speech-acts, silence/grief, anger/fear — receive within-pair cosine similarities compressed into a narrow 0.86–0.94 band, so that cosine magnitude carries no information about the *kind* of relationship each pair has. Against this baseline, the coordinate system separates the pairs by two orders of magnitude in coordinate distance (0.007 to 0.523): near-synonyms and register-variants that receive near-identical cosines (0.94 vs 0.92) are an order of magnitude apart in the coordinate basis (0.067 vs 0.523), while surface-distinct terms that cosine scores no differently from any other pair (silence/grief at 0.89; anger/fear at 0.86) collapse to near-zero coordinate distance with majority-shared attractor sets. The methodology is generalisable: any documented scholarly relational structure in any language could in principle function as a fitted basis the same way.

## 1. Introduction

### 1.1 The problem

Contextual embeddings produced by transformer encoders such as CAMeLBERT-ca (Inoue et al., 2021) capture distributional regularities in classical Arabic vocabulary at the level of token co-occurrence and contextual prediction. Two terms used in similar contexts will receive geometrically similar contextual embeddings, and cosine similarity between their representations will be high. This property supports a wide range of downstream tasks including retrieval, classification, and translation. It does not, however, capture the *relational* structure that classical Arabic philology has formalised: which root family a term belongs to, what morphological pattern it instantiates, what numerical (Abjad) value it carries, and — for terms in the rich emotional, cognitive, moral, and theological vocabulary of classical Arabic — which structural relationships it has with other terms in the same domain.

This omission produces concrete failures. Consider the Arabic emotional vocabulary for “joy”: فرح *faraḥ*, سرور *surūr*, بهجة *bahja*, طرب *ṭarab*, إنشراح *inshirāḥ*. A dictionary glosses all five as “joy” or “happiness.” A contextual embedding model places all five in a tight cluster. But classical Arabic philology distinguishes them precisely: *faraḥ* is assertive and outward; *surūr* is tranquil and contained; *bahja* is celebratory; *ṭarab* is musical/ecstatic; *inshirāḥ* is the breast-opening of relief. These distinctions matter for translation, classical Arabic NLP applications, and any task requiring fine-grained semantic discrimination.

The problem generalises beyond near-synonym disambiguation. Speech-acts — أحبك *uḥibbuka* (“I love you” in a stable affectional register) versus أعشقك *aʿshaquki* (“I love you” in a passionate-obsessive register) — function differently in social practice but cluster identically in distributional space. Antonyms such as حب *ḥubb* (love) and كره *kurh* (hatred) sit closer to each other than either does to indifference, a fact classical philology has documented but distributional tools fail to register. Terms with deep philological history — ظلم *ẓulm*, etymologically “putting in the wrong place,” classically used for both oppression and ontological error — carry structural information that a distributional model cannot recover.

### 1.2 The contribution

We present a methodology and resource for incorporating a documented scholarly relational structure as a *fitted coordinate basis* on CAMeLBERT-ca contextual embeddings. The methodology is structured around three elements:

- A **basis structure** — a closed, well-attested, internally relational set of terms with documented structural properties (hierarchical tier, paired opposites, root families, morphological patterns, numerical values). The basis is fixed; it does not change with query.

- A **fitting procedure** — for each basis term, a CAMeLBERT-ca contextual embedding is computed, and the basis is reduced to a Poincaré disk via geometric projection that preserves hierarchical relationships.

- A **projection procedure** — for an arbitrary query term, CAMeLBERT-ca contextual embedding is computed, similarities against the basis are calculated, and the term is located in the disk according to its attractor and repellor profile against the basis.

The result is a coordinate system: every Arabic term receives a position (px, py, r), a hierarchical tier classification (Dhāt / Ṣifāt / Afʿāl), an attractor set (which basis terms it is structurally similar to), and a repellor set (which basis terms it is structurally opposed to).

Our specific basis is the **99 Names of God** in classical Islamic tradition. We treat this set as a coordinate-defining relational structure, not as a theological commitment. The 99 Names satisfy the basis requirements: closed (n = 99), well-attested (documented continuously since the 8th century), internally relational (with paired opposites such as Al-Bāṣiṭ ⇄ Al-Qābiḍ documented as cosmic-equilibrium axes), and structurally annotated (with root families, Abjad values, morphological patterns, and tier classification all attested in classical philological sources). The basis is functioning as a methodological instrument; the methodology generalises to any analogous structure in any language.

### 1.3 What we contribute

This paper makes four contributions:

- A methodology for using a documented scholarly relational structure as a fitted coordinate basis on contextual embeddings, applicable to any language and any analogously-structured basis.

- A 99-entry annotated dataset covering 19 fields per entry (8 operational + 11 documentary), released on HuggingFace under permissive licence, with explicit field-level documentation of which fields are used by the coordinate engine and which are philological metadata.

- A live deployment — a coordinate engine accessible via an MCP server — exposing five operations: philological lookup, root-family analysis, semantic projection of cross-linguistic candidates, semantic neighbour retrieval, and pairwise structural comparison.

- Six empirical case studies demonstrating that the methodology produces structural distinctions a CAMeLBERT-ca baseline cosine similarity cannot make, drawn from emotion vocabulary (joy near-synonyms; love speech-acts), structural-relational vocabulary (silence/grief, anger/fear), and morally-loaded vocabulary (oppression, intimacy).

### 1.4 What we do not claim

We do not claim the 99 Names are *the* correct basis for classical Arabic. Other documented relational structures (the 28-letter Arabic alphabetical hierarchy, the seven mu’allaqāt poetic structure, the 99-pole Sufi nafs/qalb/rūḥ taxonomy) would function as alternative bases the methodology supports equally well. We do not claim the coordinate system captures the *full* relational structure of classical Arabic; we claim it captures the relational structure of the 99-Name basis specifically, and demonstrates substantive distinctions distributional similarity cannot make. We do not claim theological commitments; the methodology is reproducible by a researcher with no Islamic-studies background, using only the released dataset and code.

## 2. The 99-Names as a relational structure

### 2.1 Background

The 99 Names of God (*al-asmāʼ al-ḥusnā*, “the most beautiful Names”) are a canonical set of divine attributes documented in classical Islamic tradition since approximately the 8th century. Each Name names a divine function or quality (Al-Raḥmān, “the All-Merciful”; Al-Jabbār, “the Compeller”; Al-Ḥakam, “the Judge”). The set is closed; the standard enumeration of 99 has been stable across classical scholarly tradition for over a millennium (al-Ghazālī, 1095/1992; Ibn ʿArabī, 1240/2004).

For the methodological purposes of this paper, three properties of the 99-Name structure are operationally important.

**Hierarchical tier.** Classical Akbarī cosmology classifies each Name into one of three tiers: **Dhāt** (essence-level, ontologically primary Names: Al-Lāh, Al-Aḥad, Al-Ḥayy, Al-Ḥaqq); **Ṣifāt** (attribute-level, persistent qualities: Al-Raḥmān, Al-Ḥakīm, Al-Karīm); and **Afʿāl** (action-level, operative-functional Names: Al-Jabbār, Al-Qābiḍ, Al-Bāsiṭ). The tier classification is documented in classical sources and is operationally stable.

**Paired opposites.** Classical tradition documents specific axes of cosmic equilibrium: Al-Qābiḍ ⇄ Al-Bāsiṭ (constriction/expansion), Al-Muʿizz ⇄ Al-Mudhill (honor/humiliation), Al-Rāfiʿ ⇄ Al-Khāfiḍ (elevation/lowering), Al-Raḥmān ⇄ Al-Muntaqim (mercy/avengement). Each paired-opposite axis is annotated in our dataset with its classical attestation.

**Root families.** The 99 Names exhaust most of the high-frequency triliteral roots of classical Arabic theological vocabulary. The root ر-ح-م (mercy) appears in three Names: Al-Raḥmān, Al-Raḥīm, Al-Raʾūf. The root ك-ر-م (generosity) appears in Al-Karīm, Al-Akram. The root families create natural neighbourhoods in coordinate space.

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

**philological_lookup(term)** — returns the full coordinate record for a single term: Abjad value with per-letter breakdown, top attractors with tier/root/wazn/paired-opposite annotations, repelled basis Names, Poincaré position, hierarchical tier, and dominant morphological pattern.

**root_analysis(root)** — given a triliteral root, returns all basis Names sharing that root with full annotation. Use case: exploring the semantic field of a root across the divine Names.

**semantic_project(candidates, context_arabic)** — given a set of cross-linguistic candidate Arabic forms for a concept and a list of Arabic context terms, projects each candidate through the basis and returns its tier, axis, attractor profile, neighbours in the accumulated query dataset, and geometric fit score against the context centroid. Use case: selecting the geometrically correct Arabic form for a specific context.

**semantic_neighbors(term, k, min_r, max_r)** — given a term, returns the k accumulated queries geometrically closest to it by Poincaré distance, with r-value and attractor profile filters. Use case: exploring what other terms occupy similar coordinate regions.

**compare_terms(term1, term2)** — given two terms, returns shared attractors, divergent attractors unique to each, opposing poles (one attracts what the other repels), coordinate distance, hierarchy levels, and Abjad values. Use case: pairwise structural comparison.

### 3.4 Why the methodology generalises

The methodology has three components: a basis structure, a fitting procedure (CAMeLBERT-ca embeddings → Poincaré disk), and a projection procedure (arbitrary query → coordinate). Of these, only the fitting and projection are tied to CAMeLBERT-ca. The basis structure is exchangeable.

This means the same methodology applied to a different basis (the 7 mu’allaqāt for poetic structure; the 28 letters for phonetic-numerical structure; the rasāʼil of the Ikhwān al-Ṣafāʼ for cosmological structure) would produce a different coordinate system on the same model. Cross-basis comparisons become possible: a term’s coordinate under the 99-Names basis can be compared with its coordinate under the 28-letters basis. The methodology is therefore extensible to any documented relational structure in any language, given a contextual embedding model trained on the appropriate register.

## 4. The resource

### 4.1 Dataset structure

The released dataset contains 99 entries, one per Name, with 19 fields per entry partitioned into two layers (Table 1).

| Layer | Field count | Use |
| --- | --- | --- |
| **Layer 1 — Operational** | 8 | Used by the coordinate engine. Required for tool operation. |
| **Layer 2 — Documentary** | 11 | Philological metadata. Not used by the coordinate engine; provided for scholarly reference. |

**Table 1.** Field partition. Layer 1 fields are minimal for tool operation; Layer 2 fields are documentary.

The full field inventory is given in Table 2.

| Field | Layer | Type | Description |
| --- | --- | --- | --- |
| name_arabic | 1 | string | Arabic form of the Name (e.g. الرَّحْمَان) |
| name_transliterated | 1 | string | Romanised form (e.g. Al-Raḥmān) |
| meaning_english | 1 | string | Concise English gloss |
| tier | 1 | enum | One of Dhāt / Ṣifāt / Afʿāl |
| root | 1 | string | Triliteral root with hyphen-separated consonants |
| wazn | 1 | string | Morphological pattern (e.g. faʿʿāl, faʿīl) |
| abjad_value | 1 | int | Sum of letter values in Mashriqi order |
| paired_opposite | 1 | string | Name forming the documented equilibrium axis |
| meaning_extended | 2 | string | Extended philological description |
| classical_source | 2 | string | Primary attestation (al-Ghazālī, Ibn ʿArabī, etc.) |
| phonetic_layer | 2 | string | Articulatory analysis |
| numerical_layer | 2 | string | Abjad-relational and gematric annotations |
| geometric_layer | 2 | string | Visual-letter analysis |
| breath_layer | 2 | string | Vocalisation rhythm and pattern |
| invocation_context | 2 | string | Classical contexts of liturgical use |
| frequency_in_corpus | 2 | int | OpenITI classical corpus frequency |
| seasonal_associations | 2 | string | Classical calendrical associations |
| color_associations | 2 | string | Classical chromatic associations |
| poetic_meter | 2 | string | Associated metrical patterns |

**Table 2.** Full field inventory of the 99-Names dataset. Layer 1 fields (rows 1–8) are operational; Layer 2 fields (rows 9–19) are documentary.

We release **Layer 1 + Layer 2 (all 19 fields)** in this resource publication. A third layer of framework-interpretive fields (cosmological mapping, ML homolog annotations, displacement signatures) is held back for a separate forthcoming publication where the interpretive framing is the load-bearing contribution.

### 4.2 Construction and quality control

Layer 1 fields were extracted from primary classical sources: al-Ghazālī’s *al-Maqṣad al-asnā* (1095/1992) for tier classification and meaning glosses, Ibn ʿArabī’s *Fuṣūṣ al-Ḥikam* (1240/2004) for paired-opposite axes, and standard root-extraction for root and wazn fields. The abjad_value field is deterministic from the Arabic form using the Mashriqi mapping (Colin, 1986). Two annotators with classical-Arabic philological training cross-checked all Layer 1 fields; inter-annotator agreement on tier classification was 0.97 (Cohen’s κ) and on paired-opposite identification was 0.91.

Layer 2 fields were compiled from multiple classical and secondary sources; per-field provenance is documented in the dataset’s metadata file. Layer 2 fields are provided as scholarly reference and are not load-bearing for the coordinate engine’s operation.

### 4.3 Release and access

The dataset is released on HuggingFace under CC BY 4.0: - **Dataset:** WELLyes1/almiraah_coordinate_db - **Engine code:** released at https://github.com/ahmedmest81-ctrl/ALmiraah-project - **Live MCP server:** https://wellyes1-almiraah-transformer.hf.space/mcp

The MCP server exposes the five operations of §3.3 via the Model Context Protocol, allowing direct integration into LLM tool-calling workflows. The server is rate-limited but freely accessible for non-commercial use.

## 5. Empirical evaluation

We evaluate the coordinate system on six demonstration cases: four pairs that a raw-cosine baseline cannot rank, and two single-term profiles whose structure the baseline cannot express at all. All values in this section are computed under the released v3 engine (carrier-sentence layer-8 embeddings, Karcher placement); the full per-term records are in the released dataset.

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
| Abjad value | 288 | 466 |
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

The coordinate engine relies on CAMeLBERT-ca contextual embeddings produced by the same forward-pass mechanism Paper A measures. Paper A establishes empirically that classical Arabic root and wazn structure are encoded as geometrically separable, linearly decodable properties of CAMeLBERT-ca embeddings: roots cluster, and morphological pattern membership is significantly separable and linearly decodable (leave-one-out analogy completion >0.92 for every well-attested pattern across 200 root families), though the per-root pattern offsets are not a single consistent translation vector. The 99-Names basis trades on the separability and clustering, not on a constant pattern direction. The root and wazn fields in the dataset (Layer 1) align with the geometry Paper A measures: the basis Names sharing a root cluster in coordinate space. Paper A’s empirical findings provide the structural justification for the coordinate basis to function as it does on CAMeLBERT-ca specifically.

### 6.3 Limitations

- **Single model.** The methodology has been developed and evaluated on CAMeLBERT-ca only. Cross-model robustness (whether the same fitted basis produces similar coordinates on AraBERT, MARBERT, or other Arabic encoders) is not yet tested.

- **Classical-register specificity.** The basis is fitted on classical-Arabic-trained encoder embeddings. Application to modern-Arabic-trained encoders on modern corpora has not been evaluated; we expect the methodology will require register-matched re-fitting for modern Arabic.

- **Basis-specific results.** The coordinate distances and tier classifications reported in §5 are properties of the 99-Names basis specifically. A different basis would produce a different coordinate system on the same model.

- **Carrier-sentence sensitivity.** Query embeddings are sensitive to the choice of carrier sentences; we use a fixed set of three carriers across all queries to ensure consistency.

- **Query-wazn parsing not yet deployed.** The released server reports the attractor cluster’s dominant wazn rather than an independent parse of the query term’s own morphological pattern. An independent query-parsing module (positional-radical substitution) is implemented but not yet deployed; until it is, the demonstration cases in §5 omit a per-query wazn rather than report the cluster-inherited value. This does not affect the tier, attractor, coordinate-distance, or Abjad results, none of which depend on the query wazn.

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

- **Dataset:** WELLyes1/almiraah_coordinate_db on HuggingFace, released under CC BY 4.0

- **Engine code:** https://github.com/ahmedmest81-ctrl/ALmiraah-project

- **Live MCP server:** https://wellyes1-almiraah-transformer.hf.space/mcp

- **Model:** CAMeL-Lab/bert-base-arabic-camelbert-ca

- **Carrier sentences:** Three fixed templates (specified in §3.1); released alongside the engine code

- **Annotation provenance:** Per-field source documentation in dataset_metadata.json

- **Compute requirements:** Engine fitting requires one CAMeLBERT-ca forward pass per basis entry (99 × 3 = 297 forward passes total); single CPU inference at ~3 seconds total. Query latency ~1.2 seconds per query on consumer hardware.

- **Random seed:** 42 throughout. Poincaré-disk fitting is deterministic given seed.

## 8. Conclusion

We present a methodology for using a documented scholarly relational structure as a fitted coordinate basis on contextual embeddings, applied specifically to the 99 Names of God in classical Islamic tradition fitted to CAMeLBERT-ca. The methodology produces a coordinate system in which arbitrary Arabic terms receive a Poincaré-disk position, hierarchical tier classification, attractor and repellor profiles against the basis, and morphological annotations. We demonstrate on six empirical cases — drawn from emotion, cognition, and morality vocabulary — that the coordinate system makes structural distinctions a CAMeLBERT-ca baseline cosine similarity cannot make. We release the annotated 99-entry dataset (19 fields per entry, partitioned into operational and documentary layers), the coordinate engine code, and a live MCP server exposing five primary operations.

The contribution is methodological as much as it is a resource. The framework — basis structure plus fitting procedure plus projection procedure — applies to any documented relational structure in any language. The 99 Names of God is one instance. The empirical demonstrations show what one such instance does for classical Arabic. The resource is offered to the NLP community as a reproducible instrument for fine-grained semantic analysis where distributional similarity is insufficient.

## Acknowledgements

This work was conducted as an independent research project without institutional affiliation or funding. The methodology design, dataset annotation, engine implementation, and writing were carried out collaboratively with the AL-MIRʾĀH analytical framework, an internal philological-computational toolset developed by the author. The author thanks the maintainers of OpenITI (M. Romanov and collaborators) for the classical Arabic corpus, the CAMeL Lab at NYU Abu Dhabi for the CAMeLBERT model family, and HuggingFace for hosting the live MCP server.

## References

Al-Ghazālī, Abū Ḥāmid Muḥammad. (1095/1992). *Al-Maqṣad al-asnā fī sharḥ asmāʼ Allāh al-ḥusnā* [The Most Beautiful Names of God]. (D. Burrell & N. Daher, Trans.). Islamic Texts Society.

Antoun, W., Baly, F., & Hajj, H. (2020). AraBERT: Transformer-based model for Arabic language understanding. In *Proceedings of OSACT4, LREC 2020*.

Colin, G. S. (1986). Abjad. In *Encyclopaedia of Islam*, 2nd ed., vol. I. Brill.

Ethayarajh, K. (2019). How contextual are contextualized word representations? Comparing the geometry of BERT, ELMo, and GPT-2 embeddings. In *Proceedings of EMNLP-IJCNLP 2019*, 55–65.

Gao, J., He, D., Tan, X., Qin, T., Wang, L., & Liu, T.-Y. (2019). Representation degeneration problem in training natural language generation models. In *Proceedings of ICLR 2019*.

Ibn ʿArabī, Muḥyī al-Dīn. (1240/2004). *Fuṣūṣ al-ḥikam* [The Ringstones of Wisdom]. (C. Dagli, Trans.). Kazi.

Inoue, G., Alhafni, B., Baimukan, N., Bouamor, H., & Habash, N. (2021). The interplay of variants, size, and task type in Arabic pre-trained language models. In *Proceedings of WANLP*.

Mislati, A. (2026). Templatic morphology as decodable geometry, and Abjad letter-values as an attention probe, in classical Arabic transformer models. *arXiv preprint cs.CL*.

Romanov, M. (2019). OpenITI: A machine-readable corpus of Islamicate texts. In *Proceedings of DH2019*.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In *NeurIPS*.

*Word count: ~6,400 (excluding tables, references). Estimated typeset length: 10–11 pages including tables.*

*Paper B draft v1. End.*