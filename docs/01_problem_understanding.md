# 1. Problem understanding

## 1.1 What is stance detection?

**Stance detection** decides whether the author of a text is *in favour of*, *against*, or *neutral toward* a given **target**: an entity, a topic or, here, a **claim** (Mohammad et al., 2016; Küçük & Can, 2020). The output is a property of the pair *(text, target)*, not of the text alone:

> "Paani khatam ho raha hai, ped kat rahe hain" ("water is running out, trees are being cut")
> → **Favour** for the target "climate change is a serious concern"
> → **None** for the target "India should host the Olympics"

## 1.2 The claim

> **C = "Climate change and global warming is a serious concern."**

This is the same proposition that Luo, Card & Jurafsky (2020) use for the Global Warming Stance Dataset (GWSD), and close to the SemEval-2016 target *"Climate Change is a Real Concern"*. Both datasets are therefore directly usable as supervision (see [03_datasets.md](03_datasets.md)).

| Label | Meaning with respect to C | Typical surface forms |
|---|---|---|
| **Favour** | The author believes climate change is real **and** serious, worries about it, or supports action | "ग्लोबल वार्मिंग बहुत गंभीर है", complaints about rising heat framed as a trend, "plant trees or our kids will suffer", blaming humans/governments for environmental damage |
| **Against** | The author denies it, calls it a hoax/conspiracy/exaggeration, or downplays it as unimportant | "সব মিথ্যা", "it was always hot", "first fix poverty, global warming later", sarcasm about cold weather |
| **None** | No stance can be inferred: unrelated, spam, neutral information, or neutral questions | "next video kab aayega?", weather forecasts, "what is the difference between climate change and global warming?" |

## 1.3 Why this is not ordinary text classification

1. **Stance ≠ sentiment.** "I'm so scared, Delhi crossed 50°C 😰" is negative in sentiment but **Favour** in stance. "Haha global warming in this cold? 😂" is positive in sentiment but **Against**. Sentiment features alone mislead.
2. **Stance ≠ topic.** Many comments mention heat, rain or pollution but take no position ("Mumbai rains start tomorrow", **None**). Others take a position without naming climate at all ("ped kat rahe hain, phir bhi log samajh nahi rahe", **Favour**). The model has to infer the relation to the claim.
3. **The target is implicit.** YouTube comments are replies to a video, and the claim is almost never restated. A stance is often expressed toward a *sub-target* (activists, governments, trees, heat), and the label must be inferred from that.
4. **Negation and polarity flips.** "Those who say global warming *isn't* real are wrong" is **Favour** despite containing a denial phrase. Hindi नहीं / ना / मत and Bengali না / নয় / নেই (often *post-verbal* in Bengali) flip polarity at varying positions.
5. **Sarcasm and rhetorical questions** reverse literal meaning ("हाँ हाँ, दुनिया कल ही खत्म हो जाएगी 🙄", **Against**).
6. **Class imbalance.** In every public climate-stance set, *Against* is the minority class: 17% in GWSD and only 5% in SemEval-2016. Macro-F1 punishes ignoring it.

## 1.4 The specific difficulty here: cross-lingual and cross-domain shift

The AISoME 2026 call for participation (FIRE 2026) states that **the publicly available training data is English, while the test data is Hindi and Bengali YouTube comments**. So there are three shifts at once:

| Shift | Source (train) | Target (test) |
|---|---|---|
| Language | English | Hindi, Bengali (plus romanised and code-mixed variants) |
| Domain / genre | news sentences (GWSD), tweets (SemEval) | YouTube comments |
| Register | edited or semi-edited | very noisy: spelling variants, elongation, emoji, missing punctuation |

Our first experiment quantifies this ([06_experiment_log.md](06_experiment_log.md)). 99.5% of word types in Hindi/Bengali comments never occur in the English training data. A TF-IDF model that scores 0.60 macro-F1 on English falls to **0.13 (Bengali) and 0.19 (Hindi)**.

## 1.5 Linguistic challenges in Hindi and Bengali social-media text

| Phenomenon | Example | Why it hurts |
|---|---|---|
| **Romanisation** (Hinglish/Banglish) | "garmi bohot zyada hai", "gorom e mara jacchi" | No standard spelling; Devanagari-trained vocabularies fragment it into many sub-words |
| **Script-level code-mixing** | "Climate change का सबसे बड़ा असर…", "Global warming এর জন্য…" | Tokens from two scripts in one sentence; English carries the topic words |
| **Spelling variation** | बहुत / बहोत / bahut / bohot / bht; উষ্ণায়ন / উস্নায়ন | Sparse features; out-of-vocabulary |
| **Unicode variation** | Bengali য় as one code point (U+09DF) or as য + nukta; ZWJ/ZWNJ; Devanagari vs ASCII digits | Strings that look identical do not match; we apply NFC and digit normalisation |
| **Morphology** | Hindi postpositions (का/की/के), Bengali agglutinative suffixes (-টা, -গুলো, -এর) | Many surface forms per lemma |
| **Bengali negation** | verb + না, নয়, নেই, নি (post-verbal); নি marks past negation | Negation scope is unlike English |
| **Emoji as stance markers** | 🙄😂 (mocking), 🙏😢 (concern) | Carry stance that text tokens lack |
| **Elongation / shouting** | "bahuuuut", "!!!!" | Inflates vocabulary |
| **Implicit, culturally grounded references** | "दादा जी के ज़माने में भी गर्मी थी", "পুকুর বুজিয়ে ফ্ল্যাট" | Needs world knowledge |

## 1.6 The research questions we commit to

* **RQ1:** How far can *zero-shot cross-lingual transfer* (English training data, multilingual encoder) go on Hindi and Bengali?
* **RQ2:** Does *translate-train* (machine-translating English data into Hindi and Bengali) improve over RQ1, and does it help or hurt robustness to romanised and code-mixed text?
* **RQ3:** Does *noise-aware augmentation* (romanisation, code-mix substitution, typos, emoji) close the gap on noisy inputs, measured with a CheckList-style robustness suite?
* **RQ4:** Is *stance-as-NLI* (zero-shot or NLI-initialised fine-tuning) better at implicit stance and negation than plain classification?
* **RQ5:** Do ensembles of complementary models (NLI + fine-tuned encoders) improve macro-F1 and calibration?
