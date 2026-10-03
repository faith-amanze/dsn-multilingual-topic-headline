# Multilingual Topic Classification & Headline Generation for Nigerian Languages

One **mT5-base** model, fine-tuned for two tasks at once (topic classification and headline generation) on news text in **Hausa, Igbo, Yoruba and Nigerian Pidgin**. Built for the DSN Bootcamp Hackathon 2026 (LLM/Agent Track) on Kaggle.

Kaggle notebook: https://www.kaggle.com/code/faithamanze/nigerian-languages-topic-headline-mt5-base

## Task and metric

Each article gets two predictions: a **topic** (one of 7 labels: business, health, politics, religion, sports, entertainment, technology) and a **headline**.

- `MacroF1`: macro-F1 on topic, pooled across languages
- `GenScore` = 0.5 x ROUGE-L + 0.5 x BERTScore (multilingual BERT, layer 9)
- `Final` = 0.5 x MacroF1 + 0.5 x GenScore

## Results (dev set)

Dev is a 12% hold-out of `train.csv`, stratified by language (728 rows), because no separate dev file was provided.

| Language | Baseline Final | Fine-tuned MacroF1 | Fine-tuned GenScore | Fine-tuned Final |
|----------|---------------|--------------------|---------------------|------------------|
| **All (pooled)** | 0.2520 | 0.8709 | 0.5129 | **0.6919** |
| Hausa (hau) | 0.2571 | 0.8849 | 0.5484 | 0.7167 |
| Igbo (ibo) | 0.2534 | 0.8354 | 0.5049 | 0.6701 |
| Pidgin (pcm) | 0.2907 | 0.9582 | 0.5127 | 0.7355 |
| Yoruba (yor) | 0.2379 | 0.8967 | 0.4658 | 0.6812 |

The baseline predicts the majority label and uses the first 12 words of the article as the headline. Full numbers are in `results/`.

## Method

| | |
|---|---|
| Model | `google/mt5-base`, 582,401,280 params after tying shared/encoder/decoder embeddings (about 966M untied) |
| Training | Full fine-tune, fp32, Adafactor (lr 1e-3, constant, no scaling), linear warmup (6%) + decay |
| Schedule | 8 epochs, batch size 4, gradient accumulation 4, max input 320 tokens, max target 40 tokens |
| Multi-task | One model. Prompts are `topic <lang>: <text>` and `headline <lang>: <text>`; both tasks are interleaved in the training set |
| Checkpointing | Best epoch by dev Final (epoch 8) is kept in memory and saved to disk |
| Decoding | Topic: greedy, max 6 new tokens, forced onto the 7 labels (fallback: majority label). Headline: beam search (4), no-repeat 3-gram, max 32 new tokens, never blank (fallback: first 12 words) |
| Hardware | Kaggle T4, about 238 minutes total |

Dev Final by epoch: 0.646, 0.669, 0.676, 0.678, 0.662, 0.690, 0.688, 0.693. Train loss kept falling (avg 4.43 to 0.86) while dev gains flattened after about epoch 4, so more epochs would likely not help much.

## Findings

- **Topic classification is the easier task.** MacroF1 rose from 0.05 to 0.87, while GenScore only moved from 0.46 to 0.51.
- **Headline quality has a low ceiling.** Reference headlines barely overlap with their source articles (mean ROUGE-L 0.062), so they are paraphrased rather than extracted. That limits what ROUGE-based scoring can reward.
- **Pidgin scores highest on topic (0.958 MacroF1); Igbo scores lowest (0.835).** Yoruba has the weakest headline score (0.466). Reasons are untested; plausible candidates are tokenizer coverage, training-set size (Hausa has the most data, Pidgin the least) and Pidgin's similarity to English.
- **Headlines are capped at 32 generated tokens**, so many predictions end mid-sentence.
- **Caveat:** the best checkpoint was chosen on the same dev set used for reporting, so the dev numbers are slightly optimistic.

## Examples (dev set)

| Lang | Article (truncated) | Gold topic | Pred topic | Gold headline | Pred headline |
|---|---|---|---|---|---|
| Hausa | Barcelona za ta koma buga wasanninta na gida a Olympic Stadium a 2023-24... | sports | sports | Barcelona za ta koma buga wasa a Olympic Stadium daga 2023-24 | Barcelona za ta koma buga wasannin gida a 2024-25 |
| Hausa | Latsa alamar lasifikar da ke sama domin sauraren Bayanin Dahiru Bauchi kan rokon Ganduje... | politics | religion | Sheikh Dahiru Bauchi ya gargadi Ganduje kan sabbin masarautu | Bidiyon hirar da Sheikh Dahiru Bauchi kan hadisin Ganduje ya janye sabbin masarautu |
| Igbo | Ọ bụrụ na a jụọ ụfọdụ ndị ntorobịa taa onye tiri egwu a kpọrọ "Tetanụ n'ụra"... | entertainment | entertainment | Christy Essien-Igbokwe: Nwaanyị a lụtara na mba bịara bụrụ Ada Igbo ji eme ọnụ | 'Tetanụ n'ụra' - Christy Essie-Igbokwe |
| Igbo | Lai Mohammed bụ minista mgbasaozi Naịjirịa suru akara akụkọ ụgha... | politics | business | Lai Mohammed ọ rịorọ Obi Cubana ego maka ịkwụ ụgwọ Naịjirịa ji? Lee ihe anyị ma | Cubana: Lai Mohammed rịọrọ Obi Cubana bụ onye azụmahịa na-ewu kamgbe mwụcha izu ụk *(cut off at 32 tokens)* |
| Pidgin | As Nigeria draw Egypt, Guinea Bissau and Sudan for Afcon... | sports | sports | Afcon 2021 draw: Nigeria vs Egypt go open Group D, five oda times di two teams don meet | Nigeria vs Egypt: Prediction, time & how to watch di Afcon qualifier |
| Pidgin | Game of Thrones actor, Hafthor Bjornsson don set world deadlifting record... | sports | entertainment | Hafthor Bjornsson: Game of Thrones actor breaks world deadlifting record | Bjornson set world deadweight record as im lift 1000kg |
| Yoruba | Arun Coronavirus ti tan de, o kere tan, ọgọrin orilẹ-ede... | health | health | Coronavirus symptoms: Kí ni àwọn àpẹẹrẹ àrùn yìí, àti pé báwo ni mo ṣe leè dáàbò bo ara mi? | Coronavirus tips: Wo àwọn ohun tó yẹ kí o mọ̀ nípa àrùn Coronavirus |
| Yoruba | Ajọ to n ja fun ẹtọ ọmọniyan nidi ọrọ aje ati ijiyin isẹ iriju ẹni, SERAP... | politics | health | Muhammadu Buhari: Ọ̀pọ̀ ọmọ Naijiria ló ń jìyà nítorí ìwà àjẹbánu tó wà ní ẹ̀ka ètò ìlera - SERAP | SERAP: Aàrẹ Buhari lẹ́jọ́ lórí ikuna rẹ̀ láti ṣe ìwàdìí bíl *(cut off at 32 tokens)* |

Errors cluster around adjacent categories — politics confused with religion, business, or health — rather than unrelated ones. Two headlines above are visibly truncated by the 32-token generation cap (see Limitations).

## Gotchas

- **Data schema differs from the brief.** `train.csv` columns are `category, headline, text, url, id, split, lang`; `test.csv` has `id, lang, text`. The notebook renames them to `label` and `language`.
- **No dev split** is provided, so one is carved out of train, stratified by language.
- **Tied embeddings** cut mT5-base from about 966M to 582M params. Loading prints "tied weights" warnings; they are expected.
- The displayed training-time dev score (0.6934, beam 6) differs slightly from the final evaluation (0.6919, beam 4) because the beam settings differ.

## Reproduce

1. Open the notebook on Kaggle with the competition data attached, GPU set to T4 and Internet on.
2. Run all cells. Training takes about 4 hours on a T4.
3. Output: `submission.csv` (two rows per test id: `<id>_topic` and `<id>_headline`) and the best checkpoint in `/kaggle/working/mt5_best`.

## Files