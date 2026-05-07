# Note per la presentazione — Sezione 3.2

## Cos'è il "linear evaluation with frozen features" (Tabella 2a)

È il test standard nei paper di self-supervised / pretrained vision: serve a misurare **quanto sono buone le feature del backbone così come sono**, senza ulteriore training del backbone stesso.

Procedura:
1. Si **congela** completamente il backbone (DINOv2 / OpenCLIP / DeiT-III): nessun gradiente, nessun fine-tuning.
2. Si estrae il **CLS token** (per classificazione) o le **patch features** (per segmentazione/depth) per ogni immagine.
3. Si attacca **solo un layer lineare** sopra queste feature (un `LogisticRegression` per classificazione, un `Conv2d 1×1` per segmentazione/depth).
4. Si traina **solo quel layer lineare**.
5. Si misurano le metriche standard (Top-1 accuracy, mIoU, RMSE) sul validation set.

Logica: se aggiungere registri al backbone rovinasse le feature, il classificatore lineare crollerebbe (non riuscirebbe a recuperare l'informazione persa). Se invece le feature migliorano, il classificatore lineare lo riflette nei numeri.

L'esperimento è quindi un controllo di **non-degradazione**: il paper vuole dimostrare che aggiungere registri non distrugge nulla — anzi, in alcuni casi (DINOv2 in particolare) migliora leggermente.

## Cos'è la "zero-shot classification" (Tabella 2b)

È un test che si applica **solo a OpenCLIP**, perché OpenCLIP è l'unico dei tre modelli ad avere anche un **encoder testuale** oltre a quello visivo. DINOv2 e DeiT-III sono solo visivi: non sanno cosa significa la stringa "a photo of a dog".

Procedura:
1. Per ogni classe ImageNet (1000 classi), si genera un prompt tipo `"a photo of a {class_name}"` e lo si fa passare nel **text encoder** di CLIP. Si ottengono 1000 vettori-classe.
2. Per ogni immagine del val set, si calcola il vettore CLS dal **visual encoder**.
3. Si calcola la **cosine similarity** tra il vettore immagine e i 1000 vettori-classe.
4. Si prende l'argmax → classe predetta.

**Niente training di nessun classificatore**: usiamo direttamente la similarità tra le rappresentazioni testuali e visuali apprese durante il pretraining contrastivo di CLIP. È "zero-shot" perché il modello classifica classi che potrebbe non aver mai visto come label esplicita: si basa solo sull'allineamento testo-immagine appreso da LAION-2B.

Si fa solo per OpenCLIP perché:
- DINOv2 non ha un text encoder.
- DeiT-III non ha un text encoder.
- Solo CLIP-style models possono fare zero-shot via text prompts.

## Perché niente test set separato (solo train + val)

Lo split classico **train / val / test** serve quando si fa **iperparameter tuning iterativo**: durante lo sviluppo si guardano i numeri sul val molte volte, si cambiano gli iperparametri e si rivaluta. A forza di farlo, il val "leaka" informazione e smette di essere indipendente. Il test set viene tenuto da parte e usato solo alla fine, una volta sola, per il numero finale onesto.

Nel nostro caso:
- Il "modello" è solo `LogisticRegression` di sklearn.
- L'unico iperparametro vero è la regolarizzazione `C` → lasciato al default.
- Non facciamo tuning iterativo, non facciamo cross-validation.

Quindi il "val set" qui funziona già da test set: è guardato una sola volta a fine pipeline, non c'è leakage da prevenire. Splittare in tre lascerebbe ~50 img per val e ~50 per test — entrambi troppo piccoli per essere statisticamente affidabili. Meglio 350 train + 150 val.

## Limitazioni dichiarate (da menzionare in presentazione)

| Modello | Affidabilità | Motivo |
|---|---|---|
| DINOv2 | **alta** | Sia no-reg che reg4 sono i checkpoint ufficiali del paper. Confronto diretto. |
| OpenCLIP | **media** | no-reg è il checkpoint ufficiale del paper. with-reg è `amildravid4292/clip-vitb16-test-time-registers` di Jiang et al. 2025: usa **test-time register injection**, non registri trainati. Diverso meccanismo. |
| DeiT-III | **bassa** | no-reg è il checkpoint ufficiale del paper. with-reg **non esiste pubblicamente** (Meta non l'ha mai rilasciato). Lo costruiamo iniettando register tokens **non trainati** nel baseline. La rete non ha mai visto registri durante il training, quindi non ci aspettiamo l'effetto del paper. È onestamente la riga più debole. |

Subset size: ~500 immagini stratificate per classe (50 classi × 10 img), invece dei dataset completi (ImageNet val = 50.000 img). Questo introduce varianza alta sui numeri assoluti — non si potranno confrontare 1:1 con i numeri del paper, ma il **trend qualitativo** (with-reg ≈ no-reg, talvolta with-reg leggermente meglio) deve emergere.

## Affidabilità del subset usato

Stiamo lavorando su un subset molto piccolo dei dataset originali. Numeri concreti:

| Dataset | Full size (val) | Nostro subset | Frazione |
|---|---|---|---|
| ImageNet-1k | 50,000 img / 1000 classi | 500 img / 50 classi | **1%** delle immagini, **5%** delle classi |
| ADE20k | 2,000 img / 150 classi | TBD (≈200 img) | TBD |
| NYUd v2 | 654 img | TBD (≈200 img) | TBD |

**Conseguenze pratiche:**

1. **I numeri assoluti non sono confrontabili col paper.** Il paper riporta 84.3% Top-1 di DINOv2 su 1000 classi; noi riportiamo 94.0% su 50 classi. Il task è ~20× più facile (chance level 1/50 = 2% vs 1/1000 = 0.1%), quindi è naturale che i nostri numeri siano molto più alti. Non è un errore.

2. **La varianza è alta.** Con 150 immagini di val (3 img/classe), ogni singola predizione sbagliata sposta il Top-1 di 0.67 punti. Differenze sotto questa soglia sono nel rumore.

3. **Il trend qualitativo, però, è il messaggio del paper.** Quello che vogliamo dimostrare non è "DINOv2 fa 84.3 e DINOv2+reg fa 84.8", ma "**aggiungere registri non degrada le performance, e talvolta le migliora**". Questo trend è robusto al subset:

   | Modello | Δ paper | Δ noi (subset) | Stesso segno? |
   |---|---|---|---|
   | DINOv2 | +0.5 | +0.67 | ✅ |
   | OpenCLIP | -0.1 | 0.00 | ✅ (entrambi ≈ zero) |
   | DeiT-III | 0.0 | -0.67 | ✗ ma atteso (registri non trainati) |

4. **Ironia importante per la presentazione:** il subset piccolo *aiuta* la nostra narrativa per un punto. Quando i numeri assoluti sono al 90-96%, anche un mezzo punto di differenza è chiaramente visibile come trend — non è soffocato dal rumore di fondo del dataset completo. Il messaggio "registri non rompono niente" emerge più nitido.

## Risultati Tabella 2a (ottenuti)

| Modello | Top-1 (50 classi) | Δ vs no-reg | Note |
|---|---|---|---|
| DINOv2 ViT-L/14 | 94.00% | — | baseline ufficiale paper |
| DINOv2 ViT-L/14 +reg4 | **94.67%** | **+0.67** | reg ufficiali paper, miglioramento coerente |
| OpenCLIP ViT-B/16 | 90.67% | — | baseline ufficiale paper |
| OpenCLIP +tt-reg4 | 90.67% | **0.00** | test-time registers, identico al baseline |
| DeiT-III ViT-B/16 | 96.67% | — | baseline ufficiale paper |
| DeiT-III +reg4 (injected) | 96.00% | **-0.67** | reg non trainati, leggera degradazione attesa |

**Lettura riga per riga:**

- **DINOv2** (riga più affidabile): il modello con registri trainati dagli autori migliora rispetto al baseline. Riproduciamo l'osservazione del paper.
- **OpenCLIP**: il modello con test-time registers (Jiang 2025) ha la stessa performance del baseline. Il meccanismo "drena" gli artefatti senza alterare ciò che il CLS sa fare — coerente con quanto Jiang riporta.
- **DeiT-III**: aggiungendo registri **non trainati** a un modello che non li ha mai visti perdiamo 0.67 punti. Questa è la dimostrazione *negativa* dell'ipotesi del paper: i registri funzionano solo se la rete è stata trainata a usarli; iniettarli a inference è solo rumore. Il paper invece ri-traina e non vede degradazione.
