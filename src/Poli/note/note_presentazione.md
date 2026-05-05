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
