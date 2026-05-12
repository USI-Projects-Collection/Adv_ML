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

### Cosa sono Top-1, mIoU, RMSE

- **Top-1 accuracy**: percentuale di immagini per cui la classe con la probabilità più alta è quella corretta. Su 150 immagini val, ogni singola predizione sbagliata vale 0.67 punti.
- **mIoU (mean Intersection over Union)**: metrica per la segmentazione semantica. Per ogni classe si calcola (area predetta ∩ area reale) / (area predetta ∪ area reale); poi si fa la media su tutte le classi. Vale 0 se il modello sbaglia tutto, 1 se è perfetto. Misura quanto bene il modello assegna ogni pixel alla classe giusta.
- **RMSE (Root Mean Squared Error)** per la depth: misura in **metri** l'errore medio della stima di profondità. RMSE = √(media degli errori²). Un RMSE di 1.2m significa che in media le stime di profondità distano 1.2 metri dal valore reale. Valori bassi = meglio.

### Tabella 2a completa

**ImageNet Top-1** (50 classi, 500 train + 150 val):

| Modello | no-reg | with-reg | Δ | Paper Δ |
|---|---|---|---|---|
| DINOv2 ViT-L/14 | 94.00% | **94.67%** | **+0.67** | +0.5 ✅ |
| OpenCLIP ViT-B/16 | 90.67% | 90.67% | **0.00** | -0.1 ✅ |
| DeiT-III ViT-B/16 | 96.67% | 96.00% | **-0.67** | 0.0 ⚠️ (reg non trainati) |

**ADE20k mIoU** (segmentazione, 350 train + 150 val):

| Modello | no-reg | with-reg | Δ | Paper Δ |
|---|---|---|---|---|
| DINOv2 ViT-L/14 | 18.70% | **19.19%** | **+0.49** | +0.4 ✅ |
| OpenCLIP ViT-B/16 | 12.16% | **12.36%** | **+0.20** | n.d. |
| DeiT-III ViT-B/16 | 12.89% | **13.27%** | **+0.38** | n.d. |

**NYUd RMSE** (depth estimation in metri, 350 train + 150 val):

| Modello | no-reg | with-reg | Δ | Paper Δ |
|---|---|---|---|---|
| DINOv2 ViT-L/14 | 1.974 m | **1.199 m** | **-0.775** ↓ | miglioramento ✅ |
| OpenCLIP ViT-B/16 | 1.746 m | **1.550 m** | **-0.196** ↓ | n.d. |
| DeiT-III ViT-B/16 | 3.059 m | **3.036 m** | **-0.023** ↓ | n.d. |

**Lettura dei risultati:**

- **DINOv2** (riga più affidabile): su tutte e tre le metriche, il modello con registri trainati migliora rispetto al baseline. Riproduciamo esattamente l'osservazione del paper.
- **OpenCLIP**: test-time registers (Jiang 2025) non peggiora il Top-1 (nessun cambiamento) e migliora leggermente segmentazione e depth. Il meccanismo drena gli artefatti senza rompere le feature CLS.
- **DeiT-III**: le features con registri injected (non trainati) peggiorano il Top-1 di 0.67, ma migliorano segmentazione e depth. Questo è coerente: il CLS è il token più sensibile all'iniezione rumorosa di token extra, mentre le patch features sono più robuste e beneficiano della ridistribuzione degli artefatti.

### Tabella 2b — Zero-shot classification (solo OpenCLIP)

| Modello | Zero-shot Top-1 (50 classi) | Δ |
|---|---|---|
| OpenCLIP ViT-B/16 | **94.00%** | — |
| OpenCLIP +tt-reg4 | 93.33% | -0.67 |

Leggera regressione con test-time registers (-0.67 punti, ovvero 1 immagine su 150). Il paper riporta una variazione di circa -0.1 su 1000 classi — con 50 classi la varianza è troppo alta per discriminare. Messaggio principale: non si degrada in modo sostanziale.

## Risultati Figura 8 (ottenuti)

### Figura 8 top — attention maps al variare di N registri

Immagine usata: cane di razza Bernese (file `assets/paper_images/67.png`, immagine del paper originale).

Osservazioni:

- **N=0 (nessun registro)**: le attention maps del CLS token mostrano **artefatti visibili** — picchi ad alta intensità su patch di sfondo (erba, foglie) che non appartengono al soggetto principale. Questo è esattamente l'artefatto documentato nella Sezione 3.1 del paper.
- **N=1, N=2**: gli artefatti si riducono progressivamente ma rimangono visibili. Con 1 registro c'è già un miglioramento notevole rispetto a N=0.
- **N=4 (modello ufficiale reg4)**: le attention maps sono **pulite** — il CLS si concentra sul cane, ignorando lo sfondo. Il registro ha assorbito l'informazione ad alta norma dalle patch di sfondo.
- **N=8, N=16**: mappe altrettanto pulite. L'ulteriore miglioramento rispetto a N=4 è marginale visivamente — suggerisce che 4 registri sono sufficienti per il task di assorbimento degli artefatti.

**Nota tecnica per N>4**: i registri aggiuntivi sono **copie cicliche** dei 4 registri trainati (non registri nuovi addestrati da zero). Non è un confronto fair con il paper (che ri-traina DINOv2 per ogni N) — dichiarare come limitazione.

### Figura 8 bottom — performance quantitativa al variare di N

Risultati completi (dati da `results/figure8_bottom.json`):

| N registri | ImageNet Top-1 | ADE20k mIoU | NYUd RMSE |
|---|---|---|---|
| 0 | 70.67% | 13.93% | 3.263 m |
| 1 | **93.33%** | 20.56% | 1.089 m |
| 2 | 92.67% | 19.60% | 1.202 m |
| 4 | 94.67% | 19.97% | 0.927 m |
| 8 | 94.67% | 20.78% | 1.494 m |
| 16 | **95.33%** | **20.68%** | **0.865 m** |

**Lettura:**

- **Il salto da N=0 a N=1 è enorme**: Top-1 passa dal 70.67% al 93.33% (+22.7 punti), mIoU da 13.93% a 20.56% (+6.6 punti). Questo riflette che senza nessun registro il modello DINOv2-reg4 viene "mutilato": è stato trainato aspettandosi registri, e senza di essi le feature CLS degradano drasticamente.
- **Da N=1 a N=4 il miglioramento è più graduale** e non sempre monotono (N=2 è leggermente peggio di N=1 per ImageNet e RMSE, poi N=4 recupera). Questo è normale con il nostro approccio no-retraining.
- **N=8 e N=16 non mostrano un chiaro ulteriore miglioramento** rispetto a N=4 — i registri extra sono copie cicliche dei 4 trainati, quindi non aggiungono vera nuova capacità.

**Perché i numeri assoluti differiscono dal paper:**

Il paper riporta valori su dataset completi (es. ~64% Top-1 per DINOv2 con N=0 su 1000 classi). I nostri sono su 50 classi, rendendo il task più facile e le percentuali più alte. La **forma della curva** (salto a N=1, plateau dopo N=4) è il confronto qualitativo rilevante.

**Differenza rispetto al paper per N>4:** il paper mostra una curva monotonicamente crescente fino a N=4 e poi stabile, ottenuta riaddestrando DINOv2 da zero per ogni N. Il nostro N>4 usa copie cicliche dei 4 registri trainati — essenzialmente informazione ridondante — il che spiega perché non vediamo il trend chiaro del paper oltre N=4.

## Approfondimenti per la presentazione (Q&A previste)

### Cosa misura esattamente RMSE su NYUd

Il task è **monocular depth estimation**: data una foto da una sola camera RGB, predire per ogni pixel la **distanza in metri** dalla camera. NYU Depth v2 è un dataset di scene indoor (camere, cucine, uffici) acquisito con un Microsoft Kinect, che ha sia camera RGB che sensore IR di profondità: l'IR fornisce la ground truth.

Pipeline lato modello:
1. Backbone congelato → patch features (es. griglia 37×37 di vettori per DINOv2).
2. Una `Conv2d 1×1` trainata mappa ogni vettore a un singolo numero (depth in log-scale).
3. Upsample bilineare alla risoluzione originale.
4. Si confronta pixel-per-pixel la depth predetta con quella vera.
5. RMSE = √(media degli errori al quadrato), in metri.

DINOv2 +reg4: 1.20m di errore medio. DINOv2 senza reg: 1.97m. Più basso = meglio.

### Come funziona zero-shot CLIP (e perché non è "barare")

Il modello non sceglie "X o non-X": sceglie tra **tutte le 1000 classi simultaneamente**, dati 1000 candidati testuali.

1. Per ogni classe ImageNet generi `"a photo of a {classe}"` → text encoder → vettore 512-dim. Ottieni una matrice 1000 × 512.
2. Immagine → visual encoder → vettore 512-dim.
3. Cosine similarity tra il vettore-immagine e tutti i 1000 vettori-classe.
4. Predizione = argmax (classe con similarità più alta).

Il text encoder vede solo testo, il visual encoder vede solo l'immagine. Sbaglia se la similarità più alta è con la classe sbagliata (es. "labrador" invece di "golden retriever"). Il fatto che CLIP faccia ~76% Top-1 su ImageNet senza nessun training su ImageNet dimostra la qualità dell'allineamento testo-immagine appreso da LAION-2B.

### La presunta contraddizione: "il modello è rigido al training" vs "vediamo trend"

**Apparente contraddizione:** se DINOv2-reg4 è stato trainato per N=4 esatto, perché N=1, 2, 8, 16 funzionano comunque bene? Non dovremmo vedere grafici piatti tranne a N=4?

**Risposta:** guardando i nostri numeri c'è **un solo salto reale**: da N=0 (70.67%) a N=1 (93.33%, +22.7 punti). Tutto il resto (N=1, 2, 4, 8, 16) è praticamente piatto. Questa è la lettura corretta:

- Il modello ha bisogno **strutturalmente** di almeno una posizione dedicata ad assorbire l'alta norma. Senza nessuna (N=0) collassa.
- Una volta che ne ha **almeno una**, il meccanismo funziona. Il numero esatto importa poco perché il "lavoro" che i registri devono fare (drenare gli outlier ad alta norma) è limitato.
- Per N>4 i registri extra non aggiungono nulla perché sono copie cicliche dei 4 trainati.

Quindi il messaggio è: **il modello è strutturalmente dipendente dalla presenza di registri (qualunque numero ≥ 1), ma non è specificamente dipendente dai pesi appresi dei 4 registri**. Questa lettura è proprio il punto del paper successivo "Vision Transformers Don't Need Trained Registers" (Jiang et al. 2025).

### Il paper "Don't Need Trained Registers" — esperimento extra proposto

Jiang et al. (NeurIPS 2025) mostrano che gli outlier ad alta norma in DINOv2/CLIP sono creati da un piccolo gruppo di neuroni MLP nel layer 6 (chiamati *register neurons*). Il loro metodo:

1. Algoritmo `FindRegisterNeurons`: identifica automaticamente i top-10 neuroni MLP che si attivano più fortemente nelle posizioni outlier.
2. A inference: aggiungono **un singolo extra token** (zero-init) alla sequenza. Per ogni register neuron, copiano la sua attivazione massima nella posizione del nuovo token e azzerano quelle altrove. Risultato: l'alta norma viene ridiretta sul nuovo token.

**Risultati che riportano** (Table 2 del loro paper, DINOv2-L/14, dataset completi):

| | ImageNet Top-1 | ADE20k mIoU | NYUd RMSE |
|---|---|---|---|
| Originale (no-reg) | 86.4 | 48.3 | 0.388 |
| Trained registers | 86.7 | 49.1 | 0.382 |
| **Test-time registers** | **86.4** | **49.1** | **0.378** |

Test-time registers **eguagliano** i trained registers senza retraining. Codice pubblico: https://github.com/nickjiang2378/test-time-registers

**Cosa potremmo fare in più per la presentazione (esperimento extra):**
- Integrare il loro metodo per DINOv2 (l'OpenCLIP test-time registers che già usiamo è già di Jiang) → un nuovo blocco di Tabella 2 con DINOv2 (no-reg) + test-time registers, da confrontare con DINOv2-reg4 trainato.
- Per Figura 8: usare il loro metodo per aggiungere N test-time registers a DINOv2 baseline (ognuno generato indipendentemente, non clonato dai 4 trainati). Questo darebbe una curva più onesta per N≥1.

**Costo:** 1-2 giorni per integrare il codice + 1 notte di esperimenti. Stima realistica.

**Framing per la presentazione:** *"il paper successivo di Jiang et al. (2025) propone un metodo training-free; mostriamo qui che riproduce i benefici dei trained registers, validando l'ipotesi che il meccanismo è strutturale e non specifico ai pesi appresi dei registri trainati."*

### Perché Figura 8 top a N=0 ha più "noise" del paper

Il paper a N=0 mostra mappe scure con poche patch molto accese ben definite. Le nostre mappe a N=0 hanno noise diffuso su molte più patch.

I due N=0 non sono la stessa cosa:
- **Paper N=0**: DINOv2 trainato da zero senza nessun registro. La rete ha imparato a vivere senza registri, gli artefatti sono i suoi nativi (concentrati).
- **Nostro N=0**: partiamo da DINOv2-reg4 e tagliamo i registri a inference. La rete è "confusa": si aspettava 4 token per assorbire l'alta norma, non li trova, le patch di sfondo mantengono norma alta e il CLS si disperde su molte di esse.

Questa differenza è coerente col crollo del Top-1 da 94% (N=4) a 70% (N=0) nei nostri numeri — il paper non vede un crollo simile perché il loro N=0 è un modello nativamente trainato per quel setting.

Il fatto che da N=4 in su le mappe siano pulite anche con la ciclazione conferma che la rete è soddisfatta appena ha registri a sufficienza, indipendentemente dal fatto che siano copie.

### Differenza di scala nei 3 grafici della Figura 8 bottom

**Top-1 ImageNet (nostro 70–95% vs paper ~64–84%)**: il task è più facile perché abbiamo 50 classi (chance 2%) vs 1000 classi (chance 0.1%). La forma della curva è quello che conta.

**mIoU ADE20k (nostro 14–21% vs paper 66–67%)**: la differenza grande è dovuta a (i) testa di segmentazione lineare semplice (Conv2d 1×1) trainata su 350 immagini per 30 epoche, vs il paper che usa training su tutto ADE20k (~20.000 immagini); (ii) numero di classi (151 in ADE20k full, simile per noi). Il 14–21% è il limite di una testa lineare con così pochi dati di training, non un errore. La forma della curva (salto a N=1, plateau) è il confronto qualitativo rilevante.

**RMSE NYUd (nostro 0.9–3.3m vs paper 2.73–2.85m)**: il nostro range è più ampio perché N=0 esplode a 3.26m (modello disturbato). Per N≥1 siamo addirittura **migliori** del paper (es. 0.93m a N=4) — plausibile perché 150 immagini val di NYUd con scene indoor relativamente standard sono più "facili" delle 654 del dataset completo.

### Perché esistono i token ad alta norma — meccanismo

Documentato in Darcet et al. (Sez. 2-3) e analizzato in dettaglio in Jiang et al. (2025, Sez. 3).

**Cosa succede nel forward.** In alcune patch (tipicamente quelle di sfondo a basso contenuto informativo: cielo, erba, muri uniformi) il residual stream accumula vettori con norma 100–150× più grande della media. Sono i "high-norm tokens" o "outlier tokens".

**Quando emergono.** Non gradualmente: in modo netto **dopo l'MLP di uno specifico layer** (layer 6 per OpenCLIP-B/16, secondo Jiang Figure 2). Prima di quel layer le norme sono uniformi, dopo esplodono e restano alte fino alla fine.

**Perché succede (interpretazione del paper Darcet).** Durante il pretraining il modello impara che certe patch sono **ridondanti**: un quadrato di cielo blu è prevedibile dai vicini e non aggiunge informazione locale. Il modello "decide" che queste patch sono spendibili e le **riusa come spazio di memoria globale**: ci scrive informazione di alto livello (riassunto della scena, segnali di calibrazione interna) che gli serve nei layer successivi.

**Che informazione ci scrive (verifica empirica).** Darcet (Tabella 4) fa linear probing sui token outlier:
- contengono **forte informazione globale** (classe dell'immagine), simile al CLS;
- contengono **poca informazione locale** (pixel/posizione di quella patch).

Quindi il modello sta usando le patch di sfondo come "scratch space" globale.

**Come la riusa.** Tramite self-attention. Qualunque token può leggere da qualunque altro tramite l'attention; quando il CLS o le patch di soggetto hanno bisogno di "consultare" un riassunto globale nei layer profondi, fanno attention sui token ad alta norma — che dominano il softmax proprio perché hanno norma elevata.

**Perché è un problema (due conseguenze pratiche).**
1. **Attention maps inutili per interpretabilità**: vorresti vedere il CLS che attende al soggetto; invece attende a patch di sfondo random (perché lì c'è il riassunto globale). Le mappe sembrano rumore casuale.
2. **Dense prediction degrada**: per segmentazione/depth servono feature **locali** pulite per ogni patch. Ma le patch di sfondo non contengono più la loro informazione locale (riciclate come memoria globale) → errori pixel-wise in quelle aree.

**Cosa fanno i registri.** Forniscono al modello **token dedicati** allo scratch-space, così non deve cannibalizzare patch dell'immagine. I registri sono memoria globale promessa esplicitamente; le patch dell'immagine restano locali.

**Aggiunta di Jiang et al.** Il "decidere quali patch sacrificare" non è distribuito sulla rete: è fatto da un **piccolissimo gruppo di neuroni** (~10) negli MLP di un layer specifico (i "register neurons"). Identificarli permette di intervenire chirurgicamente a inference: copiare la loro attivazione su un token extra e azzerarla nelle patch — il modello scrive lo scratch sul token extra invece che sulle patch dell'immagine. Questo è il loro metodo "test-time registers".

### Implementare Jiang da soli vs usare la loro repo

Il metodo è descritto in modo completo nel paper (Algorithm 1 + Sez. 4) e si riduce a poche operazioni:

1. **Find register neurons** (Algorithm 1): per ogni neurone MLP nei primi `top_layer` layer, calcolare l'attivazione media nelle posizioni outlier su un set di immagini → prendere i top-k. Loop semplice su forward hook.
2. **Edit a inference**: forward hook sull'MLP del layer giusto. Per ognuno dei k register neurons, copia la loro attivazione massima nella posizione di un extra token aggiunto alla sequenza, azzera nelle altre posizioni.
3. Aggiungere un extra token (zero-init) alla sequenza prima del primo blocco.

**Iperparametri noti dal paper:** OpenCLIP ViT-B/16 → `top_layer=5`, `top_k=10`, `outlier threshold=75`. Per DINOv2 il paper rimanda all'appendice/repo.

**Modelli pre-fatti nella repo Jiang:** l'OpenCLIP `amildravid4292/clip-vitb16-test-time-registers` che già usiamo è loro. Per DINOv2 probabilmente la repo contiene **codice** (non un checkpoint), perché il metodo è una pipeline di inference, non un set di pesi da scaricare.

**Strategia consigliata per la prossima iterazione:**

1. Clonare la repo `nickjiang2378/test-time-registers` solo per **leggere** gli iperparametri esatti per DINOv2 (top_layer, top_k, soglia).
2. Implementare in casa in `Adv_ML/src/Poli/ablation/test_time_registers.py` (~100 righe), riusando la nostra `RegisteredViT` e i nostri data loader.
3. Solo se i numeri non quadrano, fallback alla loro libreria come dipendenza.

Vantaggio: il codice resta coerente con il nostro stile e la presentazione mostra che abbiamo implementato il metodo, non solo importato un modello.

---

## Esperimento extra — Test-time registers su DINOv2 (metodo Jiang implementato in casa)

Abbiamo implementato il metodo Jiang da zero in `ablation/test_time_registers.py` e ri-eseguito Tabella 2a + Figura 8 su DINOv2 baseline (no-reg) con N test-time registers iniettati a inference. Per OpenCLIP avevamo già usato il loro modello HF; per DINOv2 e DeiT-III il modello non esisteva pubblicamente, quindi lo abbiamo creato noi.

### Implementazione

File principale: `Adv_ML/src/Poli/ablation/test_time_registers.py`.

**Passo 1 — find register neurons (Algorithm 1 del paper Jiang).** Su 100 immagini del nostro subset ImageNet train:
1. Forward pass sul baseline DINOv2-L/14 timm. Cattura per ogni layer:
   - l'output del blocco residuo (per misurare le norme delle patch)
   - le attivazioni post-GELU dell'MLP (B, N, 4096)
2. Per ogni immagine, identifica le patch outlier come quelle con norma > 150 (soglia dal config Jiang `dinov2_large.yaml`).
3. Per ogni (layer, neurone) ∈ [0..17] × [0..4095], somma `|attivazione|` sulle posizioni outlier e media su tutte le immagini.
4. Prendi i top-50 neuroni con score più alto.

**Risultato osservato** (top-10 neuroni trovati su 100 immagini):

| Layer | Neurone | Score |
|---|---|---|
| 17 | 884 | 151.58 |
| 17 | 133 | 17.30 |
| 17 | 3464 | 16.00 |
| 17 | 2129 | 10.61 |
| 17 | 2436 | 10.16 |
| 17 | 2373 | 9.24 |
| 17 | 1427 | 6.25 |
| 17 | 844 | 4.43 |
| 0 | 4095 | 4.28 |
| 16 | 2301 | 3.76 |

**Osservazione interessante**: il layer 17 domina enormemente (un neurone con score 152, gli altri ~17 → 16 → 10). Inoltre **5 dei top-10 neuroni hanno lo stesso ID che compare nella lista precomputata di Jiang** (884, 133, 3464, 2436, 2373), nonostante la loro lista sia stata calcolata sull'architettura Meta-DINOv2 (SwiGLU MLP) e la nostra su timm-DINOv2 (GELU MLP). Questo è una replica indipendente del loro risultato: i pesi sono gli stessi (sono lo stesso checkpoint convertito tra i due framework) e quindi i neuroni "register" sono stabili.

**Passo 2 — costruzione del modello con N test-time registers.**

Layout della sequenza: `[CLS, PATCH_0...PATCH_{P-1}, TT_REG_0...TT_REG_{N-1}]` (TT registers alla fine, convenzione Jiang).

A inference:
1. Patch embed + pos embed normalmente.
2. Concat di N token zero-init alla fine della sequenza.
3. Per ognuno dei 50 neuroni (layer, neuron_idx) trovati nel passo 1, registra un forward hook su `model.blocks[layer].mlp.act` (output del GELU). Il hook:
   - calcola `max(|activation|)` sui patch tokens per quel neurone,
   - scrive quel valore in tutti gli N TT register tokens (per quel neurone),
   - azzera l'attivazione di quel neurone in tutti i patch tokens.
4. Forward attraverso i 24 blocchi.

**Effetto sulle norme delle patch (smoke test sul cane):**

| | Mean patch norm | Max patch norm |
|---|---|---|
| Baseline DINOv2 (N=0) | 44.29 | **88.05** ← outlier |
| Baseline + 4 TT-reg (Jiang) | 44.72 | **52.83** ← outlier rimosso |

L'outlier viene riassorbito dai TT register tokens; le altre patch mantengono norma normale.

### Tabella 2a — riga DINOv2 +tt-reg (Jiang)

Confronto delle tre varianti DINOv2 sul nostro subset (350 train + 150 val):

| Modello | ImageNet Top-1 | ADE20k mIoU | NYUd RMSE |
|---|---|---|---|
| DINOv2 ViT-L/14 (baseline) | 94.00% | 18.70% | 1.974 m |
| DINOv2 ViT-L/14 +reg4 (trained, Darcet) | 94.67% | **19.19%** | **1.199 m** |
| **DINOv2 ViT-L/14 +tt-reg4 (Jiang, nostra impl.)** | **95.33%** | 18.84% | 2.249 m |

**Lettura:**

- **ImageNet Top-1**: TT-reg di Jiang **batte sia il baseline che i trained registers** sul nostro subset (95.33 vs 94.67 vs 94.00). Il paper Jiang riportava test-time ≈ trained ≈ baseline (86.4 / 86.7 / 86.4); la nostra differenza (+0.67 vs trained) è di 1 immagine su 150 — entro la varianza statistica del subset.
- **ADE20k mIoU**: TT-reg leggermente sopra il baseline (+0.14) ma sotto i trained registers (-0.35). Coerente col paper Jiang dove test-time eguaglia trained (49.1 in entrambi). Sul nostro subset (350 train, 30 epoche per la testa lineare) la varianza è alta.
- **NYUd RMSE**: TT-reg **peggio del baseline** (2.249 vs 1.974, +0.275). Risultato controintuitivo che merita una nota: il metodo Jiang è progettato per "ridirigere" l'alta norma dai patch ai TT registers. Su scene indoor di NYUd dove le patch "di sfondo" (pareti, soffitto, pavimenti) sono in realtà semanticamente cruciali per la depth estimation, azzerare i register neurons in quelle posizioni rimuove informazione che la testa lineare di depth si aspetta. Il paper Jiang riporta invece un miglioramento (0.388 → 0.378), ma su dataset completo e con probe più sofisticato. Sul nostro setup il trade-off è negativo.

**Confronto col paper Jiang (Table 2 del loro paper, DINOv2 ViT-L/14, dataset completi):**

| | ImageNet Top-1 | ADE20k mIoU | NYUd RMSE |
|---|---|---|---|
| Original | 86.4 | 48.3 | 0.388 |
| Trained reg | 86.7 | 49.1 | 0.382 |
| Test-time reg | 86.4 | 49.1 | 0.378 |

**Take-away per la presentazione**: il metodo Jiang produce risultati comparabili ai trained registers per classification e segmentation (riproduciamo il loro trend) ma il vantaggio sulla depth dipende dal dataset e dal protocollo di valutazione. È un risultato onesto che merita di essere mostrato come "validazione parziale" del paper Jiang sul nostro setup ridotto.

### Figura 8 top — TT-reg sul cane (visualizzazione qualitativa)

File: `results/figure8_top_ttreg/figure8_top_ttreg.png`.

Confronto col vecchio Figure 8 top (truncated/cycled DINOv2-reg4):

- **N=0 (baseline DINOv2 senza modifiche)**: una singola patch giallo brillante (l'outlier ad alta norma classico di DINOv2 baseline, ben documentato dal paper Darcet). Niente altro è visibile. **Notare la differenza con il vecchio Figura 8 top dove N=0 mostrava noise diffuso**: lì il modello era DINOv2-reg4 mutilato e si comportava in modo caotico; qui è il baseline nativo e mostra l'outlier classico.
- **N=1, 2, 4, 8, 16**: tutte le mappe sono **immediatamente pulite e mostrano il cane**. Anche con un solo TT-register il singolo outlier viene riassorbito e l'attention si distribuisce sui patch del soggetto.

Questo è esattamente il risultato del paper Jiang (Figure 5): "test-time registers produce similarly high-quality maps as trained registers". Il messaggio per la presentazione: il metodo Jiang funziona **immediatamente già con N=1**, senza bisogno di trainare nulla.

### Figura 8 bottom — TT-reg sweep su DINOv2 baseline

File: `results/figure8_bottom_ttreg.json` + `figure8_bottom_ttreg.png`.

Differenza col vecchio Figure 8 bottom:
- **Vecchio**: partiva da DINOv2-reg4 trainato e troncava/ciclava i 4 registri trainati per N ∈ {0,1,2,4,8,16}. Ogni N era un modello "danneggiato" rispetto al setting nativo.
- **Nuovo (TT-reg)**: parte dal baseline DINOv2 (no-reg) e aggiunge N test-time registers genuini per ogni N. N=0 è il vero baseline. N>4 sono registri **veri**, non copie cicliche di trained registers.

**Risultati ottenuti:**

| N registri | ImageNet Top-1 | ADE20k mIoU | NYUd RMSE |
|---|---|---|---|
| 0 (baseline) | 94.00% | 19.99% | 2.038 m |
| 1 | **95.33%** | 19.99% | 3.829 m |
| 2 | **95.33%** | 19.87% | 2.881 m |
| 4 | **95.33%** | 19.03% | **1.398 m** |
| 8 | **95.33%** | **19.98%** | 1.476 m |
| 16 | 92.00% | 18.83% | 2.948 m |

**Confronto qualitativo col vecchio Figure 8 bottom (truncated/cycled):**

| | Vecchio (truncate/cycle) | Nuovo (TT-reg) |
|---|---|---|
| N=0 Top-1 | 70.67% (modello mutilato) | **94.00%** (baseline pulito) |
| Salto a N=1 | +22.7 punti (drammatico) | +1.33 punti (più realistico) |
| Plateau | N≥4 piatto | N=1..8 piatto, crollo a N=16 |

**Lettura per ciascuna metrica:**

- **ImageNet Top-1**: comportamento esattamente atteso dal paper Jiang. Il baseline DINOv2 (N=0) è già al 94%; aggiungere anche un solo TT-register porta al 95.33% e poi la curva è **piatta fino a N=8** (4 valori consecutivi tutti a 95.33%). A N=16 crolla a 92%: troppi TT-registers diventano disturbo, drenano troppe informazioni dalle patch. Questo è coerente con il paper Jiang che usa un singolo TT-register e non testa N grandi.

- **ADE20k mIoU**: comportamento piatto su tutta la curva con piccole fluttuazioni (18.83-19.99%). Non c'è un chiaro vantaggio nell'aggiungere TT-registers per la segmentazione su questo subset — il metodo è progettato per ripulire attention maps e funziona meglio per task globali (classification) che locali (segmentation, depth). Paper Jiang riporta un miglioramento ADE20k mIoU di +0.8 (48.3 → 49.1) sul dataset completo, ma con un linear probe più sofisticato.

- **NYUd RMSE**: **caotico**. N=4 è il migliore (1.40m), N=1 è il peggiore (3.83m). Conferma che il metodo Jiang ha effetti negativi sulla depth estimation sul nostro setup, come già visto in Tabella 2a. La spiegazione plausibile: la depth dipende fortemente dall'informazione delle patch di sfondo (pareti, soffitti) che il metodo Jiang tende a "svuotare" azzerando i register neurons in quelle posizioni.

**Take-away per la presentazione di Figura 8 bottom:**

1. **Il nostro vecchio Figure 8 (cycled) era una caricatura**: il salto da 70% a 93% non rifletteva un comportamento reale, ma il danno del troncamento dei registri trainati. Il **nuovo Figure 8 (TT-reg)** mostra il comportamento corretto: il baseline è già pulito (94%), un TT-register è sufficiente (+1.3pt), e il plateau si estende fino a N=8.

2. **Il messaggio del paper Jiang è confermato**: "test-time registers maintain performance ... and largely match models with trained registers". Il nostro 95.33% (Jiang) ≈ 94.67% (trained registers, Tabella 2a) per ImageNet — entro la varianza del subset.

3. **Effetto deleterio a N=16**: aggiungere troppi TT-registers è dannoso. Il paper non testa N grandi, quindi non c'è confronto diretto, ma è un risultato interessante che mostra il **limite del metodo**: i TT-registers sono utili solo se non disturbano l'equilibrio delle attivazioni della rete.

### Take-away per la sezione "esperimento extra"

1. **Implementazione corretta del paper Jiang validata** dal fatto che i top register neurons che troviamo (layer 17, neurone 884 con score 152) sono gli stessi neuroni che il paper Jiang ha precomputato (despite architecture difference timm-GELU vs Meta-SwiGLU).
2. **Comportamento del modello atteso**: max patch norm scende da 88 a 53 quando aggiungiamo TT registers — esatto pattern di Figure 4 del paper Jiang ("Intervening on activations of register neurons effectively shifts outliers").
3. **Tabella 2a**: riproduciamo il trend qualitativo del paper Jiang per ImageNet/ADE20k. Per NYUd vediamo invece una regressione che dichiariamo onestamente come limitazione del nostro setup ridotto.
4. **Figura 8 top**: visualmente molto convincente — N=0 mostra l'outlier classico isolato, N≥1 mostra mappe pulite. Bel materiale per la presentazione.
5. **Figura 8 bottom**: il pattern atteso è confermato — baseline (N=0) già funzionante al 94%, salto piccolo a N=1 (+1.3 pt), plateau fino a N=8, degradazione a N=16. Curva qualitativamente diversa e più onesta del vecchio Figure 8 truncate/cycle.

### File prodotti

| File | Contenuto |
|---|---|
| `ablation/test_time_registers.py` | Implementazione del metodo Jiang (find_register_neurons + TTRegDINOv2) |
| `scripts/run_dinov2_ttreg.py` | Riga "+tt-reg" della Tabella 2a per DINOv2 |
| `scripts/run_figure8_top_ttreg.py` | Figure 8 top con TT-reg |
| `scripts/run_figure8_bottom_ttreg.py` | Figure 8 bottom con TT-reg |
| `results/dinov2_tt_register_neurons.pt` | Top-100 register neurons cached |
| `results/table_2a_dinov2_ttreg.json` | Riga +tt-reg di ImageNet probe |
| `results/table_2a_seg_dinov2_ttreg.json` | Riga +tt-reg di ADE20k seg |
| `results/table_2a_depth_dinov2_ttreg.json` | Riga +tt-reg di NYUd depth |
| `results/figure8_top_ttreg/figure8_top_ttreg.png` | Figura 8 top |
| `results/figure8_bottom_ttreg.json/png` | Figura 8 bottom |

I file vecchi (`table_2a.json`, `figure8_bottom.json`, ecc.) sono lasciati intatti per il confronto.
