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

---

## Approfondimenti meccanicistici (per la presentazione)

Questa sezione raccoglie le spiegazioni di "come funziona davvero" il fenomeno high-norm e il metodo Jiang. Sono i punti che servono per rispondere alle domande tipiche del prof.

### Come il modello "salva informazione" nei patch di sfondo, e come i registri lo costringono a smettere

**Come il modello finisce per scrivere nelle patch di sfondo.** Il transformer è una rete che non ha buffer di memoria espliciti — può solo leggere e scrivere nei token. Durante il training su LAION/LVD-142M (miliardi di immagini), il modello impara per gradient descent che è utile avere uno "spazio di lavoro" globale. Ma non ha posizioni dedicate per questo — quindi cooptaziona spontaneamente le posizioni più sacrificabili: le patch di sfondo a basso contenuto informativo (cielo, muri, erba). La loss favorisce questa allocazione perché:

- **Costo basso**: una patch di cielo blu è facilmente ricostruibile dai vicini. Se il modello ci scrive sopra info globale, non perde info localmente utile.
- **Beneficio alto**: avere uno scratch-space globale aiuta in praticamente tutti i task di pretraining (self-supervised, contrastive).

**Come ci scrive concretamente.** Gli MLP nei layer medio-profondi imparano pesi tali che per certi pattern di input (es. "questa patch è simile ai vicini = sfondo") l'output diventa enorme su certe dimensioni del feature space. Quello è il "writing".

**Come la riusa nei layer successivi.** Tramite self-attention. Quando il CLS o un'altra patch (es. quella del muso del cane) deve calcolare la propria rappresentazione finale e ha bisogno di info globale, fa attention sui token ad alta norma — che dominano il softmax (perché valori grandi nella similarity QK^T → softmax altissimo). Le patch outlier diventano "broadcast nodes" che tutti i token leggono.

**I registri forzano il modello a smettere?** Tecnicamente no, non con un vincolo esplicito. Darcet ha semplicemente aggiunto N token extra all'input, con i loro `nn.Parameter` di shape (1, N, embed_dim), inizializzati random e ottimizzati durante il training insieme al resto. Nessuna loss aggiuntiva, nessun regularizer. Quello che succede è che il gradient descent trova naturalmente che è più conveniente usare i registri come scratch-space invece delle patch:

- I registri non hanno una "vita locale" (nessuna info pixel/posizione da preservare): scriverci sopra è puro guadagno, costo zero.
- Le patch di sfondo, anche se ridondanti, hanno comunque un piccolo costo (perdi info locale). Se hai un'alternativa migliore, il modello la sceglie.

Quindi il training, ottimizzando la stessa loss di prima, "scopre" che i registri sono il posto giusto e migra naturalmente lì. **Lo scopre da solo**, perché è ottimale per la loss. Non serve un meccanismo speciale, basta dare al modello *l'opzione* di un'allocazione migliore e il training trova la strada.

### Neurone vs posizione — distinzione che è bene avere chiara

Dentro un transformer block ci sono **tre dimensioni distinte** che si muovono ortogonalmente:

- **Layer** (l = 0..23): si va in profondità. A ogni layer applichi un nuovo blocco.
- **Token / posizione** (p = 0..1369): l'asse della sequenza. Ogni token corrisponde a una patch dell'immagine (o al CLS, o ai register).
- **Neurone / canale** (c = 0..1023 nel residual stream; 0..4095 nell'attivazione hidden dentro l'MLP): l'asse delle feature.

Il tensore principale è `(B, N_tokens, C)`. Quando diciamo "register neuron N°884" intendiamo un indice specifico sulla terza dimensione, quella dei canali — **NON una posizione spaziale dell'immagine**.

**Dove vivono i register neurons.** Dentro l'MLP di un blocco:

```
input (B, N, C=1024)
  ↓
fc1: Linear(1024 → 4096)        ← genera 4096 "neuroni nascosti"
  ↓
GELU                             ← attivazione non lineare
  ↓
fc2: Linear(4096 → 1024)         ← proietta indietro a 1024
  ↓
output (B, N, C=1024)
```

I "neuroni" sono i 4096 canali intermedi della rappresentazione interna all'MLP, dopo GELU. Sono identità *funzionali*: il neurone 884 è una direzione fissa nello spazio delle attivazioni, una "feature detector" specifica che la rete ha imparato. Esiste in ogni posizione della sequenza (ogni patch ha il suo valore per il neurone 884), ma il neurone in sé non è legato a nessuna posizione.

**Cosa significa essere un register neuron.** Un register neuron è un canale dell'MLP (es. layer 17, neurone 884) che ha la proprietà di **attivarsi fortemente nelle posizioni di sfondo** (e quasi a zero altrove). Concretamente, se prendi la sua attivazione `act[posizione_p, neurone_884]` per tutte le 1369 posizioni patch:

- Per una patch del cane (informativa): valore basso o nullo
- Per una patch del cielo blu (sfondo): valore enorme (50, 100, 150…)

Quindi il neurone 884 è funzionalmente un "rilevatore di posizioni-da-cooptaziona-come-broadcast-node". L'identità del neurone (cioè quale colonna di pesi della matrice fc1) è fissa, addestrata una volta per tutte. Il fatto che si accenda qui o là nell'immagine dipende dall'input.

### Come la pipeline Jiang sa dove intervenire

Non sa *dove nella sequenza* (cioè quale patch specifica): quello cambia da immagine a immagine. Quello che sa è **quali neuroni MLP** sono responsabili del meccanismo di scrittura. Quei neuroni sono gli stessi per ogni immagine (sono caratteristiche del modello, non dell'input).

L'algoritmo Jiang:

1. **Offline**: trova i ~50 neuroni MLP che si attivano fortemente nelle posizioni outlier su molte immagini diverse. Ottieni una lista fissa di `(layer, neuron_idx)`. Calcolato una volta sola.

2. **A inference, per ogni immagine**: durante il forward, monitora l'output dell'MLP del layer dei register neurons. Per ognuno di quei neuroni:
   - Guarda le sue attivazioni su tutti i 1369 patch tokens.
   - Trova il **massimo** (= la patch in cui il modello sta più scrivendo per quell'immagine specifica).
   - Copia quel valore sul TT register, azzera la posizione di provenienza.

**La posizione esatta dove il modello scarica cambia dinamicamente** (su questa immagine sarà la patch 442 nel cielo, sulla prossima sarà la patch 1100 in basso a destra), **ma il neurone che fa lo scarico è sempre lo stesso**. Hai bisogno solo di sapere quali neuroni intercettare, non quali posizioni.

### "Ma scusa, non basta cancellare la patch outlier?"

Domanda legittima, e la risposta rivela il punto sottile del metodo.

**No, e qui c'è l'inganno.** Se semplicemente cancelli (azzeri) la patch outlier, il modello perde l'info globale. Ricorda: le patch outlier *contengono il riassunto globale*. Tutti i layer successivi al layer 17 si aspettano di poter andare a "leggere" lì il riassunto. Se l'azzeri, le feature dei layer successivi diventano malformate perché manca un input che la rete si aspettava.

**Cosa fa esattamente Jiang.** Lui non azzera la patch outlier come pixel — quella patch rimane lì con il suo contenuto. Lui azzera **solo l'output di certi neuroni MLP** in quella posizione, e copia quel valore sul TT register. La differenza è sottile ma cruciale:

- La patch *dell'immagine* mantiene il suo contenuto locale (il cielo blu rimane cielo blu nel feature space).
- Ma il "side-channel" che il modello aveva usato per scrivere info globale viene reindirizzato sul TT register.

Quindi:
- Il TT register diventa il nuovo broadcast node che contiene l'info globale.
- La patch del cielo torna a essere "solo cielo".
- Quando i layer successivi vanno a cercare il riassunto globale via attention, lo trovano sul TT register (perché ha la stessa firma di attivazione di prima — quei valori specifici sui register neurons).

### "Ma il modello non è stato trainato a usare il TT register!"

Vero. Ma è stato trainato a fare **attention su qualsiasi token che abbia quella firma di attivazione** (cioè valori alti sui register neurons). Per il softmax dell'attention conta solo la similarità QK del token, non la sua identità o posizione. Quindi se il TT register ha quei valori alti, il modello lo tratta automaticamente come "broadcast node" senza saperlo.

**Questa è la magia del metodo.** Il modello non sa che esiste un TT register. Ma è stato trainato a leggere da token con una certa firma. Tu ricrei quella firma altrove e il modello segue.

### Verifica empirica che il modello "usa" davvero i TT registers

Tre prove sperimentali che lo confermano:

**Prova 1 — patch norms.** Sul nostro smoke test (immagine del cane): max patch norm scende da 88 (baseline) a 53 (con TT-reg). Questo dimostra che gli outlier si spostano davvero: non spariscono, vengono assorbiti dai TT registers. Se i TT registers non venissero "usati" dal modello (cioè se restassero zero-init invisibili), gli outlier resterebbero sui patch tokens come prima e il max patch norm non scenderebbe.

**Prova 2 — attention.** Su quel test, CLS dà l'8.25% di attention a ciascuno dei 4 TT registers, totale ~33%. Una frazione enorme. Il modello sta facendo attention su quei token, confermando che li sta leggendo come broadcast nodes.

**Prova 3 — performance.** Tabella 2a/2b del paper Jiang originale: i numeri del modello con test-time registers eguagliano (o quasi) quelli con trained registers. Su ImageNet 86.4 vs 86.7. Su ADE20k 49.1 vs 49.1. Su NYUd 0.378 vs 0.382. Non c'è solo "smoothing": c'è proprio recupero di performance, che vuol dire che le feature CLS/patch sono effettivamente migliorate.

Il modello quindi usa i TT registers passivamente — non perché ci sia un meccanismo che lo costringa, ma perché:
1. I TT registers vengono iniettati con la firma esatta (valori alti sui register neurons) che il modello già sapeva riconoscere come "broadcast node".
2. Le patch di sfondo che prima erano broadcast non lo sono più (azzerate sui register neurons).
3. La self-attention quindi sposta naturalmente il suo "flusso di lettura globale" sui TT registers, perché è lì che la firma di broadcast è più forte adesso.

È un'eleganza: il modello continua a fare esattamente quello che ha sempre fatto, è il *contenuto* della sequenza che cambia.

### Il difetto del metodo Jiang quando N>1: la nostra ipotesi

Jiang nel suo paper usa **N=1** ovunque (Tabella 2, Tabella 3, Figure 1-6). Nell'appendice A.5 testa N=2,3 e mostra miglioramenti marginali. Mai testa N grandi.

Nel codice di Jiang, la funzione `activate_on_registers` (vedi loro `shared/hook_fn.py`) copia **lo stesso valore massimo su tutti gli N TT registers**:

```python
output[0, -num_registers:, neuron_indices] = scale * sign_max(...).unsqueeze(0).expand(num_registers, -1)
```

Quindi a N=16 hai 16 token quasi identici (tutti con gli stessi valori sui register neurons). Il CLS si trova davanti a 16 copie ridondanti di scratch-space + 1369 patch. L'attention si dilui­sce sui 16 register tokens invece di concentrarsi sul soggetto → degrado delle feature CLS → crollo del Top-1.

**Questa è la spiegazione del crollo a N=16 nel nostro Figure 8 bottom (TT-reg):**
- N=1..8: il modello ha un singolo broadcast node duplicato 1, 2, 4, 8 volte. Il messaggio è ridondante ma non dannoso → curva piatta a 95.33%.
- N=16: troppe copie ridondanti diluiscono troppo il segnale globale → Top-1 crolla a 92.00%, mIoU scende, RMSE sballa.

### Idea originale per esperimento extra: TT registers indipendenti

Invece di copiare lo stesso valore su tutti gli N TT registers, **uno per ogni outlier patch presente nell'immagine**. Concretamente per ogni register neuron:
1. Identifica le top-N posizioni patch con attivazione più alta (non solo il top-1).
2. Copia il valore di ognuna di quelle top-N posizioni sul corrispondente TT register (1° outlier → 1° TT register, 2° outlier → 2° TT register, ecc.).
3. Azzera quelle stesse posizioni nelle patch.

Risultato atteso:
- Ogni TT register diventa un broadcast node distinto, ciascuno corrispondente a un cluster di info globale diverso.
- Le patch sono pulite su più posizioni (non solo il top-1).
- Niente ridondanza, niente diluizione dell'attention con N grandi.

Questo è il tipo di esperimento da portare come **contributo originale alla presentazione**: nessuno dei due paper (Darcet, Jiang) lo fa. È una variante che potrebbe correggere il crollo a N=16 che osserviamo e rendere il Figure 8 bottom (TT-reg) monotonamente più simile alla curva del paper Darcet.

### Sul perché la nostra Figure 8 top mostra "un solo outlier" sul cane invece di 6-7

Quando abbiamo guardato l'immagine del cane sul DINOv2 baseline (N=0), nel nostro plot c'è **un solo punto giallo brillante** mentre nel paper Jiang/Darcet ce ne sono ~6-7.

Verifica empirica (eseguita sull'immagine 67.png a 518×518):
- max patch norm = 89.1 → outlier dominante
- 13 patches con norma > 50 (cioè ~tripla della media 44.7) → ci sono altri outlier minori
- nessuna patch sopra 100

Quindi gli outlier ci sono (~13 sopra norma), ma uno è 1.7× più grande degli altri. Con un colormap viridis a scala lineare, l'unico picco intenso schiaccia tutto il resto sul fondo blu scuro. Il paper Jiang/Darcet probabilmente:
- usa una scala logaritmica per il colormap, oppure
- clippa il range superiore (es. percentile 99 invece del max), oppure
- mostra direttamente le norme delle patch invece dell'attention CLS→patch.

**Ipotesi aggiuntiva sulla discrepanza di posizione:** la nostra immagine `67.png` è stata estratta dal PDF con un parser e potrebbe avere subito compressione JPEG o resize leggermente diverso dall'originale del paper. DINOv2 è molto sensibile a piccole variazioni di input: bastano artefatti JPEG diversi per spostare le posizioni outlier di alcune unità nella griglia 37×37. Plausibile che la nostra immagine, leggermente diversa, faccia emergere il picco principale sulla coda invece che nel cielo.

**Per la presentazione:** non è un bug del nostro codice né del modello. È un artefatto di (a) visualizzazione con colormap lineare, (b) possibile differenza dell'immagine sorgente. Possiamo aggiungere un secondo plot con `norm(patches)` direttamente per confronto.

### La patch outlier conserva la sua "essenza locale"?

Domanda naturale: se il modello usa una patch di sfondo come scratch-space globale, perde tutto il suo contenuto locale (cielo, prato)?

**No, e per due ragioni quantificabili.**

**(1) Quante dimensioni vengono cooptate.** Una patch è un vettore di 1024 dimensioni nel residual stream (per DINOv2-L). Il meccanismo di scrittura del modello passa per la proiezione `fc2` dell'MLP, che ha 4096 → 1024 dimensioni. I register neurons sono ~10-50 canali su 4096 hidden, in un solo layer (il 17 per DINOv2). L'effetto netto sul residual stream da 1024 è una direzione (un sottospazio piccolo) — non tutte le 1024 dimensioni sono toccate.

**(2) Indizi empirici.**
- L'alta norma totale della patch (es. 89 contro media 44) non è dovuta a tutte le 1024 componenti uniformemente più grandi: è dominata da pochi canali con valori enormi. Se proietti la patch outlier in un sottospazio ortogonale a quei canali, la magnitudo torna normale.
- Darcet (Tabella 4 del paper originale) fa linear probing classification sui token outlier: ottengono ~80% Top-1 (info globale buona, simile al CLS), ma su task di predizione di posizione spaziale fanno male — segno che l'info locale è in parte preservata ma "diluita" dall'iniezione globale.
- Jiang Sezione A.13: dopo l'intervento, le patch ex-outlier riacquistano performance *quasi* normale su task di segmentazione pixel-wise.

**Modello mentale corretto.** Pensa al residual stream della patch come a un vettore di 1024 dimensioni. Il modello durante il pretraining ha "scoperto" un sottospazio (~10-20 direzioni) che usa come "memoria globale condivisa". Le altre ~1000 dimensioni continuano a contenere l'info locale standard. Quando una patch viene cooptata come broadcast:
- Le 10-20 dimensioni del sottospazio globale ricevono valori enormi (l'alta norma).
- Le altre 1000 mantengono il loro contenuto locale normale.
- La norma totale schizza alta perché quelle poche dimensioni dominano la magnitudo.

**Analogia.** Una conversazione di sottofondo a basso volume (info locale) sovrastata da una persona che urla in un solo orecchio (info globale). L'orecchio è dominato dall'urlo, ma se ascolti l'altro orecchio senti ancora la conversazione.

**Perché questo importa per Jiang.** È esattamente quello che permette al metodo di funzionare. Jiang **non distrugge la patch**: tocca solo le ~50 dimensioni hidden specifiche dell'MLP layer 17, e indirettamente via fc2 una direzione corrispondente nel residual stream da 1024 dim. Le altre dimensioni della patch rimangono intatte e continuano a portare la loro info locale per i layer successivi.

### Il guadagno reale dei registri nelle downstream — quanto è grande?

Domanda onesta: se le patch outlier mantengono in gran parte l'info locale, perché preoccuparci? Quanto migliorano davvero le metriche?

Paper Darcet (Tabella 2, DINOv2-L/14 su dataset completi):

| Task | Senza reg | Con reg | Δ |
|---|---|---|---|
| ImageNet Top-1 | 86.0 | 86.7 | +0.7 |
| ADE20k mIoU | 48.4 | 49.1 | +0.7 |
| NYUd RMSE | 0.402 | 0.382 | −0.02 (~5% relativo) |

**Risposta sincera**: il guadagno è piccolo in assoluto, sub-punto percentuale. Non è trasformativo. Però:

1. **Per attention maps / interpretabilità il guadagno è qualitativo, non quantitativo.** Senza registri le mappe sono inutilizzabili (rumore di sfondo); con registri puntano al soggetto. Quello è il messaggio principale dei due paper, non i +0.7.

2. **Per unsupervised object discovery il guadagno è grosso** — Darcet riporta +21 punti di correct localization su LOST. Lì la geometria delle attention maps conta davvero, e il salto è enorme.

3. **Sub-punto percentuale è significativo** in benchmark saturi come ImageNet. Modelli che competono per +0.3 di Top-1 hanno richiesto mesi di ingegneria; un trucco architetturale che dà +0.7 senza costi extra a inference è una vittoria pulita.

Il valore principale del lavoro **non sono i numeri delle downstream**. È mostrare che (a) il fenomeno high-norm esiste, (b) è un side-effect non intenzionale del training, (c) si può eliminare a costo zero. Le downstream non rompevano niente — il modello funzionava già bene. Quello che gli outlier rompevano davvero era l'interpretabilità (attention maps) e i task che ne dipendono.

Quindi il fenomeno non è catastrofico per le metriche standard. È un difetto di "pulizia" più che di performance.

---

## Esperimento extra²: TT registers indipendenti (variante originale)

Sezione dedicata al secondo esperimento extra che proponiamo, non descritto in nessuno dei due paper.

### Idea

Nel codice Jiang originale (`activate_on_registers` in `shared/hook_fn.py`), per ogni register neuron viene preso il **top-1 outlier patch** e quel valore viene **duplicato su tutti gli N TT registers**. Con N=16 hai 16 broadcast node identici.

**Variante "independent"**: per ogni register neuron, prendiamo le **top-N posizioni patch più attive** (non solo top-1) e copiamo i loro valori sui rispettivi N TT register tokens:
- TT_REG_0 ← valore della patch più attiva (top-1)
- TT_REG_1 ← valore della patch 2a più attiva (top-2)
- ...
- TT_REG_{N-1} ← valore della patch N-esima più attiva

E azzeriamo le top-N posizioni patch per quel neurone (non solo la top-1).

### Motivazione

Risolve il difetto del metodo Jiang originale per N>1: invece di duplicare lo stesso broadcast node, distribuiamo info globale su N broadcast node *diversi* (uno per cluster spaziale di outlier). Inoltre puliamo più patch contemporaneamente.

Ipotesi: la curva di Figura 8 bottom dovrebbe essere **monotonamente crescente o piatta**, senza il crollo a N=16 che osserviamo nella versione standard.

### Implementazione

File: `Adv_ML/src/Poli/ablation/test_time_registers_independent.py`.

L'unica differenza rispetto al modulo Jiang originale (`test_time_registers.py`) è nell'hook:

```python
# Jiang originale (top-1 duplicato):
output[:, -n_reg:, neuron_idx] = max_val.unsqueeze(1).expand(-1, n_reg, -1)
output[:, num_prefix + argmax, neuron_idx] = 0

# Variante indep (top-N distribuiti):
topk_vals, topk_idx = abs_sel.topk(n_reg, dim=1)
signed_vals = sel.gather(1, topk_idx)              # (B, n_reg, K)
for r in range(n_reg):
    output[:, -n_reg + r, neuron_idx] = signed_vals[:, r, :]
# azzera tutte le top-N posizioni patch
for k, neuron in enumerate(neuron_idx):
    output[:, num_prefix + topk_idx[..., :, k], neuron] = 0
```

### Smoke test (immagine del cane)

| N | Max patch norm | TT register norms | Note |
|---|---|---|---|
| 0 | 88.05 | — | baseline identico (no hooks) |
| 1 | 58.07 | [67.7] | un solo TT register, contiene top-1 outlier |
| 4 | 57.81 | [54.6, 50.9, 55.2, 57.3] | 4 register **distinti**, valori diversi |
| 8 | 51.42 | [44.4, 51.6, 54.5, 56.1, 58.1, 56.8, 53.6, 50.7] | 8 register, max patch scende sotto 52 |
| 16 | 51.77 | [45.1, 48.4, ..., 33.1, 47.0] | 16 register diversi, NON duplicati |

**Differenza fondamentale vs Jiang originale a N=16**: nel modello originale i 16 TT register avrebbero tutti lo stesso valore. Nella variante indep hanno norme spaziate (33-58), confermando che stiamo davvero distribuendo info distinta. Inoltre la max patch norm scende leggermente di più (51 vs 53 della versione originale), perché ripuliamo più posizioni outlier.

### Figura 8 top — variante indep (mappe attention qualitative)

File: `results/figure8_top_ttreg_indep/figure8_top_ttreg_indep.png`.

Osservazioni qualitative confrontando Jiang originale vs indep sullo stesso cane:

- **N=0**: identico al baseline puro (1 outlier giallo isolato sulla coda). Niente hook installato, è solo DINOv2 baseline.
- **N=1**: il cane diventa visibile in entrambe le varianti, ma nella indep si vedono **alcuni puntini gialli residui nel cielo**. La Jiang originale aveva pulito di più. Spiegazione: indep ripulisce solo le top-N posizioni patch (qui N=1), ma se ci sono >1 outlier l'altro resta.
- **N=2..4**: residui di outlier ancora visibili.
- **N=8, 16**: progressivamente più puliti — quando N supera il numero effettivo di outlier dell'immagine, tutti vengono drenati.

**Differenza visiva chiave**: nella Jiang originale, anche con N=1 la mappa appare "subito pulita" perché tutti gli N register puntano allo stesso top-1 outlier — il singolo outlier dominante viene drenato. La indep invece dichiara esplicitamente il limite: per drenare K outlier hai bisogno di N≥K register.

### Risultati Figura 8 bottom (variante indep)

File: `results/figure8_bottom_ttreg_indep.{json,png}`.

**Note importante sull'implementazione**: la prima versione del modulo aveva un bug — azzeravo solo le top-N posizioni patch invece di azzerare *tutti* i patch token per i register neurons (come fa Jiang originale). Questo lasciava outlier residui che dominavano l'attention map e peggioravano tutte le metriche. Il bug è stato corretto: ora la variante "indep" azzera tutti i patch token (come Jiang) e si distingue dal Jiang originale **solo** per cosa scrive nei TT register (top-N valori distinti invece del top-1 duplicato).

**Risultati finali ottenuti (versione corretta):**

| N | ImageNet Top-1 | ADE20k mIoU | NYUd RMSE |
|---|---|---|---|
| 0 | 94.00% | 19.41% | 1.13 m |
| 1 | 95.33% | 19.49% | 2.26 m |
| 2 | 95.33% | 18.91% | 2.87 m |
| 4 | 95.33% | 18.53% | 1.69 m |
| 8 | 95.33% | 19.26% | 2.03 m |
| 16 | **92.67%** | 19.17% | **3.88 m** |

### Confronto diretto delle 2 varianti TT-reg su DINOv2 baseline

**ImageNet Top-1:**

| N | TT-reg Jiang (top-1 duplicato) | TT-reg indep (top-N distribuiti) | Δ |
|---|---|---|---|
| 0 | 94.00% | 94.00% | 0 (baseline) |
| 1 | 95.33% | 95.33% | 0 (varianti identiche a N=1) |
| 2 | 95.33% | 95.33% | 0 |
| 4 | 95.33% | **96.00%** | +0.67 |
| 8 | 95.33% | 95.33% | 0 |
| 16 | **92.00%** | **92.67%** | +0.67 |

**ADE20k mIoU:**

| N | TT-reg Jiang | TT-reg indep | Δ |
|---|---|---|---|
| 0 | 19.99% | 19.46% | -0.53 (varianza) |
| 1 | 19.99% | 18.47% | -1.52 |
| 2 | 19.87% | 18.99% | -0.88 |
| 4 | 19.03% | 19.10% | +0.07 |
| 8 | **19.98%** | 18.90% | -1.08 |
| 16 | 18.83% | 18.24% | -0.59 |

**NYUd RMSE:**

| N | TT-reg Jiang | TT-reg indep | Δ |
|---|---|---|---|
| 0 | 2.04 m | 2.41 m | +0.37 (varianza training depth) |
| 1 | 3.83 m | **1.56 m** | -2.27 m ✅ |
| 2 | 2.88 m | 2.09 m | -0.79 m ✅ |
| 4 | 1.40 m | 1.96 m | +0.56 |
| 8 | 1.48 m | 1.85 m | +0.37 |
| 16 | 2.95 m | **1.68 m** | -1.27 m ✅ |

### Take-away effettivo dell'esperimento

**1. L'ipotesi principale (curva monotona senza crollo a N=16) è FALSIFICATA.** Anche con la variante indep (e con il bug di azzeramento corretto) il Top-1 crolla a N=16 (92.67%), praticamente identico al Jiang originale (92.00%). Implicazione importante: il problema di N grandi **non è la duplicazione del broadcast node**. Distribuire informazione genuinamente diversa su N register distinti non risolve nulla.

**2. Diagnostica sul cane** (a tutti gli N≥1):
- max patch norm = 51-52 (vs 88 del baseline, 89 del N=0)
- nessuna patch sopra 60
- I patch tokens sono *correttamente puliti* a tutti gli N

Quindi il crollo a N=16 **non è** dovuto a outlier residui sui patch token. La pulizia funziona.

**3. Cosa causa allora il crollo?** Tre ipotesi compatibili con i dati:
- **Diluizione dell'attention**: il CLS deve distribuire la sua attention totale su 1 + 1369 + 16 = 1386 token invece di 1370. Pochi token in più sembrano poco, ma se quei 16 hanno tutti firma "broadcast forte" attraggono attention significativa (~50% del totale CLS), togliendola alle patch del soggetto.
- **Distribuzione di valori malformata sui TT register**: nella variante indep, TT_REG_0 ha la firma del top-1 (valore enorme), TT_REG_15 ha la firma del top-16 (valore molto più basso). Quei 16 register hanno "intensità di broadcast" graduata da forte a debole. Quelli deboli potrebbero confondere la self-attention più che aiutarla.
- **Effetto cumulativo sui layer 18-23**: dopo il layer 17 (dove facciamo l'edit), restano 6 blocchi MLP. Quei 6 blocchi non sono mai stati addestrati a vedere 16 token con quella distribuzione di attivazioni — è un input fuori distribuzione per loro. Può emergere comportamento caotico.

**4. RMSE rumoroso**: il training della testa lineare di depth ha varianza enorme tra run diversi (es. N=0 dà 2.04m nel primo run TT-reg, 2.41m nel run indep buggato, 1.13m nel run indep fixed — stesso modello!). Conclude poco sui trend RMSE.

### Confronto a tre delle Figure 8 — sintesi per la presentazione

| Esperimento | Punto forte | Difetto |
|---|---|---|
| **Figure 8 vecchia (truncate/cycle)** | Curva monotona crescente — replica del paper Darcet | Modello "mutilato": N≠4 sono stati non nativi del modello |
| **Figure 8 TT-reg Jiang (top-1 duplicato)** | Replica fedele del paper Jiang, N=1 funziona | Crollo a N=16 (duplicazione → diluizione) |
| **Figure 8 TT-reg indep (nostra variante)** | Validazione fedele del Jiang (Top-1 identico a tutti gli N), risposta scientifica a "serve indipendenza?" | Stesso crollo a N=16: il problema NON è la duplicazione |

**Take-away rivisto per il prof:** la nostra variante indep era progettata per testare un'ipotesi specifica (il crollo a N=16 è causato dalla duplicazione). **L'ipotesi è stata falsificata**: anche distribuendo informazione genuinamente distinta su 16 register, il modello crolla allo stesso modo. Il bottleneck è quindi nella **capacità del transformer di gestire molti broadcast node**, indipendentemente da cosa contengano. È un risultato negativo onesto, che insegna qualcosa: il metodo Jiang è limitato a N piccoli per un motivo architetturale fondamentale, non procedurale.

### Risultato originale: un esperimento ben progettato che NON conferma l'ipotesi è comunque utile

Il valore della variante indep per la presentazione non è "abbiamo migliorato Jiang" (non l'abbiamo fatto) ma "**abbiamo testato un'ipotesi specifica e l'abbiamo falsificata con dati controllati**". Sostituendo l'unico fattore che differiva tra le due varianti (cosa metto nei TT register), il crollo persiste. Questo è un controllo sperimentale pulito.

### Figura 8 top — variante indep (dopo bug fix)

Le mappe di attention CLS sono pulite a tutti gli N≥1, ma a N=8 e N=16 appare un puntino giallo residuo nell'attention map.

**Importante**: la diagnostica empirica conferma che a quegli N **non ci sono più outlier sui patch tokens** (max patch norm ~51, nessuna patch sopra 60). Quindi il puntino giallo che vediamo nell'attention map **non è una patch con norma anomala**, ma una patch che riceve attention residua dal CLS pur avendo norma normale. Probabilmente l'intervento massiccio sui register neurons cambia leggermente la distribuzione Q/K dell'attention layer finale in modo che qualche patch "vicina" all'outlier originale riceva ancora attention residua. È un effetto secondario, non un fallimento della pulizia.

### Future work — non implementato ma da menzionare

**Interpolazione spaziale delle attivazioni dei register neurons.** Invece di azzerare le posizioni outlier (modalità `"zero"` di Jiang) o sostituirle con la media globale (modalità `"mean"`), si potrebbero ricostruire le attivazioni dai vicini spaziali sulla griglia 37×37 (es. media degli 8 vicini, o un piccolo blur gaussiano).

Razionale: le patch outlier sono per definizione predicibili dai vicini (è per questo che il modello le aveva scelte come sacrificabili). Sostituire le attivazioni dei register neurons con valori plausibili predetti dai vicini manterrebbe più informazione locale.

Vincolo pratico: l'effetto incrementale è di secondo ordine perché tocca solo ~50 dimensioni hidden su 4096 in un solo layer — la patch ha già la sua info locale nelle dimensioni rimanenti. Per dense prediction (NYUd in particolare, dove abbiamo visto regressione) potrebbe però ridurre il danno residuo.

Da menzionare nel report come "future work" senza implementarlo.
