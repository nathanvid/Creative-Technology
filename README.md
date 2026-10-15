# Creative Technology — Intelligence Artificielle

Ce dépôt rassemble le code, les slides et les ressources du module, séance après séance. Il est complété à la fin de chaque bloc.

Revenez-y pour retrouver ce qui a été fait et pour continuer à vous entraîner de votre côté.

## Objectif du module

Comprendre le deep learning en le reconstruisant à la main : de la dérivée d'une fonction à un réseau de neurones qui apprend, jusqu'au réseau convolutif qui reconnaît des images, puis au petit GPT qui écrit du texte.

## Contenu du dépôt

```text
.
├── seance-1/              # Séance 1 — des dérivées au réseau convolutif
│   ├── slides.pdf         # support de cours
│   ├── micrograd.ipynb    # rétropropagation reconstruite à la main
│   ├── mnist.ipynb        # premier réseau avec PyTorch
│   ├── cifar10.ipynb      # réseaux convolutifs (CNN)
│   ├── data/              # datasets (créé au premier lancement, non versionné)
│   └── runs/              # courbes TensorBoard (créé à l'exécution, non versionné)
├── seance-2/              # Séance 2 — du bigramme au GPT (NLP)
│   ├── 01_texte_vers_nombres.ipynb  # vocabulaire, encodage, lots
│   ├── 02_bigram.ipynb              # compter les paires de caractères
│   ├── 03_embeddings.ipynb          # vecteurs appris, MLP de Bengio
│   ├── 04_rnn.ipynb                 # réseau récurrent, LSTM
│   ├── 05_self_attention.ipynb      # query, key, value
│   ├── 06_transformer.ipynb         # assembler un GPT caractère
│   ├── 07_generation.ipynb          # température, top-k
│   ├── 08_bpe.ipynb                 # tokenisation BPE, tiktoken
│   ├── data/                        # corpus (créé au premier lancement, non versionné)
│   ├── runs/                        # courbes TensorBoard (non versionné)
│   └── checkpoints/                 # modèle entraîné par 06, relu par 07 (non versionné)
├── pyproject.toml         # dépendances du projet
├── uv.lock                # versions exactes, identiques pour tout le monde
└── .python-version        # version de Python attendue (3.13)
```

## Installation

### 1. Installer uv

Le projet utilise [uv](https://docs.astral.sh/uv/) pour gérer Python et les dépendances. Il n'y a rien d'autre à installer : uv télécharge lui-même Python 3.13 s'il manque.

```bash
brew install uv                                                       # macOS (Homebrew)
curl -LsSf https://astral.sh/uv/install.sh | sh                       # macOS / Linux
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows
```

Vérifiez l'installation :

```bash
uv --version
```

### 2. Créer l'environnement

Depuis la racine du dépôt :

```bash
uv sync
```

Cette commande crée le dossier `.venv/` et y installe, aux versions exactes de `uv.lock`, tout ce dont les notebooks ont besoin : PyTorch, torchvision, NumPy, Matplotlib, scikit-learn, TensorBoard, Graphviz, tiktoken et JupyterLab. Elle est à relancer si `pyproject.toml` ou `uv.lock` changent.

### 3. Graphviz (pour `micrograd.ipynb`)

Le premier notebook dessine le graphe de calcul avec Graphviz. Le paquet Python est installé par `uv sync`, mais il a aussi besoin du **binaire système** `dot` :

```bash
brew install graphviz              # macOS
sudo apt install graphviz          # Linux (Debian/Ubuntu)
winget install Graphviz.Graphviz   # Windows
```

## Utiliser l'environnement

Il y a deux façons équivalentes de travailler dans l'environnement `.venv`.

**Option A — sans l'activer, avec `uv run` (recommandé).** Chaque commande préfixée par `uv run` s'exécute dans l'environnement du projet, et uv le resynchronise au besoin.

```bash
uv run python --version
uv run jupyter lab
uv run tensorboard --logdir seance-1/runs
```

**Option B — en l'activant, comme un venv classique.** Le prompt affiche alors `(createch)` et `python`, `jupyter` et `tensorboard` pointent vers `.venv/`.

```bash
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows (PowerShell / cmd)

jupyter lab                      # les commandes s'écrivent sans `uv run`
deactivate                       # pour en sortir
```

Pour vérifier que tout fonctionne :

```bash
uv run python -c "import torch; print(torch.__version__)"
```

## Utiliser les notebooks

### Les notebooks

Chaque séance a son dossier (`seance-1/`, `seance-2/`…) avec ses slides et ses notebooks. Dans une séance, les notebooks se suivent, chacun réutilise les idées du précédent :

- **Séance 1** : `micrograd` → `mnist` → `cifar10`.
- **Séance 2** : dans l'ordre de leur numéro, de `01_texte_vers_nombres` à `08_bpe`. Le notebook `07_generation` reprend le modèle entraîné par `06_transformer` : lancez le 06 d'abord.

Le détail de chaque notebook est dans [Ce qu'on a fait](#ce-quon-a-fait). Exécutez les cellules dans l'ordre (`Shift + Entrée`), ou d'un coup via _Run → Run All Cells_.

### Données et dossier de travail

Les notebooks écrivent dans le **dossier courant du notebook** : `seance-1/data/`, `seance-2/data/`, etc. pour les données, et `runs/` pour les courbes. Ces dossiers sont ignorés par git : ils se recréent tout seuls, on peut les supprimer sans risque pour repartir de zéro.

- **Séance 1** : `mnist.ipynb` et `cifar10.ipynb` téléchargent leur dataset au premier lancement (quelques centaines de Mo, connexion nécessaire). `micrograd.ipynb` n'a besoin d'aucune donnée.
- **Séance 2** : deux petits corpus sont téléchargés au premier lancement, une liste de 32 000 prénoms (`names.txt`, ~230 Ko, notebooks 02 à 04) et les pièces de Shakespeare (`input.txt`, ~1 Mo, notebooks 01 et 06 à 08). Le notebook 05 n'utilise aucune donnée. `06_transformer.ipynb` enregistre aussi le modèle entraîné dans `seance-2/checkpoints/gpt_char.pt`, que `07_generation.ipynb` recharge.

### Suivre les entraînements avec TensorBoard

Les notebooks MNIST, CIFAR-10 et Transformer enregistrent leurs courbes. Pour les visualiser, lancez dans un second terminal, à la racine du dépôt :

```bash
uv run tensorboard --logdir seance-1/runs
uv run tensorboard --logdir seance-2/runs
```

Puis ouvrez <http://localhost:6006>. On peut aussi cibler une seule expérience, par exemple `--logdir seance-1/runs/cifar10`. Les notebooks indiquent, après chaque expérience, quel onglet regarder.

### Accélérateur (GPU)

PyTorch utilise le CPU par défaut ; `cifar10.ipynb` et `06_transformer.ipynb` détectent tout seuls un GPU (`mps` sur Mac Apple Silicon, `cuda` sur NVIDIA) et s'en servent s'il existe. Les modèles du module sont petits : tout tourne aussi sur CPU, simplement plus lentement pour les CNN et le Transformer (quelques minutes sur `mps` pour `06_transformer.ipynb`, nettement plus sur CPU).

## Ce qu'on a fait

### Séance 1 — Des dérivées au réseau convolutif

- **`seance-1/micrograd.ipynb`** — reconstruction, à la main, du moteur de différentiation automatique de [micrograd](https://github.com/karpathy/micrograd) : dérivée, classe `Value`, graphe de calcul, rétropropagation, puis `Neuron` / `Layer` / `MLP` entraînés sur un petit exemple.
- **`seance-1/mnist.ipynb`** — premier vrai réseau de neurones avec PyTorch, entraîné sur MNIST : tenseurs et autograd, `Dataset` / `DataLoader`, boucle d'entraînement, évaluation (matrice de confusion).
- **`seance-1/cifar10.ipynb`** — passage aux réseaux convolutifs (CNN) sur CIFAR-10, puis un duel CNN contre MLP à budget égal.
- **`seance-1/slides.pdf`** — slides : présentation du Deep Learning.

### Séance 2 — Du bigramme au GPT

Deux journées de traitement du langage (NLP) : on construit un modèle de langue caractère par caractère, du plus simple (compter) jusqu'au Transformer, l'architecture des GPT.

- **`seance-2/01_texte_vers_nombres.ipynb`** — transformer du texte en nombres : vocabulaire, `encode` / `decode`, découpage train / validation, contexte et lots d'entraînement.
- **`seance-2/02_bigram.ipynb`** — le modèle de langue le plus simple : compter les paires de caractères, générer des prénoms, mesurer avec la log-vraisemblance négative (NLL), lisser, puis voir la table exploser avec les n-grammes.
- **`seance-2/03_embeddings.ipynb`** — donner à chaque caractère un vecteur appris : le MLP de Bengio, les embeddings dessinés en 2D, la similarité cosinus et les plus proches voisins.
- **`seance-2/04_rnn.ipynb`** — le réseau récurrent : un contexte de longueur quelconque, le gradient qui s'évanouit dans le temps, et la réponse du LSTM.
- **`seance-2/05_self_attention.ipynb`** — la self-attention pas à pas : des moyennes du passé à query / key / value, le masque, le facteur 1/√d, l'encodage positionnel, les têtes multiples.
- **`seance-2/06_transformer.ipynb`** — assembler un GPT caractère sur Shakespeare : têtes d'attention, feed-forward, connexions résiduelles et LayerNorm, avec une boucle d'entraînement comptée en pas comme pour les vrais LLM.
- **`seance-2/07_generation.ipynb`** — tirer dans la distribution du modèle : température, top-k et leurs effets sur le texte généré.
- **`seance-2/08_bpe.ipynb`** — la tokenisation BPE : Unicode et UTF-8, l'algorithme des fusions, un tokenizer entraîné sur Shakespeare, puis le vrai tokenizer de GPT avec `tiktoken`.

## Ressources externes

### Micrograd et rétropropagation

- Andrej Karpathy — [_The spelled-out intro to neural networks and backpropagation: building micrograd_](https://www.youtube.com/watch?v=VMj-3S1tku0) (vidéo)
- Notebooks originaux de la vidéo : [`micrograd_lecture_first_half_roughly.ipynb`](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/micrograd/micrograd_lecture_first_half_roughly.ipynb) et [`micrograd_lecture_second_half_roughly.ipynb`](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/micrograd/micrograd_lecture_second_half_roughly.ipynb)
- Dépôt [karpathy/micrograd](https://github.com/karpathy/micrograd)
- 3Blue1Brown — la série _Neural networks_, pour les bases que le CNN réutilise (couches, loss, descente de gradient, rétropropagation) :
  1. [_But what is a neural network?_](https://www.youtube.com/watch?v=aircAruvnKk)
  2. [_Gradient descent, how neural networks learn_](https://www.3blue1brown.com/lessons/gradient-descent/)
  3. [_What is backpropagation really doing?_](https://www.3blue1brown.com/lessons/backpropagation/)
  4. [_Backpropagation calculus_](https://www.3blue1brown.com/lessons/backpropagation-calculus/)

### PyTorch

- [PyTorch — Training a Classifier](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) (le tutoriel suivi dans `cifar10.ipynb`)
- [Documentation officielle PyTorch](https://pytorch.org/docs/stable/index.html)

### Réseaux convolutifs (CNN)

Des vidéos pour _voir_ ce que fait une convolution et ce que calcule un réseau, en complément de `cifar10.ipynb`.

- [Deepia](https://www.youtube.com/@Deepia-ls2fo) — chaîne d'animations sur le deep learning :
  - [_Convolutional Neural Networks_](https://www.youtube.com/watch?v=8NbXFoVk1mo) : le contexte, les filtres, le fonctionnement d'une couche convolutive, les fonctions d'activation et le pooling.
  - [_CNN Receptive Field_](https://www.youtube.com/watch?v=ip2HYPC_T9Q) : ce que « voit » chaque neurone à mesure qu'on empile les couches.
- 3Blue1Brown — [_But what is a convolution?_](https://www.youtube.com/watch?v=KuXjwB4LzSA) : la convolution discrète, des moyennes glissantes au traitement d'image (flou, détection de contours), c'est-à-dire l'opération que les filtres d'un CNN appliquent aux pixels.

### Modèles de langue et Transformer (séance 2)

Les notebooks de la séance 2 suivent la série _Neural Networks: Zero to Hero_ d'Andrej Karpathy :

- [_Building makemore_](https://www.youtube.com/watch?v=PaCmpygFfXo) — le bigramme et la NLL (`02_bigram.ipynb`).
- [_Building makemore Part 2: MLP_](https://youtu.be/TCH_1BHY58I) — les embeddings et le MLP de Bengio (`03_embeddings.ipynb`).
- [_Let's build GPT: from scratch, in code, spelled out_](https://www.youtube.com/watch?v=kCc8FmEb1nY) — la self-attention et le Transformer (`05_self_attention.ipynb`, `06_transformer.ipynb`) ; code de référence : [karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture).
- [_Let's build the GPT Tokenizer_](https://www.youtube.com/watch?v=zduSFxRajkE) — le BPE (`08_bpe.ipynb`) ; code de référence : [karpathy/minbpe](https://github.com/karpathy/minbpe).
- [_The Unreasonable Effectiveness of Recurrent Neural Networks_](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) — l'article de Karpathy sur les RNN (`04_rnn.ipynb`).
- [Tiktokenizer](https://tiktokenizer.vercel.app) — pour voir comment les tokenizers de GPT découpent n'importe quel texte.

### uv

- [Documentation uv](https://docs.astral.sh/uv/) — en particulier [Working on projects](https://docs.astral.sh/uv/guides/projects/)
