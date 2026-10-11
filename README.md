# 📱 Mobile Price Classification

Prédire la **gamme de prix** d'un téléphone (de 0 = économique à 3 = très cher) à partir de ses caractéristiques techniques, avec du Machine Learning.

## 🎯 Objectif

Construire et comparer plusieurs modèles de **classification multiclasse** pour trouver celui qui prédit le mieux la gamme de prix, puis expliquer pourquoi il fonctionne le mieux.

## 📊 Dataset

[Mobile Price Classification](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification) (Kaggle)

- 2000 téléphones, 20 caractéristiques (RAM, batterie, résolution, mémoire interne, 4G, wifi, etc.)
- Cible : `price_range` (4 classes : 0, 1, 2, 3)
- Aucune valeur manquante, classes parfaitement équilibrées (500 par classe)

> Le dataset n'est pas inclus dans ce dépôt. Télécharge-le sur Kaggle et place `train.csv` et `test.csv` dans un dossier `data/`.

## 🔍 Démarche

1. **Analyse exploratoire (EDA)** : distribution des classes, lien entre chaque variable et le prix, corrélations
2. **Séparation** entraînement (80 %) / validation (20 %), avec stratification
3. **Modèles de base** : régression logistique (avec standardisation), arbre de décision, Random Forest
4. **Optimisation** des hyperparamètres avec `GridSearchCV` (validation croisée à 5 folds, sur le jeu d'entraînement uniquement)
5. **Comparaison** avant / après optimisation (accuracy et F1-score macro)
6. **Validation croisée** du modèle final pour vérifier la stabilité des résultats
7. **Bonus** : sauvegarde du modèle (`joblib`) et prédictions sur `test.csv`

## 🏆 Résultats (jeu de validation, 400 téléphones)

| Modèle | Accuracy avant | Accuracy après optimisation | Gain |
|---|---|---|---|
| **Régression logistique** | 96,5 % | **98,25 %** | +1,75 pt |
| Random Forest | 88 % | 91 % | +3 pts |
| Arbre de décision | 83 % | 83,25 % | +0,25 pt |

Meilleurs hyperparamètres trouvés :

- Régression logistique : `C = 100`
- Arbre de décision : `max_depth = None`, `min_samples_leaf = 2`, `ccp_alpha = 0.001`
- Random Forest : `n_estimators = 100`, `max_depth = 10`, `min_samples_leaf = 1`, `max_features = 0.8`

## 💡 Ce qu'on retient

- **Le modèle le plus simple gagne.** Dans ce dataset, le prix dépend presque linéairement des caractéristiques. La régression logistique bat donc le Random Forest et l'arbre de décision.
- **La RAM est la variable la plus importante, de très loin** (environ 48 % de l'importance dans le Random Forest), devant la batterie et la résolution de l'écran.
- **Les erreurs sont rares et logiques** : 7 erreurs sur 400, toujours entre des gammes de prix voisines.
- **L'optimisation aide surtout le Random Forest** (+3 points) et la régression logistique (+1,75 point).

## ⚠️ Limites

- Dataset petit (2000 exemples) et probablement synthétique : les performances seraient plus faibles sur des données réelles.
- Les scores de validation viennent d'un seul découpage ; la validation croisée dans le notebook permet de vérifier leur stabilité.
- Possibles améliorations : tester d'autres modèles (SVM, XGBoost).

## 🚀 Lancer le projet

```bash
# 1. Installer les bibliothèques
pip install -r requirements.txt

# 2. Placer train.csv et test.csv dans le dossier data/

# 3. Ouvrir le notebook
jupyter notebook Mobile_Price_Classification.ipynb
```

## 📁 Structure du projet

```
├── data/
│   ├── train.csv
│   └── test.csv
├── Mobile_Price_Classification.ipynb
├── requirements.txt
└── README.md
```

## 🛠️ Technologies

Python · pandas · scikit-learn · matplotlib · seaborn · joblib