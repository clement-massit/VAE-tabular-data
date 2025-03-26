# Autoencodeur Variationnel pour la Génération de Données Tabulaires

## Introduction aux Autoencodeurs Variationnels (VAE)

Les Autoencodeurs Variationnels (VAE) sont une extension des autoencodeurs classiques, introduits pour apprendre une distribution latente des données. Contrairement aux autoencodeurs classiques, les VAE ne se contentent pas d'apprendre une représentation compressée des données, mais contraignent cette représentation à suivre une distribution probabiliste, souvent une distribution normale multivariée.

### Principe des VAE

Les VAE sont composés de deux parties :

1. **L'encodeur** : Il prend une entrée \( x \) et l'encode en une distribution latente paramétrée par une moyenne `μ` et une variance `σ²`, formant un espace latent \( z \) :

    ```math
   q_\phi(z|x) = \mathcal{N}(\mu(x), \sigma^2(x))
   ```


2. **Le décodeur** : Il prend un échantillon \( z \) de cette distribution latente et tente de reconstruire les données d'origine :
   ```math
   p_\theta(x|z)
   ```
La perte du VAE est composée de deux termes :
- **L'erreur de reconstruction** : Mesure la qualité de la reconstruction des données d'origine à partir de l'espace latent.
- **La divergence de Kullback-Leibler (KL)** : Encourage l'espace latent à suivre une distribution normale multivariée `𝒩(0, I)`, facilitant la génération de nouvelles données.

L'objectif global du modèle est de minimiser :  
```math
\mathcal{L} = \mathbb{E}_{q_\phi(z|x)} [ \log p_\theta(x|z) ] - D_{KL} (q_\phi(z|x) || p(z))
```

## Architecture du VAE pour Données Tabulaires

### 1. Encodeur
L'encodeur prend en entrée un vecteur de caractéristiques tabulaires et l'encode dans un espace latent de dimension réduite.
- Utilisation de couches fully connected (Dense)
- Activation de type ReLU
- Deux sorties : une pour la moyenne (`μ`) et une pour la variance (`σ²`)

### 2. Échantillonnage
L'échantillonnage de `z` est effectué via la **reparamétrisation** :
```math
z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
```
### 3. Décodeur
Le décodeur reconstruit les données d'origine à partir de `z` via plusieurs couches fully connected avec une activation appropriée selon le type de variable (sigmoïde pour les variables binaires, softmax pour les catégories, linéaire pour les réels).

### 4. Fonction de perte
La fonction de perte combine :
- **Erreur de reconstruction** : Mesurée avec l'**erreur quadratique moyenne (MSE)** pour les valeurs continues et l'**entropie croisée** pour les variables catégoriques.
- **Divergence KL** : Encourage `q_φ(z|x)` à suivre `p(z)`.

## Génération de Données Synthétiques
Une fois entraîné, on génère des données en :
1. Échantillonnant un `z` depuis une distribution normale `𝒩(0, I)`
2. Passant `z` dans le décodeur pour obtenir de nouvelles données synthétiques

## Conclusion
Les VAE sont puissants pour générer des données tabulaires réalistes tout en préservant leurs distributions statistiques. Ils sont utilisés dans diverses applications comme l'anonymisation de données, l'augmentation de jeux de données et la modélisation générative dans le domaine médical et financier.
