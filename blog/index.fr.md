---
title: "Ce que l'exposition gamma du SPY m'a appris sur la volatilité du lendemain"
description: "Une étude hors ligne du gamma des options sur SPY, de la variance réalisée le lendemain et des limites cachées dans un pipeline quantitatif propre."
date: 2026-07-13
image: images/gamma-surface-cover.png
categories: ["Quantitative Research", "Options"]
---

# Ce que l'exposition gamma du SPY m'a appris sur la volatilité du lendemain

Les commentaires de marché racontent souvent une histoire très assurée sur le gamma. Une forte concentration de gamma chez les dealers devrait amortir les mouvements de prix; un gamma court devrait les amplifier. Le mécanisme est plausible, mais un bon pipeline de recherche doit distinguer ce mécanisme de ce que les données permettent réellement d'identifier.

J'ai posé une question plus étroite à partir de données SPY versionnées avec le projet : une mesure du gamma pondérée par l'open interest au jour $t$ est-elle associée à la variance intrajournalière réalisée le jour de bourse suivant, $t+1$? L'échantillon de janvier 2024 comprend 21 dates d'exposition et 20 paires facteur-réponse correctement alignées. Pour cette courte fenêtre, la réponse est nette : aucune association convaincante.

Ce résultat nul mérite d'être conservé. Il révèle aussi un problème plus lourd dans la définition du facteur. L'implémentation agrège des gammas d'options positifs sans déduire le sens de l'inventaire des dealers. Son `net_gamma_exposure` n'est donc pas un gamma dealer signé.

## La mesure passe avant le récit

Pour le contrat d'option $i$ à la date $t$, on note $OI_{i,t}$ l'open interest en nombre de contrats, $M=100$ le multiplicateur standard des options américaines sur actions, $S_t$ la clôture du SPY en dollars et $\Gamma_{i,t}$ le gamma de l'option, soit la variation du delta pour un mouvement d'un dollar du SPY. Le moteur calcule l'exposition du contrat $g_{i,t}$ ainsi :

$$
g_{i,t} = OI_{i,t} M S_t^2 \Gamma_{i,t}.
$$

Il additionne ensuite les contrats appartenant à l'ensemble $\mathcal{O}_t$ des lignes valides observées à la date $t$ :

$$
G_t = \sum_{i \in \mathcal{O}_t} g_{i,t}.
$$

Le facteur $S_t^2$ transforme une mesure locale de courbure selon la convention d'exposition monétaire retenue par l'implémentation. On obtient un facteur descriptif du marché des options. Il ne dit pas qui détient chaque contrat, si un market maker en est acheteur ou vendeur, ni quelle part a déjà été couverte.

L'expression centrale du nettoyage reste volontairement simple :

```python
gamma_exposure = (
    open_interest
    * CONTRACT_MULTIPLIER
    * spot_close
    * spot_close
    * gamma
)
```

Cette simplicité facilite l'audit : aucun signe lié au type call/put ou à la position du dealer n'entre dans la multiplication.

## Construire une réponse au lendemain sans lookahead

Pour la minute $j$ du jour de bourse $t$, on note $P_{t,j}$ le cours de clôture de la minute et $r_{t,j}$ son rendement logarithmique :

$$
r_{t,j} = \log(P_{t,j}) - \log(P_{t,j-1}).
$$

Si le jour $t$ contient $n_t$ rendements minute valides, la variance réalisée quotidienne vaut

$$
RV_t = \sum_{j=1}^{n_t} r_{t,j}^2.
$$

Chaque ligne de recherche associe $G_t$ à $RV_{t+1}$. Ici, $t+1$ désigne la prochaine date de bourse observée, pas le lendemain civil. Le constructeur du jeu de données trie les dates d'exposition, décale ce calendrier d'une ligne et ne conserve une réponse que si sa date correspond à la date décalée. Le facteur du vendredi ne se retrouve donc pas associé au samedi, et le mouvement réalisé du vendredi ne fuit pas dans le prédicteur du vendredi.

```python
ordered_exposures = exposures.sort("trade_date")
exposure_calendar = ordered_exposures.with_columns(
    pl.col("trade_date").shift(-1).alias("_next_exposure_date")
)
aligned = exposure_calendar.join(
    response_payload,
    left_on="_next_exposure_date",
    right_on="_response_join_date",
    how="inner",
)
```

La frontière hors ligne est aussi utile que l'alignement des dates. Les deux fichiers Parquet bruts sont versionnés, l'exécution n'appelle jamais ClickHouse, et une seule commande régénère toutes les tables présentées plus bas. On réduit ainsi les écarts accidentels entre un notebook exploratoire et le résultat publié.

## Ce que montrent les 20 observations

Le facteur quotidien varie de 2,98 billions à 4,74 billions dans les unités d'exposition du moteur. La variance réalisée le lendemain suit une trajectoire bien différente.

![Exposition gamma quotidienne du SPY et variance réalisée le lendemain](images/01-daily-alignment.png)

Les deux séries bougent sensiblement d'un jour à l'autre, mais leurs sommets et leurs creux ne coïncident pas de façon régulière. Le graphique suggère cette absence; une statistique de rang permet de la tester précisément.

La corrélation de rang de Spearman mesure une association monotone. On définit $R(G_t)$ comme le rang de $G_t$ et $R(RV_{t+1})$ comme le rang de la variance réalisée le lendemain. En notant $\operatorname{Cov}$ la covariance et $\sigma$ l'écart-type, la statistique s'écrit

$$
\rho_s = \frac{\operatorname{Cov}\left(R(G_t), R(RV_{t+1})\right)}
{\sigma_{R(G)}\sigma_{R(RV)}}.
$$

Dans cet échantillon, $\rho_s=0.030$ et la p-value bilatérale vaut 0.900. Un test de Kruskal-Wallis sur cinq groupes de gamma donne $H=4.786$ et une p-value de 0.310. Aucun des deux tests ne rejette l'hypothèse nulle d'absence de relation systématique aux seuils usuels.

| Vérification | Résultat | Interprétation |
|---|---:|---|
| Observations alignées | 20 | Un mois est un échantillon de diagnostic, pas une base pour une estimation stable |
| Corrélation de rang de Spearman | 0.030 | Presque aucune association monotone |
| P-value de Spearman | 0.900 | La relation de rang observée est compatible avec du bruit |
| Statistique de Kruskal-Wallis | 4.786 | Les distributions par groupe diffèrent trop peu dans cet échantillon |
| P-value de Kruskal-Wallis | 0.310 | Aucun rejet entre les cinq groupes |

La vue par quintile raconte la même histoire. Chaque groupe ne contient que quatre observations. La variance moyenne du lendemain monte jusqu'aux groupes intermédiaires, puis recule dans le quintile supérieur. Les intervalles bootstrap par percentile à 95% se chevauchent largement.

![Variance réalisée le lendemain par quintile d'exposition gamma](images/02-quantile-variance.png)

Un effet gamma monotone devrait produire une séquence plus ordonnée. La moyenne du cinquième quintile, $4.04 \times 10^{-5}$, reste sous celle du troisième, $5.91 \times 10^{-5}$. Avec quatre jours par groupe, une seule observation peut déplacer fortement l'une ou l'autre valeur.

![Nuage de points entre exposition gamma et variance réalisée le lendemain](images/03-factor-scatter.png)

Le nuage de points rend le problème de taille d'échantillon très concret. La droite ajustée est descriptive, pas prédictive. La corrélation de rang affichée dans le graphique reste la statistique pertinente ici.

## L'audit a trouvé une limite plus sérieuse que la p-value

Le snapshot brut contient 168 762 lignes d'options. Le nettoyage en conserve 131 215 et exclut 37 547 lignes dont l'open interest est nul ou négatif, soit 22,25% des données. Parmi les lignes retenues, 24 331 ont un gamma nul. Ces diagnostics rendent le filtre visible au lieu de laisser l'échantillon diminuer silencieusement.

L'audit du facteur compte davantage. Pour chacun des 21 snapshots quotidiens :

- `net_gamma_exposure` est exactement égal à `absolute_gamma_exposure`;
- la distance au principal strike de gamma négatif n'est pas définie;
- toutes les contributions de gamma agrégées sont positives ou nulles.

La formule suffit à l'expliquer. Le gamma standard d'un call comme celui d'un put est positif pour une option détenue à l'achat. Sans hypothèse sur le sens des positions, l'agrégat mesure une masse de gamma pondérée par l'open interest. Le qualifier de gamma dealer signé ajouterait une information absente des données.

Cette distinction modifie l'interprétation économique. Un modèle d'inventaire dealer pourrait attribuer des signes à partir du flux client, du sens des transactions ou d'une heuristique explicite. Chaque choix apporte ses propres hypothèses et erreurs de mesure. Le facteur actuel évite ces hypothèses, mais il ne peut pas tester le récit habituel opposant dealers longs gamma et dealers courts gamma.

## Le petit échantillon bloque aussi les annexes les plus sophistiquées

Le classificateur de régime de volatilité exige 20 observations antérieures avant d'étiqueter une journée. La comparaison prédictive en walk-forward demande elle aussi 20 lignes d'entraînement avant de produire sa première prévision hors échantillon. Après l'alignement des expositions au jour $t$ avec les réponses au jour $t+1$, il reste exactement 20 lignes. Les deux tables de sortie sont donc vides par construction.

C'est le comportement souhaitable. Abaisser les seuils jusqu'à obtenir un score produirait un résultat presque dépourvu d'historique d'évaluation. Les tables vides documentent le manque de données plus honnêtement qu'un modèle ajusté sur le mois entier.

Les corrélations de rang changent aussi de signe entre les deux moitiés de l'échantillon : $-0.188$ sur les dix premières observations et $0.152$ sur les dix dernières. Leurs p-values sont 0.603 et 0.676. Cette coupure ne prouve pas une instabilité, puisque chaque moitié est minuscule, mais elle ne donne aucune raison de croire que l'estimation globale est durable.

## Ce que je changerais avant d'utiliser ce facteur

La prochaine exécution a besoin de plus de dates avant d'avoir besoin d'un modèle plus complexe. Plusieurs années de données permettraient de construire des régimes uniquement à partir du passé, des prévisions walk-forward et des tests leave-one-month-out qui contiennent de vraies observations.

Je renommerais aussi le facteur actuel `open_interest_weighted_gamma`, afin que le code décrive exactement ce que les données étayent. Une estimation distincte du gamma dealer pourrait ensuite appliquer et tester une convention de signe documentée. La comparaison entre versions signée et non signée montrerait si un éventuel effet vient de la concentration du gamma ou de l'hypothèse d'inventaire.

Enfin, l'open interest quotidien est déjà ancien pendant la séance et ne permet pas d'identifier les changements de position intrajournaliers. Pour expliquer la variance du lendemain, le contrat de données doit préciser l'horodatage des snapshots, le traitement des opérations sur titres et les effets d'échéance.

L'exécution de janvier ne valide pas le récit de marché. Elle valide la discipline de recherche : respecter l'ordre temporel, exposer la convention du facteur et accepter une sortie vide ou statistiquement banale lorsque les données ne permettent pas une conclusion plus forte.
