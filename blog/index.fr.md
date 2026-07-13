---
title: "Le gamma du SPY sans inventer le signe des dealers"
description: "Une étude reproductible du gamma du SPY pondéré par l'open interest, de la variance réalisée le lendemain et de l'écart entre structure observable et positions des dealers."
date: 2026-07-13
image: images/gamma-surface-cover.png
categories: ["Quantitative Research", "Options"]
---

# Le gamma du SPY sans inventer le signe des dealers

Le récit habituel sur le gamma est séduisant. Un dealer long gamma couvre à
contre-courant du mouvement et peut l'amortir; un dealer short gamma couvre dans
le sens du mouvement et peut l'amplifier. Mais une chaîne d'options contenant
l'open interest et les Greeks n'indique pas qui détient chaque position. Sans
ce signe, les données ne permettent pas d'identifier le gamma des dealers.

Le code appelait initialement sa somme quotidienne `net_gamma_exposure`.
L'audit a montré que ce nom était faux : la série était égale à
`absolute_gamma_exposure` aux 21 dates, et le prétendu nœud de gamma négatif
n'existait jamais. Le moteur corrigé décrit maintenant ce que les entrées
permettent réellement de mesurer : une masse de gamma non signée, pondérée par
l'open interest.

Ce facteur renommé pose tout de même une question empirique valable. La masse de
gamma observée sur les options SPY à la date $t$ est-elle associée à la variance
intrajournalière réalisée lors de la prochaine séance observée, $t+1$? Pour
janvier 2024, la réponse est non. L'échantillon ne contient que 20 observations
alignées. C'est un diagnostic, pas une conclusion définitive sur la
microstructure des marchés.

## Ce que la chaîne permet de mesurer

Pour le contrat d'option $i$ à la date $t$, on définit :

- $OI_{i,t}$ : l'open interest, mesuré en contrats;
- $M=100$ : le nombre d'actions représentées par un contrat standard sur SPY;
- $S_t$ : le cours de clôture du SPY, en dollars par action;
- $\Gamma_{i,t}$ : le gamma d'une option longue, soit la variation du delta
  pour une hausse d'un dollar du SPY, mesuré en inverse de dollars;
- $m_{i,t}$ : la masse de gamma du contrat pondérée par l'open interest.

Le moteur calcule

$$
m_{i,t}=OI_{i,t} M S_t^2 \Gamma_{i,t}.
$$

Le suivi des unités est instructif. Les contrats et les actions s'annulent, et
$S_t^2\Gamma_{i,t}$ laisse un dollar. Ainsi, $m_{i,t}$ est une mesure de
courbure en dollars pour un mouvement proportionnel unitaire du spot. Pour la
convention plus courante d'un mouvement de un pour cent, on définit

$$
m_{i,t}^{1\%}=0.01m_{i,t}.
$$

Le total quotidien sur l'ensemble $\mathcal O_t$ des contrats retenus est

$$
G_t=\sum_{i\in\mathcal O_t}m_{i,t},
\qquad
G_t^{1\%}=0.01G_t.
$$

Il ne s'agit pas du flux de couverture attendu des dealers. Pour l'estimer, il
faudrait aussi connaître le détenteur et le signe de chaque position, le
mouvement de prix et la règle de rééquilibrage des dealers. L'open interest est
un stock non signé de contrats ouverts, pas un inventaire de dealer.

Un call ou un put vanilla détenu à l'achat a un gamma non négatif. Le nettoyeur
corrigé rend ce contrat de données explicite :

```python
open_interest_weighted_gamma = (
    open_interest
    * CONTRACT_MULTIPLIER
    * spot_close
    * spot_close
    * gamma
)
```

Un gamma vanilla négatif est désormais rejeté comme donnée fournisseur
invalide, et non interprété comme une position short. L'interface quotidienne
contient `total_open_interest_weighted_gamma`, `near_spot_gamma_mass_share`,
`front_expiry_gamma_mass_share`, `largest_gamma_mass_strike_distance`,
`call_put_gamma_mass_imbalance` et `gamma_mass_concentration_index`. Les anciens
champs net, absolu, nœud positif et nœud négatif ont disparu.

## La réponse est la variance réalisée le lendemain

Soit $P_{t,j}$ la clôture de la minute $j$ à la date de bourse $t$. Le rendement
logarithmique minute $r_{t,j}$ vaut

$$
r_{t,j}=\log(P_{t,j})-\log(P_{t,j-1}).
$$

Si la date $t$ contient $n_t$ rendements minute valides, sa variance réalisée
$RV_t$ est

$$
RV_t=\sum_{j=1}^{n_t}r_{t,j}^2.
$$

La variance réalisée est sans dimension, puisque chaque rendement logarithmique
l'est aussi. Sa racine carrée donne la volatilité réalisée sur la séance
échantillonnée, avant annualisation.

Chaque ligne de recherche associe $G_t$ à $RV_{t+1}$. Ici, $t+1$ désigne la
prochaine date présente dans le calendrier d'exposition, pas le lendemain
civil. La jointure est volontairement explicite :

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

Ce décalage empêche une réponse du jour d'entrer dans son propre prédicteur et
traite les fins de semaine sans fabriquer de dates.

## Ce que montre l'exécution corrigée

Le fichier brut d'options contient 168,762 lignes. Le nettoyage en conserve
131,215. Il retire 37,547 lignes dont l'open interest est nul ou négatif, soit
22.25% de l'entrée; 24,331 lignes conservées ont un gamma nul. Aucun gamma
négatif n'apparaît dans l'échantillon livré.

Sur 21 dates, $G_t$ varie de \$2.98 billions à \$4.74 billions selon l'échelle
brute d'un mouvement proportionnel unitaire. Les équivalents pour un pour cent,
$G_t^{1\%}$, vont de \$29.81 milliards à \$47.41 milliards. Ce sont des échelles
de masse gamma, pas des transactions prévues.

![Masse de gamma quotidienne du SPY et variance réalisée le lendemain](images/01-daily-alignment.png)

Les deux séries varient, mais leurs sommets et creux ne coïncident pas de façon
régulière. Le pic de variance le plus visible arrive alors que la masse de gamma
se trouve près du milieu de sa plage observée.

Pour tester une association monotone, j'utilise la corrélation de rang de
Spearman. Soit $R(G_t)$ le rang de la masse de gamma et $R(RV_{t+1})$ celui de la
variance réalisée le lendemain. En notant $\operatorname{Cov}$ la covariance et
$\sigma$ l'écart-type,

$$
\rho_s=
\frac{\operatorname{Cov}\left(R(G_t),R(RV_{t+1})\right)}
{\sigma_{R(G)}\sigma_{R(RV)}}.
$$

L'estimation donne $\rho_s=0.030$ avec une p-value bilatérale de $0.900$. Un
test de Kruskal-Wallis sur cinq quintiles de masse gamma donne $H=4.786$ et une
p-value de $0.310$. Aucun des deux tests ne rejette son hypothèse nulle aux
seuils usuels.

| Vérification | Résultat | Lecture |
|---|---:|---|
| Observations alignées | 20 | Un mois est trop court pour une estimation stable |
| Corrélation de rang de Spearman | 0.030 | Presque aucune association monotone |
| P-value de Spearman | 0.900 | La relation de rang est compatible avec le bruit |
| Statistique de Kruskal-Wallis | 4.786 | Les distributions par quintile sont peu séparées |
| P-value de Kruskal-Wallis | 0.310 | Aucun rejet entre les cinq quintiles |

![Variance réalisée le lendemain par quintile de masse gamma](images/02-quantile-variance.png)

Chaque quintile contient quatre observations. La variance moyenne du lendemain
passe de $3.80\times10^{-5}$ dans le premier quintile à $5.91\times10^{-5}$ dans
le troisième, puis retombe à $4.04\times10^{-5}$ dans le cinquième. Les
intervalles bootstrap percentiles à 95% se chevauchent largement. Il n'y a pas
de relation dose-réponse monotone.

![Masse de gamma et variance réalisée le lendemain](images/03-factor-scatter.png)

La droite ajustée n'est qu'un résumé visuel. Ce n'est pas une prévision hors
échantillon. Le nuage de points et la statistique de rang racontent la même
histoire : ce mois fournit peu d'indices en faveur d'une relation.

## Ce qui résiste à l'audit

Renommer le facteur ne change ni ses valeurs, ni leur ordre, ni les quintiles,
ni les statistiques de test. La correction change la portée économique du
résultat. C'est précisément son intérêt.

Plusieurs facteurs structurels restent valides parce qu'ils n'exigent aucun
signe de propriété. La part proche du spot mesure la fraction de la masse totale
dans une bande de moneyness donnée. L'indice de concentration est un indice de
Herfindahl-Hirschman calculé sur les parts par strike et échéance. Le déséquilibre
calls-puts décrit la composition entre types d'options; ce n'est pas un signal
long-short sur les dealers.

Le modèle de régime configuré exige 20 observations antérieures avant
d'attribuer une étiquette. La comparaison walk-forward demande elle aussi 20
lignes d'entraînement avant son premier score. Après l'alignement au lendemain,
il reste exactement 20 lignes. Les deux tables sont donc vides. Abaisser ces
garde-fous uniquement pour obtenir un chiffre affaiblirait le protocole.

Les corrélations de rang de la première et de la seconde moitié sont $-0.188$ et
$0.152$, avec des p-values de $0.603$ et $0.676$. Dix observations par moitié ne
suffisent pas à établir une instabilité, mais le changement de signe n'appuie
pas l'idée d'un effet durable.

## Ce qu'exigerait une étude signée du gamma dealer

Une étude signée a besoin de données qui distinguent les positions des clients
et des dealers, ou au minimum du sens des transactions accompagné d'un modèle
d'inventaire documenté. Une règle call positif, put négatif ne résout rien : un
call long et un put long sont tous deux longs gamma. Cette règle confond le type
d'option et le sens de la position.

Un meilleur protocole ajouterait plusieurs années de snapshots, enregistrerait
l'heure exacte de l'observation, traiterait explicitement les opérations sur
titres et les échéances, puis confronterait chaque règle de signe à des données
réelles de flux ou d'inventaire. Le facteur non signé devrait rester le benchmark.
Sa comparaison avec chaque estimation signée permettrait de séparer l'effet de
la concentration observable de celui de l'hypothèse d'inventaire.

Cette exécution de janvier ne valide pas le récit habituel sur le gamma des
dealers. Elle fait quelque chose de plus modeste et de plus défendable : mesurer
la courbure observable des options, respecter l'ordre temporel, publier un
résultat nul et refuser d'inventer un signe de position.

## Références

- Fischer Black et Myron Scholes, [The Pricing of Options and Corporate Liabilities](https://doi.org/10.1086/260062), *Journal of Political Economy*, 1973.
- Options Clearing Corporation, [Characteristics and Risks of Standardized Options](https://www.theocc.com/company-information/documents-and-archives/options-disclosure-document), mécanique des options et risques des contrats standardisés.
- Options Industry Council, [What Is an Option?](https://www.optionseducation.org/optionsoverview/what-is-an-option), structure d'un contrat et unité standard de 100 actions.
- Options Industry Council, [Gamma](https://www.optionseducation.org/advancedconcepts/gamma), définition et interprétation du gamma.
- Torben G. Andersen, Tim Bollerslev, Francis X. Diebold et Paul Labys, [Modeling and Forecasting Realized Volatility](https://doi.org/10.1111/1468-0262.00418), *Econometrica*, 2003.
- Andrea Barbon et Andrea Buraschi, [Gamma Fragility](https://doi.org/10.1093/rfs/hhaa048), *Review of Financial Studies*, 2021.
