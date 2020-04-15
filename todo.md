Stimulations:

Marc : Figure 1:
remplacer la couleur des flêches de proche en proche actuellement en bleu par du gris
réinstroduire dans les schémas des configurations FULL-ISO et FULL-CROSS les flêches des couleurs correspondantes

Rajouter les patchs de Gabor manquant pour la stimulation FULL-CP-ISO
Placer un patch de Gabor de plus faible contraste dans le centre de tous les schémas de stimulations

QUESTION : color blind-reader
Nouvelle Remarque (NR) 6 avril: A voir une vois les figures modifiées obtenues

Résultats:

Figure 2:

- Pannel A
  ~~Barre bleue verticale en pointillé (vm et spikes)~~
- Pannel C
  ~~Barre bleue horizontale en pointillé et terminant par une flêche (pas un tiret)~~
  ~~Vérifier trace noir vm individuel (panel A, en haut à gauche) si l'axe des y négatif n'est pas coupé pour le Vm vers 120 ms~~
- Pannel D et E:
  ~~Remplacer les légendes par $DELTA$ phase (ms) et $DELTA$ amplitude~~

Figure 3: 

Police légende "Normalized Membrane potential + "Relative time (ms)" en plus grand"

>  ???? taille 

Figure 4:
Pannel police listant les vitesse lettrage "speed": contour noir des lettres rempli par la couleur sélectionnée de la vitesse correspondante
Pour les lettrages et traces à 30% et 50% augmenter la visibilité des couleurs en foncant la gamme de couleurs utilisées
Taille des Légendes (Normalized Membrane potential + Relative time (ms) en plus grand) homogénéiser (Figure 3) <==> décaler panel B vers la droite

> taille ???
>
> couleur : voir avec Marc



NR old Figure 5 is now Figure 6
Figure 5:

Figure 5
augmenter police des ticks (en x et y) et annotations 100°/s et 5°/s

> non = pb = taille de la figure : à définir

Pannel A, B et C
Centre du RF fond non en noir mais grisé bien plus faible - permet de voir les zones ON-OFF du RF stimulé 
CP-CROSS, dans la flêche jaune: Un gabor avec une petite flêche vecteur = mouvement, légende: Hauteur plus faible de la flêche jaune représentant l'intégration latérale plus faiblement étendu dans l'espace le long du width axis à basse vitesse
CP-ISO rouge: 5 patchs de Gabor non overlappant = intégration longue distance, légende longueur plus grande de la flêche rouge représentant l'intégration latérale plus étendue dans l'espace le long du main axis à haute vitesse

~~TO DO:
Bien délimiter les réponses CP-ISO (en rouge) des réponses CP-CROSS (or) en 
i)  Supprimant la transparence
ii) Ajoutant un contour noir à chacun des PSTH pour les panneaux du haut et du bas
iii) Enlever le texte High speed en haut et Low speed en bas, élargir la police~~

Graphiquement (dessin)
Pannel C: (voir si la proposition est intéressante)

 - Réduire la taille de la flêche rouge et en ajouter 2 plus fines sur les cotés pour représenter le secteur angulaire du champ d'association
 - Transformer les flêches jaunes en Triangle aboutissant par la flêche actuelle 
 - Ajouter aux extrémités de ces axes respectifs 
	i) pour le rouge un patch de Gabor horizontal de chaque coté (gauche et droite)
	ii) pour le jaune un patch de Gabor lui aussi horizontal mais en Haut et en bas 

Figure 6 (old figure 5):
Figure 6:
~~Trait bleu en pointillé~~
~~Violet en Vert foncé continu~~
~~Décaler les boxcar de stim à -2 mV~~
Taille des Légendes (Normalized Membrane potential + Relative time (ms) en plus grand) homogénéiser (Figure 3, figure 4)

> taille : cette approche n'est pas possible 

Figure 7:

~~Violet remplacé par vert foncé~~
~~Dynamic LP de ocre/marron à violet~~
~~Baisser le relative time (légende x) plus bas (? pareil que pour les autres figures ?) ~~

Taille des Légendes (Normalized Membrane potential + Relative time (ms) en plus grand) homogénéiser (Figure 3, figure 4, figure 6)Figure 8:

> taille = même problème

Figure 8:
Commentaire des mails
Figure à générer pour les deux exemples du poster, avec et sans matrice Surround-Only au milieuA obtenir (Cyril)

Figure 9:

- Pannel A:
  Lettre plus fine de MT, V1 LGN, RetinaRemplacer les traces PSTH des panels 
  i) C: par les réponses Vm de la sous-population CP-ISO significative à gauche (n=10)
  ii)D: par un histrogramme de distribution d'avance de latence de la toute la population CP-ISO en indiquant les cas significatifs en rempli et non-significatifs vides

- Panel B
  Gabor Horizontaux
  Ajouter le contour grisé du Gabor de droite dans le pannel de gauche et vice versa pour le pannel de droite => indication du mouvement  par Gabor futur/passéFigure 10:
  i) Changer de position la matrice de prédiction de Kaplan et al., 2012 pour la placer en dessous de la trace CP-ISO Filling-in
     afin de faciliter la convergence de la flêche de bas en haut 
  ii) Changer la couleur de la trace Filling-in CP-ISO et de du remplissage +/- SEM en violet (lilas)

- pannel D
  
  ~~ligne pointillée grisée à 0~~
  ~~bin entre -1.5 - 0; 0 - 1.5 etc~~
  ~~bins équivalents symmétriques /0  ==> ne pas biaiser la réprésentation vers la droite
  Ajouter un bin vide à gauche avant le premier bin contenant les avances de latence négatives~~
  ~~Contour en noir de tout l'histogramme, autour du contour en rouge~~ 

~~merger C et D~~

Matériel Supplémentaire:

Supplementary Figure 1: (Function plot_ranked_responses2Ben(dico) line  3001 (Function looped for kind in ['vm'] line 3093))
Sorted Phase and amplitude gain for all conditions with significant and non-significant cells

TO DO 
	i)  Add between the CP-CROSS SECTOR and the RND-ISO SECTOR phase advance and amplitude gain a row containing both measures for the RND-ISO FULL condition
	~~ii) Modify the Text Phase Gain and Amplitude gain (fraction of Center-Only response by $DELTA$ phase and $DELTA$ Amplitude~~ 

Supplementary Figure 2: (Function plot_figSup6('minus') line 2359) 
Average Filling-in responses for all conditions using the the CP-ISO significant subpopulation of cells

TO DO
	i) Add between the CP-CROSS SECTOR and the RND-ISO SECTOR average filling-in response a row the average RND-ISO FULL condition for the same subpopulation 
	ii) Increase ticks and Text size	

Supplementary Figure 3: (Function plot_figSup1('pop') line 2123)
Average response of the entire Vm population (n=37) for each condition 

TO DO
	i) Replace the continuous blue line of the SECTOR RND-ISO by a dotted blue line
	ii) Add the FULL RND-ISO average as a continuous blue line



Sous le coude:
le reste des figures non conservées pour la partie Matériel supplémentaire (pas de modifs pour l'instant)

