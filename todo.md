# current

- [ ] point bleu -> grey
- fig2:

  - [ ] pointillés qui s'étendent sur les deux graphs
  - [ ] 'point bleu' en gris clair
  - [ ] ? fig2 après la fig3 ?
  - [ ] fig2 avec la troisième colonne 'sig' (et nouveau filtrage pop sig spike)
    - pb si on rajoute la troisième colonne, les traces de spike doivent référer à la pop spikant significative (et non pas aux spikes de la pop vm significative)
- plot_stat : 
  - [x] energy -> $\Delta$ energy
  - [x] time50 -> latency advance
  - [ ] séparer secteur et full
  - [ ] ? 
    - mettre sorted_cells  sous sorted_responses
    - pour sect | full (2 figs)
    - mettre à G sorted et à droite stat
    - mettre sous stat l'union des pop (cell_contribution)
- cell_contribution

  - [ ] séparer secteur et full
  - [ ] séparer chaque graph en time | union | engy
  - [ ] à mettre sous 'sorted responses'
- histogramme  'sorted_response':
  - [ ] ajouter en base : les histo ('cell_contribution')
  - à gauche time50
  - au centre union
  - à droite engy 
  - (pour sect & full)
 - pop_traces2X2:
   - [ ] boolean to remove controls
- figure4
  - [ ] renommer -> speed
  - [ ] nouvelles traces -> filtrage polarisation centre seul
- fig 5 (baudot ... )
    - [ ]  waiting for Yves
- fig6 
  - [ ] renommer fig6 -> indfill
- fig7
  - [ ] renommer fig7 -> popfill
  - [ ] changer les traces
    - réalignement changé 
    - ajouter les contrôles (vm et spikes)
  - [ ] proposition figure 2x2)
    - à gauche : panel de droite en dessous du pannel de G (cf fig 6)
    - [ ] deux versions:
      - prédicteur linéaire = surroundTheCenter - centerOnly (ref = surroundOnly)
      - prédicteur linéaire = centerOnly + surroundOnly (ref = surroundThenCenter)
    - à droite : vm toutes les traces (en haut) & spike toutes les traces (en bas)
 - figure technique
    - zones de calcul time50, énergie, type de calcul
    - base sur l'exemple individuel de la figure 2 ?
    - 

****

# suite à discussion avec Yves (7 août)

## changement d'index

- [x] $\Delta$Gain -> $\Delta$Energie : FIXED bug in 2B, updated owncloud folder
  
  ```python
  # bug in plot fig2B
  colors = [color_dic[x] for x in df[signs[i]]]
  # the reference for the color (sig cells) is before the sorting of the cells
  ```
  
  
  

## traces soustraites et non soustraites à la même échelle

étapes: 

- [x] reconstruire la nouvelle sous-pop individuellement significative (vm n=15 sector, n=13 Full)
  ==> TODO path to replace: centrifigs.py ==> plot_figure3 else if age='new' line 497
  ==> TODO ajouter traces spikantes (sélectionner avec cellules oscillatoire ou non?)
  'controlFigs'
- [x] - [x] fig_3 sig non_sig soustraites on non, un panneau et même échelle en y

  - -> 3_expandSameY
- [x] en énergie
- [x] données : /Users/cdesbois/ownCloud/cgFigures/data/index
  - conditions_order.xlsx, time50energySpk.xlsx, time50energyVm.xlsx, time50gain50Spk.xlsx, time50gain50Vm.xlsx
    - -> spike données stat cellule oscilatoire changé
  - Pb sig <-> sup ou inf donc : 
    - ne prendre que sig > 0 & index > 0 (cpiso)
  - NB /Users/cdesbois/ownCloud/cgFigures/averageTraces/neuron_props.xlsx

## réalignement pic à pic

- ? intérêt sur les trois pannels

## fig_4 : élargir pour montrer la construction de la réponse

- [x] ~~bumps successifs~~

- [x] ~~effet vitesse~~
- [x] ->  updated the code to allow an online update (+ saved examples on )

## fig_3 

- [x] insert de la réponse avec échelle élargie [-150:30] uniqt pour la random + centOnly
- [x] sig à gauche
- [x] update path (cf code)
- [x] secteur | full

# passage de fig2 à fig 3	

- - [x] ~~scatter plot all cells -> pour Yves time energy~~

  - DONE -> owns/pythonPreview/cross

- obj justifier les sous populations (stim)

  - [ ] figure 2 avec uniquement exemple + pop (~~sig~~)
  - [ ] supprime fig2B
3. figure stat -> définition des index + choix sous populations

  - [x] base = plot_stat
    - [x] OK updated the code -> rebuilded with energy (check cross.py) 
  - [x] en haut sect | full (all pop, vm)
- [x] en bas sect | full (sub pop<u>s</u>) ? indication du nombre de cellules sig (barplot?)

4. figure plot_sorted_responses_sup1 (vm_engy0.png)

   - [x] -> centrifigs:plot_sorted_responses_sup1

  5. figure :

     - first tests
     
     - plot the raws

| sect / vm / normAlign  | sect/ vm / raw   | full / vm / normAlign  | full/ vm / raw  |
| ---------------------- | ---------------- | ---------------------- | --------------- |
| sect / spk / normAlign | sect / spk / raw | sect / spk / normAlign | full/ spk / raw |

 	1. même chose alignement des pics ?
 	2. 

 	1. 

## fig_sup waterfall cf indiSig cfIso

- [x] voir les bumps successifs



## stat 
- [x] plot the number of cells (stat_sig.png) 
- [x] change the sig extraction
  - [x] cells sig for time or engy (stats.py L 45)
- [x] see centrifigs.py/plot_cell_contribution (1831)
- [x] voir ou sont les fichiers (et les fichiers obsolètes à supprimer )

  - [x] 'time50_engyvm' et time50_engy_spk sur git
  - [x] basculer les données sur owncloud 
- [x] fig cf plot_stat
  - [x] en haut vm sect | vm full
  - [x] en bas vm sect | vm full (uniqt sous pop sig
    - [x] NB recalculer sig = sigCenterOnly time + sigCenterOnly engy
    - [x] \+ marginal cf cell_contribution 

# bugs

- [x] ≠ engy fig2B et sup1 -> ok = sup1




