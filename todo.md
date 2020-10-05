## suite à discussion avec Yves (7 août)

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

- [ ] voir les bumps successifs



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




