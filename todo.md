## suite à discussion avec Yves (7 août)

## changement d'index

- $\Delta$Gain -> $\Delta$Energie
  - OK for figure_2B (mais bug energy cf sup_1)

## traces soustraites et non soustraites à la même échelle

étapes: 

- reconstruire la nouvelle sous-pop individuellement significative (vm n=15 sector, n=13 Full)
  ==> TODO path to replace: centrifigs.py ==> plot_figure3 else if age='new' line 497
  ==> TODO ajouter traces spikantes (sélectionner avec cellules oscillatoire ou non?)
  'controlFigs'
- ~~fig_3 sig non_sig soustraites on non, un panneau et même échelle en y~~
  - -> 3_expandSameY
- en énergie
- données : /Users/cdesbois/ownCloud/cgFigures/data/index
  - conditions_order.xlsx, time50energySpk.xlsx, time50energyVm.xlsx, time50gain50Spk.xlsx, time50gain50Vm.xlsx
    - -> spike données stat cellule oscilatoire changé
  - Pb sig <-> sup ou inf donc : 
    - ne prendre que sig > 0 & index > 0 (cpiso)
  - NB /Users/cdesbois/ownCloud/cgFigures/averageTraces/neuron_props.xlsx

## réalignement pic à pic

- ? intérêt sur les trois pannels

## fig_4 : élargir pour montrer la construction de la réponse

- ~~bumps successifs~~

- ~~effet vitesse~~
- ->  updated the code to allow an online update (+ saved examples on )

## fig_3 

- ~~insert de la réponse avec échelle élargie [-150:30] uniqt pour la random + centOnly~~
- sig à gauche
- update path (cf code)
- secteur | full

# passage de fig2 à fig 3	

- scatter plot all cells -> pour Yves time energy

- obj justifier les sous populations (stim)

  1. figure 2 avec uniquement exemple + pop (~~sig~~)
  2. supprime fig2B
  3. figure stat -> définition des index + choix sous populations

  - base = plot_stat
  - en haut sect | full (all pop, vm)
  - en bas sect | full (sub pop<u>s</u>) ? indication du nombre de cellules sig (barplot?)

  4. figure plot_sorted_responses_sup1 (vm_engy0.png)

  5. figure :

| sect / vm / normAlign  | sect/ vm / raw   | full / vm / normAlign  | full/ vm / raw  |
| ---------------------- | ---------------- | ---------------------- | --------------- |
| sect / spk / normAlign | sect / spk / raw | sect / spk / normAlign | full/ spk / raw |

 	1. même chose alignement des pics ?
 	2. 

 	1. 

## fig_sup waterfall cf indiSig cfIso

- voir les bumps successifs



## stat
- plot the number of cells (stat_sig.png) 
- voir ou sont les fichiers (et les fichiers obsolètes à supprimer )
  - 'time50_engyvm' et time50_engy_spk sur git
  - basculer les données sur owncloud

# bugs

≠ engy fig2B et sup1 -> ok = sup1





NB : 

# choose files:

pb : keys = 'new',  'sec', 'vm' 

files = ['sigVmSectRaw.xlsx',
			 'sigVmSectPeak.xlsx',
 			'sigVmSectNormAlign.xlsx',
 			'popVmSectNormAlign.xlsx',
 			'nSigVmSectNormAlign.xlsx']

only NormAlign are present in all the pop parts



# columns names:

all the columns names start with pop

f1 = /Users/cdesbois/ownCloud/cgFigures/data/averageTraces/controlsFig/popVmSectNormAlign.xlsx
file = popVmSectNormAlign.xlsx

df1_columns_list = ['popVmCtr', 
                    'popVmscpIsoStc', 
                    'popVmscfIsoStc', 
                    'popVmscrossStc', 
                    'popVmfrndIsoStc', 
                    'popVmsrndIsoStc']

f2 = '/Users/cdesbois/ownCloud/cgFigures/data/averageTraces/controlsFig/sigVmSectRaw.xlsx'
file = 'sigVmSectRaw.xlsx'

df2_columns_list = ['popVmCtr',
                    'popVmscpIsoStc',
                    'popVmscfIsoStc',
                    'popVmscrossStc',
                    'popVmfrndIsoStc',
                    'popVmsrndIsoStc']

f3 = '/Users/cdesbois/ownCloud/cgFigures/data/averageTraces/controlsFig/nSigVmSectNormAlign.xlsx'
file = 'nSigVmSectNormAlign.xlsx'

df3_columns_list = ['popVmCtr',
                    'popVmscpIsoStc',
                    'popVmscfIsoStc',
                    'popVmscrossStc',
                    'popVmfrndIsoStc',
                    'popVmsrndIsoStc']

