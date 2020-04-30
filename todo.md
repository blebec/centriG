remarques yves

- ~~"Normalized" avec un "z" dans la Figure 6~~
\- "Firing rate" trop collé à l'axe des y dans Figure 5

pb avec figure5 -> trop lourde

~~fig sup 1 et 2 :~~

​	~~alterner axes y droite et g -> empiler les plots~~

sup 4 de Yannick

fig 9CD voir homogénéité sur les modifs proposées pas Yves

fig 2b: plot_figure2B
~~Phase and Amplitude in uppercase (cf Amplitude = Gain = Response homogeneisation) (Done)~~

fig6: ~~Membrane (potential) with an M uppercase (Done)~~

fig 9C: plot_figure9CD()
~~Phase in Uppercase (Done)~~
Retrieves data for extra histogram

fig sup1 : plot_sorted_responses_sup1() 
~~Phase with Uppercase (Done)~~

fig sup2: plot_figSup2(pop)
taille des légendes
voir comment séparer les leg.get_lines(), get_texts for line, text in zip() entre SECTOR et dernière line FULL

fig sup3: fig = plot_figSup3('minus', overlap=True)
~~Réduire overlap zone grisée non à lims[1] mais 0.4 pour éviter overlap entre bandes grisées autour de y = 0 pour chaque subplot~~
Ajouter les stims

fig sup4:  (extract_values() + autolabel() + plot_cell_contribution()
on the left: 
~~$\Delta$ added + Phase and Amplitude with Uppercase (Done)~~
~~Cond legend changed in Uppercase (Done)~~
~~Percentage text lowered to 0.25 of bar height~~ (Done but not stisfying)

on the right: + spike full (plot_figSup5('pop', 'ful')) 

fig sup5
Figure extra





#############

~~Figure 2,A, panel du haut, au milieu (panel B) et panel du haut à droite, il reste un -0.2 inutile~~

~~figure 6 il reste un -1 inutile en haut et en bas~~

~~et enfin sup_1, il reste l'axe négatif à gauche (contre seulement positif à gauche) et n"gatif + positif à droite (contre seulement négatif à droite)~~