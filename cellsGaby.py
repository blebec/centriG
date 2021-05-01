#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 13:53:38 2021

@author: cdesbois
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


pd.set_option('display.max.columns', None)
#pd.set_option('display.max.rows', None)
pd.set_option('display.precision', 2)

paths = config.build_paths()
dirname = os.path.join(paths['owncFig'], 'data', 'baudot')
files = [st for st in os.listdir(dirname)
         if not os.path.isdir(os.path.join(dirname, st))]
for file in files:
    print(file)

file = 'dataGaby2005.xls'
filename = os.path.join(dirname, file)
#
def load_gaby():
    df = pd.read_excel(filename, header=None)
    # remove empty columns
    df = df.dropna(axis=0, how='all')
    df = df.dropna(axis=1, how='all')

    # forward row fill
    for i in range(2):
        df.iloc[i] = df.iloc[i].fillna(method='ffill')
        # nb fill should be done before str conversion because of np.nan
        df.iloc[0] = df.iloc[0].apply(lambda st: str(st).lower())
    dico = {'analysé': 'ana',
            'ant-post.': 'ant',
            'area centralis': 'ac',
            'autres': 'other',
            'barflash': 'barfl',
            'barmvt': 'barmv',
            'barre en mouvement': 'barmv',
            'barre flashee': 'barfl',
            'barres': 'bar',
            'caract. bar flashee': 'caractBarFlash',
            'caract. gaby': 'caract',
            'caract. r.c.': 'caract',
            'carateristiques de stimulation': 'caractStim',
            'cassette dat': 'dat',
            'cellule': 'cell',
            'chat': 'cat',
            'commentaire': 'comment',
            'comments': 'comment',
            'contr.': 'contr',
            'contraste': 'contr',
            'cycles': 'cycl',
            'debut': 'start',
            'debut visuel': 'visStart',
            'decharg.': 'dech',
            'dir.opt.': 'optDir',
            'direct.': 'dir',
            'directions': 'dir',
            'distancepatch': 'pathcDist',
            'dom. oc.': 'dom',
            'durée': 'dur',
            'dynamique': 'dyn',
            'electrod': 'elect',
            'electrode': 'elect',
            'electrop.': 'ephy',
            'electrophysiologie': 'ephy',
            'enrgt. dat': 'dat',
            'euil spike': 'thspike',
            'fichier': 'file',
            'file': 'file',
            'fin': 'end',
            'fin visuel': 'visEnd',
            'freqspat': 'spatFreq',
            'grating': 'grat',
            'i/v files': 'ivfile',
            'inact.': 'inact',
            'l.barre': 'lBar',
            'lateral.': 'lat',
            'localisations spatiales': 'spatialLoc',
            'masques': 'masks',
            'membrane': 'mb',
            'modulation': 'modul',
            'nbori': 'nbOri',
            'nbrep': 'nbRep',
            'nbseq': 'nbSeq',
            'nb sequence': 'nbSeq',
            'nom fichier': 'filename',
            'n°': 'n',
            'orient.': 'orient',
            'oritun': 'oriTun',
            'parametres de stimulation': 'stimParam',
            'potentiel de repos': 'Vm',
            'prof.': 'depth',
            'protocole': 'protoc',
            'pup.': 'pup',
            're': 'rem',
            'remarque': 'rem',
            'remarque generale': 'rem',
            'remarques': 'rem',
            'remarques2': 'rem2',
            'remq': 'rmq',
            'retenu': 'keep',
            'reverse correlation': 'revcor',
            'rin': 'R',
            'rmq electroph.': 'rmq',
            'répetitions': 'repet',
            'spike': 'spk',
            'type': 'kind',
            'valeur': 'value',
            'vitesse': 'speed',
            'vmco': 'vmCor.',
            'vmrest': 'vmRest',
            'w.barre': 'wBar'}

# todo = visuel -> visual, remove _°

    for i in range(4):
        df.iloc[i] = df.iloc[i].apply(lambda st: str(st).lower())
        df.iloc[i] = df.iloc[i].apply(lambda st: st.strip())
        df.iloc[i] = df.iloc[i].apply(lambda st: dico.get(st, st))
        df.iloc[i] = df.iloc[i].apply(lambda st: st.replace(' ', ''))


    # column names
    # nb df.head(3) -> header on three lines
    # df.iloc[df.index[0:7]] = df.iloc[df.index[0:7]].astype('str')
    df.iloc[df.index[0:5]] = df.iloc[df.index[0:5]].astype('str')
    cols = df.iloc[df.index[0:4]].apply(lambda st: st.str.cat(sep=', '))
    cols = [_.replace('nan,', '') for _ in cols]
    cols = [_.strip() for _ in cols]
    cols = [_.replace('nan', '') for _ in cols]
    cols = [_.replace(',', '') for _ in cols]
    cols = [_.replace('  ', ' ') for _ in cols]
    cols = [_.strip() for _ in cols]
    cols = [_.replace('NOM', '') for _ in cols]
    cols = ['nd' if len(_)<1 else _ for _ in cols]
    cols = [_.strip() for _ in cols]
    cols = [_.lower() for _ in cols]
    cols = ['_'.join(_.split(' ')) for _ in cols]
    cols = [_.replace('_nom', '') for _ in cols]
    cols = [_.replace('_(ms)', '(ms)') for _ in cols]
    cols = [_.replace('_°', '') for _ in cols]
    cols = [_.replace('visuel', 'visual') for _ in cols]
    cols = [_.replace('_mv', '(mv)') for _ in cols]
    cols = [_.replace('._mv', '(mv)') for _ in cols]
    cols = [_.replace('_(na)', '(na)') for _ in cols]
    cols = [_.replace('_(microns)', '(microns)') for _ in cols]
    cols = [_.replace('solut.', 'solut') for _ in cols]
    cols = [_.replace('_sel.', '_sel') for _ in cols]
    cols = [_.replace('.opt.', 'Opt') for _ in cols]
    cols = [_.replace('_(s)', '(s)') for _ in cols]
    cols = [_.replace('_(min)', '(min)') for _ in cols]
    cols = [_.replace('.(mv)', '(mv)') for _ in cols]
    cols = [_.replace('%explor.', 'explor') for _ in cols]
    cols = [_.replace('%exp', 'exp') for _ in cols]
    cols = [_.replace('_vmcor.', '_vmcor') for _ in cols]
    cols = [_.replace('_(vm)', '(vm)') for _ in cols]
    cols = [_.replace('_kg', '(kg)') for _ in cols]
    cols = [_.replace('_i.clamp', '_iclamp') for _ in cols]
    cols = [_.replace('nom', 'cell') for _ in cols]
    # to be chekced
    cols = [_.replace('(m1)', '') for _ in cols]
    
    
    
    df.columns = cols
    df = df.drop(index=range(5))

    # drop gaby added rows
    df = df.drop(index=[193, 194, 201, 208])
    
    # lowercase
    df.gene_file_old = df.gene_file_old.apply(
        lambda st: str(st).lower())

    # fill names
    df.cell = df.cell.fillna(method='ffill')

    return df

data_df = load_gaby()

# for col in data_df.columns:
#     if col.startswith('grat'):
#         print(col)
        
#%%
# NB positions = ['gaby_cr', 'visuel_ac_od', 'visuel_ac_og',
# 'revcor_cr', 'barfl_cr', 'barmv_cr', 'grat_cr(m1)'

def list_group_keys():
    """ extract the global 'group' keys of gaby file """
    keys = {_.split('_')[0] for _ in data_df.columns}
    coord_loc = []
    for key in keys:
        select = [st for st in data_df.columns if key in st]
        select = [st for st in select if '_cr' in st]
        select = [st for st in select if st.endswith(('x', 'y', 'l', 'w', 'theta'))]
        if len(select) > 0:
            coord_loc.append(key)
    print('keys for location are {}'.format(coord_loc))
    return coord_loc

def select_corrd(key='revcor', printcells=False):
    select = [st for st in data_df.columns if key in st]
    select = [st for st in select if '_cr' in st]
    select = [st for st in select if st.endswith(('x', 'y', 'l', 'w', 'theta'))]
    select.insert(0, 'cell')
    df = data_df[select].copy()
    df = df.loc[df.cell.notnull()]
    cells = df.dropna(how='any')
    print('for key={} we have {} cells ({} records)'.format(
        key, len(cells.cell.unique()), len(df)))
    if printcells:
        print(cells)
    return df, cells

keys = list_group_keys()
key = 'revcor'
df, cells = select_corrd(key)

# for key in keys:
#     df, _ = select_corrd(key)

#%% look at all cordinates locations



# fig, ax = plt.subplots(figsize=(8,8))
# ax.scatter(cells.revcor_cr_x, cells.cell)
plt.close('all')

def plot_xy_distri(key='revcor', bins=10):

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    axes = axes.flatten()
    txt = key + ' ( {}cells {}records )'.format(
        len(cells.cell.unique()), len(cells))
    fig.suptitle(txt)

    item = key + '_cr_'
    # xy
    ax = axes[0]
    ax.set_title('(x y)')
    for neur in cells.cell.unique():
        x = cells.loc[cells.cell == neur, [item + 'x']].values
        y = cells.loc[cells.cell == neur, [item +'y']].values
        ax.scatter(x,y, alpha=0.5)
        ax.plot(x,y, alpha=0.5)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')

    ax = axes[1]
    ax.set_title('length')
    for neur in cells.cell.unique():
        x = cells.loc[cells.cell == neur, [item + 'l']].values
        ax.plot(x, marker='o', alpha=0.7)
    # ax.set_xlabel('num')
    # ax.set_ylabel('length')

    ax = axes[2]
    ax.set_title('length')
    meds = []
    for neur in cells.cell.unique():
        med = cells.loc[cells.cell == neur, [item + 'l']].median()
        meds.append(med)
    y, x = np.histogram(meds, bins=bins)
    width = round((np.max(meds) - np.min(meds))/bins * .9)
    ax.bar(x=x[:-1], width=width, height=y, align='edge', alpha=0.6)
    # ax.set_xlabel('length')
    ax.yaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.spines['left'].set_visible(False)

    ax = axes[3]
    ax.set_title('theta')
    for neur in cells.cell.unique():
        x = cells.loc[cells.cell == neur, [item + 'theta']].values
        ax.plot(x, marker='o', alpha=0.7)
    # ax.set_xlabel('num')
    # ax.set_ylabel('theta')

    ax = axes[4]
    ax.set_title('width')
    for neur in cells.cell.unique():
        x = cells.loc[cells.cell == neur, [item + 'w']].values
        ax.plot(x, marker='o', alpha=0.7)
    # ax.set_xlabel('num')
    # ax.set_ylabel('width')

    ax = axes[5]
    ax.set_title('width')
    meds = []
    for neur in cells.cell.unique():
        med = cells.loc[cells.cell == neur, [item + 'w']].median()
        meds.append(med)
    y, x = np.histogram(meds, bins=bins)
    width = round((np.max(meds) - np.min(meds))/bins * .9)
    ax.bar(x=x[:-1], width=width, height=y, align='edge', alpha=0.6)
    # ax.set_xlabel('width')
    ax.yaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.spines['left'].set_visible(False)


    for i, ax in enumerate(fig.get_axes()):
        # ax.set_title(i)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    if anot:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, 'cellsGaby:plot_xy_distri_' + key,
                 ha='right', va='bottom', alpha=0.4)
        fig.text(0.01, 0.01, date, ha='left', va='bottom', alpha=0.4)

    fig.tight_layout()
    
    return fig
    
plt.close('all')
anot=True
key = 'revcor'
_, cells = select_corrd(key)
fig = plot_xy_distri()    

save = False
for key in keys:
    _, cells = select_corrd(key)
    fig = plot_xy_distri(key)
    if save:
        dirname = '/Users/cdesbois/ownCloud/cgFigures/pythonPreview/gaby'
        file = 'gaby_xy_distri_' + key + '.pdf'
        fig.savefig(os.path.join(dirname, file))

#%%

def load_baudot_meta():
    """ load baudot meta """
    file = 'baudot_meta.xlsx'
    filename = os.path.join(dirname, file)
    df = pd.read_excel(filename)
    dico = {'NOM' : 'cell', 
            'contraste centre bas' : 'ct_center', 
            'Valeur contraste' : 'ct_patch',
            'Valeur distance en °' : 'dist' , 
            'valeur dt patch' : 'dt_patch', 
            'Valeur dtON' : 'dt_on',
            'Vitesse apparente °/s': 'speed', 
            'Longeur CR en °' : 'length', 
            'Largeur CR en °' : 'width',
            'RATIO L/W' : 'lw', 
            'MOYENNE Vm' : 'vm', 
            'MOYENNE Frequence de décharge Hz' : 'spk'
            }
    df = df.rename(columns=dico)
    df.loc[df.cell == '3900hg3 (sans spike)', ['cell']] = '3900hg3'
    df.cell.apply(lambda st: st.lower())
    return df


def check_in_gaby(df, datadf):
    """
    check if the baudot cells (df) are in the gaby file (datadf)

    """
    gaby_cells = datadf.cell.dropna().unique()
    baudot_cells = df.cell
    baudot_cells = [_[:-2] for _ in baudot_cells]
    baudot_cells = [_.lower() for _ in baudot_cells]
    gaby_cells = [_.lower() for _ in gaby_cells]

    gaby_cells = set(gaby_cells)
    baudot_cells= set(baudot_cells)

    if baudot_cells < gaby_cells:
        print('='*20)
        print('all the bausdot cells are present in Gaby data')
    else:
        print('in baudot but absent in gaby {}'.format(baudot_cells - gaby_cells))


df = load_baudot_meta()
check_in_gaby(df, data_df)


#%% locate file -> file_old

# bug with 0700kG1

def get_location(df, data_df):
    """ get the baudot cell location in the gaby file """
    founded = {}
    not_founded = []
    for cell in df.cell:
        absent = True
        cell = cell.lower()
        for col in data_df.columns:
            if data_df[col].eq(cell).any().any():
                id = data_df[col].loc[data_df[col].eq(cell)].index[0]
                loca = (id, col) 
                # print('{} : {}'.format(cell, loca))
                founded[cell] = loca
                absent = False
        if absent:
            not_founded.append(cell)
        # print('{} not founded'.format(cell))

    return founded, not_founded 

founded, not_founded = get_location(df, data_df)

print('='*20)
print('founded {}'.format(founded))        
print('='*20)
print('not founded : {}'.format(not_founded))


#%% look at data from baudot to gaby

def test_cr(datadf=data_df, founded=founded):
    cols = [st for st in datadf.columns if '_cr' in st]
    cols = [st for st in cols if st.endswith(('x', 'y', 'l', 'w', 'theta'))]
    keys = list(set([_.split('_')[0] for _ in cols]))

    resdf = pd.DataFrame()
    for k, v in founded.items():
       resdf[k] = data_df.loc[v[0], cols]

    resdf = resdf.dropna(how='all')
    return resdf

resdf = test_cr()
dirname = os.path.join(paths['owncFig'], 'data', 'baudot')
file = 'baudotMinusGaby.xls'
resdf.to_excel(os.path.join(dirname, file))
