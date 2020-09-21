#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:49:26 2020

@author: cdesbois
"""

#fig2B version

latAmp_v_df = ldat.load_cell_contributions('vm', amp='engy', age='new')
cols = latAmp_v_df.columns
df2 = latAmp_v_df[[item for item in cols if 'cpisosect' in item]].copy()
df2.sort_values(by=df.columns[0], ascending=False, inplace=True)


#sup1 version
df1 = ldat.load_cell_contributions(kind=kind, amp=amp, age=age)
# extract list of traces : sector vs full
traces = [item for item in df.columns if 'sect' in item]
# append full random
rdfull = [item for item in df.columns if 'rdisofull' in item]
traces.extend(rdfull)
# filter -> only significative cells
traces = [item for item in traces if not item.endswith('sig')]

x = range(1, len(df)+1)
# use cpisotime for ref
name = traces[0]
name = traces[key]
sig_name = name + '_sig'
df = df.sort_values(by=[name, sig_name], ascending=False)

# select engy cpiso
col2_engy = [item for item in df2.columns if 'engy' in item]

# to be continued (l 1253)