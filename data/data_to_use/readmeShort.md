- recordings : 12001 pts, dx = 0.1 ms, [-600: 600 ms]

- raw | norm | align:
  - raw : averaged cells individually significant (lat_50 and engy) for cpIsoSect        
  - norm : = normalized traces (before averaging)
  - normAligned =  normalized and realigned 
    at the first point of the center only response tresspassing 3 stddev of the blank condition before averaging       

## sector 

- files : 
  - spksectNorm, spksectNormAligned, spksectRaw
  - vmsectNorm, vmsectNormAligned, vmsectRaw

- stims: 
  - center, cpIsoSect, cfIsoSect, cpCrossSect, rndIsoSect, rndIsoFull
- cells:
    - Vm: 
        - n15 : entire sub-population	
        - n6 : only spiking cells
        - n5 : -= the oscillating and sustained one
   - Spk: 
     - n6 : entire sub-population
     - n5  : -= oscillating and sustained one

## full

- files 
  - spkfullNorm, spkfullNormAligned, spkfullRaw, 
  - vmfullNorm, vmfullNormAligned, vmfullRaw
- cells:
  - average traces of the cells individually significant to cp-iso FULL (in addition to SECTOR) condition for the lat50 and energy
- stims:
  - center,  cpIsoFull, cfIsoFull cpCrossFull, rndIsoFull
  - nb no 'rndIsoSect' condition
- cells:
   - Vm: 
       - n13 : entire sub-population	
       - n7 : only spiking cells
       - n6 : -= oscillating and sustained one
       - n5 : -= the cell showig a clear cp-iso down regulation at the spiking level
  - Spk: 
    - n7 : entire sup-population
    - n6 : -= the oscillating and sustained one
    - n5 : -= the cp-iso down-regulated one



| sector                                                       | full                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| vm (15): <br />1424M_CXG16, 1427A_CXG4, 1427K_CXG4, 1429D_CXG8, 1509E_CXG4, 1512F_CXG6, 1516D_CXG3, 1516F_CXG2, 1516G_CXG2, 1516M_CXG2, 1516P_CXG3, 1524C_CXG2, 1527B_CXG2, 1622H_CXG3, 1638D_CXG5 | vm (13): <br />1427A_CXG4, 1427K_CXG4, 1440G_CXG11, 1509E_CXG4, 1516C_CXG3, 1516F_CXG2, 1516G_CXG2, 1516M_CXG2, 1527B_CXG2, 1622H_CXG3, 1638D_CXG5, 1641A_CXG5, 1649F_CXG6 |
| spk (6): <br />1427A_CXG4, 1427K_CXG4, 1429D_CXG8, 1509E_CXG4, 1516D_CXG3, 1524C_CXG2 | spk (7): <br />1427A_CXG4, 1427K_CXG4, 1440G_CXG11, 1509E_CXG4, 1516C_CXG3, 1641A_CXG5, 1649F_CXG6 |
| oscillating (1): <br />1427K_CXG4                            | oscillating (1):<br />1427K_CXG4                             |
|                                                              | down regulated cpIso (1):<br />1641A_CXG5                    |

