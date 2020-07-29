files spksectNorm, spksectNormAligned, spksectRaw, vmsectNorm, vmsectNormAligned, vmsectRaw (.xlsx)
All subpopulations average traces corresponds to the cells individually significant to the cp-iso SECTOR condition for the lat 50 and energy t0-baseline indices

      Norm 	  corresponds to the average traces of the subpopulation normalized but not realigned before averaging 
      Raw  	  corresponds to the average traces of the subpopulation that are not normalized and not realigned before averaging       
      NormAligned corresponds to the average traces of the subpopulation that are normalized and realigned at the first point of the center only response tresspassing 3 stddev of the blank condition before averaging       

In all files, there are keys corresponding to arguments following that order
	
      condition: center, cpisosect, cfisosect, cpcrosssect, rndisosect, rndisofull
      number of cells:
		- Vm: 
			- n15 cells (entire sub-population significant to lat50 and energy t0- return of center only to baseline)	
			- n6  cells (vm average traces corresponding to the spiking cells including the oscillating and sustained one)
			- n5  cells (vm average traces corresponding to the spiking cells without the oscillating and sustained one)
		- Spk: 
			- (no 15 cells)
			- n6  cells (corresponding spike average traces including the oscillating and sustained one)
			- n5  cells (corresponding spike average traces without the oscillating and sustained one)


files spkfullNorm, spkfullNormAligned, spkfullRaw, vmfullNorm, vmfullNormAligned, vmfullRaw (.xlsx) has been added
They correspond to subpopulations average traces of the cells individually significant to the cp-iso FULL (in addition to SECTOR) condition for the lat 50 and energy t0-baseline indices

      Norm        : same as above
      Raw         : same as above
      NormAligned : same as above

keys corresponding to arguments following that order
        
      conditions: center cpisofull, cfisofull cpcrossfull rndisofull; There is one condition less than in the SECTOR file: the rndisoserct condition
      number of cells:
		- Vm: 
			- n13 cells (entire sub-population significant to lat50 and energy t0- return of center only to baseline)	
			- n7  cells (vm average traces corresponding to the spiking cells including the oscillating and sustained one)
			- n6  cells (vm average traces corresponding to the spiking cells without the oscillating and sustained one)
			- n5  cells (vm average traces corresponding to the spiking cells without the oscillating and sustained one and without a cell showig a clear cp-iso down regulation at the spiking level)
		- Spk: 
			- (no 13 cells)
			- n7  cells (corresponding spike average traces including the oscillating and sustained one)
			- n6  cells (corresponding spike average traces without the oscillating and sustained one)
			- n5  cells (corresponding spike average traces without the oscillating and sustained one and without the cp-iso down-regulated one)

SECTOR
cells subpopulation significant for the cp-iso sector combination of lat50 and energy indices:
1424M_CXG16, 1427A_CXG4, 1427K_CXG4, 1429D_CXG8, 1509E_CXG4, 1512F_CXG6, 1516D_CXG3, 1516F_CXG2, 1516G_CXG2, 1516M_CXG2, 1516P_CXG3, 1524C_CXG2, 1527B_CXG2, 1622H_CXG3, 1638D_CXG5

among those cells, the spiking ones are: 
1427A_CXG4, 1427K_CXG4, 1429D_CXG8, 1509E_CXG4, 1516D_CXG3, 1524C_CXG2

The oscillating one is : 1427K_CXG4

FULL
cells subpopulation significant for the cp-iso sector combination of lat50 and energy indices:
1427A_CXG4, 1427K_CXG4, 1440G_CXG11, 1509E_CXG4, 1516C_CXG3, 1516F_CXG2, 1516G_CXG2, 1516M_CXG2, 1527B_CXG2, 1622H_CXG3, 1638D_CXG5, 1641A_CXG5, 1649F_CXG6

among those cells, the spink ones are:
1427A_CXG4, 1427K_CXG4, 1440G_CXG11, 1509E_CXG4, 1516C_CXG3, 1641A_CXG5, 1649F_CXG6

The oscillating one is : 1427K_CXG4
The cpiso down regulated one is: 1641A_CXG5


The total length of all traces is of 12001 points, ranging from -600 to 600 ms and the temporal dx is of 0.1 ms