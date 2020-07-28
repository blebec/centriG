files spkNorm, spkNormAligned, spkRaw, vmNorm, vmNormAligned, vmRaw (.xlsx)
Al subpopulations corresponds to the cells individually significant to the cp-iso sector condition for the lat 50 and energy t0-baseline indices

      Norm 	  corresponds to the average traces of the subpopulation normalized but not realigned before averaging 
      Raw  	  corresponds to the average traces of the subpopulation that are normalized and not realigned before averaging       
      NormAligned corresponds to the average traces of the subpopulation that are normalized and realigned at the first point of the center only response tresspassing 3 stddev of the blank condition before averaging       

In all files, there are keys corresponding to arguments following that order
	
      condition: center, cpisosect, cfisosect, cpcrosssect, rndisosect, rndisofull
      number of cells:
		- Vm: 
			- n15 cells (entire sub-population significant to lat50 and energy t0- return opf center only to baseline)	
			- n6  cells (vm average traces corresponding to the spiking cells including the oscillating and sustained one)
			- n5  cells (vm average traces corresponding to the spiking cells without the oscillating and sustained one)
		- Spk: 
			- (no 15 cells)
			- n6  cells (corresponding spike average traces including the oscillating and sustained one)
			- n5  cells (corresponding spike average traces without the oscillating and sustained one)

The total length of all traces is of 12001 points, ranging from -600 to 600 ms and the temporal dx is of 0.1 ms