import numpy as np 
import pdb 
import random 
import os 
import random

from Utility import *
from Options import * 

def main():
	"""
	Set up parameters
	"""
	args = parser.parse_args()

	"""
	Set random seed 
	"""
	print('======================= Random number with seed %d ========================='%args.Trial)
	random.seed(args.Trial)
	np.random.seed(args.Trial)
	
	"""
	Generate Gaussian data
	"""
	print('======================= Generate Gaussian data ==============================')
	Data, BER, True_lb, True_ub = GetData(args)

	"""
	Method1: Ber bounds  
	"""
	print('======================= BER bounds estimation ==============================')
	BERLower, BERUpper, Dp = BEREstimate(Data)

	"""
	Method2: Point-wise BER
	"""
	print('======================= Point-wise BER estimation ==============================')
	Sigma = 0.2
	# pdb.set_trace()
	# PointBER, AvgPointBER = BinaryKernelBayesLowerbound(Data, Sigma = Sigma)

	"""
	To do suggestions:
	1. Plot relation between dimensions V.S. BER bounds (method1, you already did that last semester).
	2. Plot relation between dimensions V.S. AvgPointBER (method2).
	3. Plot relation between data size V.S. BER bounds (method1)
	4. Plot relation between data size V.S. AvgPointBER (method2)
	5. Plot relation between data seperation V.S. BER bounds (method1)
	6. Plot relation between data seperation V.S. AvgPointBER (method2)
	7. Compare method 1 and method 2 based on the plots 1 to 6, and maybe draw some conclusions. 
	"""

if __name__ == '__main__':
	main()