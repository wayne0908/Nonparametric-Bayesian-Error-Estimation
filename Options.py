import argparse 
import numpy as np 

parser = argparse.ArgumentParser(description='High-dimensional BER estimation')

parser.add_argument('--DataType', type = str, default = 'Syn', help = 'Data type')

parser.add_argument('--Sep', type = float, default = 0.5, help = 'Seperation between means')

parser.add_argument('--Del', type = float, default = 0, help = 'difference between variance')

parser.add_argument('--FeatLen', type = int, default = 5, help = 'feature length')

parser.add_argument('--Trial', type = int, default = 1, help = 'experiment trial number ')

parser.add_argument('--LoadData', type = int, default = 0, help = 'Load existing data or not')

parser.add_argument('--S', type = int, default = 500, help = 'Sample size')

parser.add_argument('--SaveData', type = int, default = 0, help = 'Save data or not')

