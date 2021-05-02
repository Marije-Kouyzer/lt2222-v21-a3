from model import *
from train import *
path = '/home/xsayas@GU.GU.SE/scratch/lt2222-v21-resources/svtest.lower.txt'

l1, l2 = a(path)
contexts, vowels = b(l1, l2)

model = train(contexts, vowels, l2, 200, 100)