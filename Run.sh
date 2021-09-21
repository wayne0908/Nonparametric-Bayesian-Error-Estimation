# Passive, uncertainty_sampling, EnhanceUncertainty, EnhanceUncertainty2
for t in {1..1}
do 	
	# python main.py --Trial=$t --FeatLen=2 --Sep=0.7 --S=20000 
	python main.py --Trial=$t --FeatLen=2 --Sep=0.7 --S=30000 
	python main.py --Trial=$t --FeatLen=2 --Sep=0.7 --S=40000 
	python main.py --Trial=$t --FeatLen=2 --Sep=0.7 --S=50000 
	python main.py --Trial=$t --FeatLen=2 --Sep=0.7 --S=70000 
done
