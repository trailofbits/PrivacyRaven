from privacyraven.models.four_layer import FourLayerClassifier
from privacyraven.models.victim import FourLayerClassifier

def TrainInversionModels(
	input_size = 784,
	output_size = 10,
	):
	
	# The following is a proof of concept of Figure 4 from the paper
	# "Neural Network Inversion in Adversarial Setting via Background Knowledge Alignment"
	# We first train a classifier on a dataset to output a prediction vector 

	forward_model = train_four_layer_mnist_victim(
		gpus=1, 
		input_size = 784, 
		output_size = 10
	)

	







