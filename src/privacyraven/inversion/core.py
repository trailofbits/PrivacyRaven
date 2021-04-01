from privacyraven.models.four_layer import FourLayerClassifier
from privacyraven.models.victim import FourLayerClassifier
from privacyraven.utils.data import get_emnist_data

# Create a query function for a target PyTorch Lightning model
def query_mnist(input_data):
    # PrivacyRaven provides built-in query functions
    return get_target(model, input_data, (1, 28, 28, 1))

# Truncates a prediction vector such that the m highest values are preserved, and all others are set to 0.
# (Section 4.2 of )
def trunc(k, v):

	# kth smallest element
	b = sorted(v)[-k - 1]
	nonzero = 0

	for (i, vi) in enumerate(v):
		if vi < b or (vi != 0 and nonzero > k): v[i] = 0	
		nonzero += 1

	return v

def joint_train_inversion_model(
	input_size = 784,
	output_size = 10,
	dataset_train = None,
	dataset_test = None,
	data_dimensions = (1, 28, 28, 1),
	t = 2,
	):
	
	# The following is a proof of concept of Figure 4 from the paper
	# "Neural Network Inversion in Adversarial Setting via Background Knowledge Alignment"
	# We first train a classifier on a dataset to output a prediction vector 

	forward_model = train_four_layer_mnist_victim(
		gpus=1, 
		input_size = input_size, 
		output_size = output_size
	)

	# This is nowhere near complete but 
	# The idea here is that we query the model each time 
	for k in range(len(dataset_train)):

		# Fwx is the training vector outputted by our model Fw
		Fwx = query_mnist(forward_model, input_data, data_dimensions)

		# Let Fw_t denote the truncated vector
		Fwx_t = trunc(t, Fwx)
	


if __name__ == "__main__":
	emnist_train, emnist_test = get_emnist_data()

	joint_train_inversion_model(dataset_train=emnist_train)