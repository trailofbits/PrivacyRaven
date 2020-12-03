"""
The synthesis tests rely on sampling data from a model.
We will be training one and returning a query function here
and not inside of a separate function in order to minimize
the cycles dedicated for training this model.
"""

model = train_mnist_victim(gpus=0)


def query_mnist(input_data):
    return get_target(model, input_data)


def valid_query():
    return st.just(query_mnist)


def valid_data():
    return arrays(np.float64, (10, 28, 28, 1), st.floats())


emnist_train, emnist_test = get_emnist_data()
