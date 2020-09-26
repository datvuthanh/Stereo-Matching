from keras.layers import Conv2D,BatchNormalization
from keras.models import Sequential


def create_network(inputs, input_shape, scope="win19_dep9"):
	
	num_maps = 64
	kw = 3
	kh = 3
	
	net = Conv2D(num_maps, (kw, kw), input_shape=input_shape, padding='valid', activation='relu')(inputs)
	net = BatchNormalization()(net)

	net = Conv2D(num_maps, (kw, kw), padding='valid', activation='relu')(net)
	net = BatchNormalization()(net)

	net = Conv2D(num_maps, (kw, kw), padding='valid', activation='relu')(net)
	net = BatchNormalization()(net)
    
	net = Conv2D(num_maps, (kw, kw), padding='valid', activation='relu')(net)
	net = BatchNormalization()(net)
    
	net = Conv2D(num_maps, (kw, kw), padding='valid', activation='relu')(net)
	net = BatchNormalization()(net)
    
	net = Conv2D(num_maps, (kw, kw), padding='valid', activation='relu')(net)
	net = BatchNormalization()(net)
    
	net = Conv2D(num_maps, (kw, kw), padding='valid', activation='relu')(net)
	net = BatchNormalization()(net)
    
	net = Conv2D(num_maps, (kw, kw), padding='valid', activation='relu')(net)
	net = BatchNormalization()(net)

	net = Conv2D(num_maps, (kw, kw), padding='valid')(net)
	net = BatchNormalization()(net)
	
	
	return net



