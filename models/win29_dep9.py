from keras.layers import Conv2D,BatchNormalization
from keras.models import Sequential

# n = 29, 5 layers of kernel 5X5, 4 layers of kernel 3X3, 9 layers in total
def create_network(inputs, input_shape, scope="win29_dep9"):
	num_maps = 64
	kw1 = 3
	kh1 = 3
	kw2 = 5
	kh2 = 5
	
	net=Conv2D(num_maps, (kw2, kw2), input_shape=input_shape, padding='valid', activation='relu')(inputs)
	net=BatchNormalization()(net)

	net=Conv2D(num_maps, (kw2, kw2), padding='valid', activation='relu')(net)
	net=BatchNormalization()(net)

	net=Conv2D(num_maps, (kw2, kw2), padding='valid', activation='relu')(net)
	net=BatchNormalization()(net)
    
	net=Conv2D(num_maps, (kw2, kw2), padding='valid', activation='relu')(net)
	net=BatchNormalization()(net)
    
	net=Conv2D(num_maps, (kw2, kw2), padding='valid', activation='relu')(net)
	net=BatchNormalization()(net)
    
	net=Conv2D(num_maps, (kw1, kw1), padding='valid', activation='relu')(net)
	net=BatchNormalization()(net)
    
	net=Conv2D(num_maps, (kw1, kw1), padding='valid', activation='relu')(net)
	net=BatchNormalization()(net)
    
	net=Conv2D(num_maps, (kw1, kw1), padding='valid', activation='relu')(net)
	net=BatchNormalization()(net)

	net=Conv2D(num_maps, (kw1, kw1), padding='valid')(net)
	net=BatchNormalization()(net)
    

	return net



