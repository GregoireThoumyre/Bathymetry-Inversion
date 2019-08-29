
import context
from src.models.cnn import Model1, Model2, Model3
from src.models.autoencoder import Model1 as aem1
from keras.utils import plot_model

print('########### Model 1 ae ###########')

modelae1 = aem1((520, 480, 1))

modeldec = modelae1.autoencoder()
modelenc = modelae1.encoder()
modelenc.summary()
plot_model(modelenc, 'modelenc.png', show_shapes=True)
modeldec.summary()
plot_model(modeldec, 'modeldec.png', show_shapes=True)

print('########### Model 1 ###########')

model1 = Model1((130, 480, 1), 480)

model = model1.model()
model.summary()
plot_model(model, 'model1.png', show_shapes=True)

print('########### Model 2 ###########')

model2 = Model2((130, 480, 1), 480)

model = model2.model()
model.summary()
plot_model(model, 'model2.png', show_shapes=True)

print('########### Model 3 ###########')

model3 = Model3((1, 405, 1), 405)

model = model3.model()
model.summary()
plot_model(model, 'model3.png', show_shapes=True)
