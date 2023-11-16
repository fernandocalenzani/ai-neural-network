from pybrain.structure import (BiasUnit, FeedForwardNetwork, Fullconnection,
                               LinearLayer, SigmoidLayer)

nn = FeedForwardNetwork()

x_layer = LinearLayer(2)
h_layer = SigmoidLayer(3)
y_layer = SigmoidLayer(1)

bias_x = BiasUnit()
bias_h = BiasUnit()

nn.addModule(x_layer)
nn.addModule(h_layer)
nn.addModule(y_layer)

nn.addModule(bias_x)
nn.addModule(bias_h)

connect_xh = Fullconnection(x_layer, h_layer)
connect_hy = Fullconnection(h_layer, y_layer)
connect_bias_x = Fullconnection(bias_x, h_layer)
connect_bias_h = Fullconnection(bias_h, y_layer)

nn.sortModules()

print(nn)
