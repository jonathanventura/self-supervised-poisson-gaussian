import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, LeakyReLU, UpSampling2D, MaxPooling2D, ZeroPadding2D, Cropping2D, Concatenate, Reshape, GlobalAveragePooling2D, BatchNormalization, Add, Subtract, Layer
from tensorflow.keras.initializers import Constant
import tensorflow.keras.backend as K
import tensorflow as tf

class GaussianLayer(Layer):
    """ Computes noise std. dev. for Gaussian noise model. """
    def __init__(self, **kwargs):
        super(GaussianLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if not input_shape:
            # global parameter
            self.b = self.add_weight(name='b', 
                                        shape=(),
                                        initializer=Constant(0),
                                        trainable=True)
        super(GaussianLayer, self).build(input_shape)

    def call(self, x):
        noise_std = K.softplus(self.b-4)+1e-3
        return noise_std

    def compute_output_shape(self, input_shape):
        if not input_shape:
            return ()
        else:
            return input_shape

class PoissonLayer(Layer):
    """ Computes input-dependent noise std. dev. for Poisson noise model. """
    def __init__(self, **kwargs):
        super(PoissonLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.a = self.add_weight(name='a', 
                                      shape=(),
                                      initializer=Constant(0),
                                      trainable=True)
        super(PoissonLayer, self).build(input_shape)

    def call(self, x):
        noise_est = K.softplus(self.a-4) + 1e-3
        noise_std = (K.maximum(x, 1e-3) * noise_est) ** 0.5
        return noise_std

    def compute_output_shape(self, input_shape):
        return input_shape

class PoissonGaussianLayer(Layer):
    """ Computes input-dependent noise std. dev. for Poisson-Gaussian noise model. """
    def __init__(self, **kwargs):
        super(PoissonGaussianLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.a = self.add_weight(name='a', 
                                      shape=(),
                                      initializer=Constant(0),
                                      trainable=True)
        self.b = self.add_weight(name='b', 
                                      shape=(),
                                      initializer=Constant(0),
                                      trainable=True)
        super(PoissonGaussianLayer, self).build(input_shape)

    def call(self, x):
        poisson_noise_est = K.softplus(self.a-4) + 1e-3
        poisson_noise_var = K.maximum(x, 1e-3) * poisson_noise_est
        noise_var = K.maximum(poisson_noise_var + self.b,1e-3)
        noise_std = noise_var**0.5
        return noise_std

    def compute_output_shape(self, input_shape):
        return input_shape

def mse_loss(y,loc):
    """ Mean squared error loss function
        Use mean-squared error to regress to the expected value
        Parameters:
            loc: mean
    """
    loss = (y-loc)**2
    return K.mean(loss)

def uncalib_gaussian_loss(y,loc,std):
    """ Uncalibrated Gaussian loss function
        Model noisy data using a Gaussian parameterized by mean and std. dev.
        Parameters:
            loc: mean
            std: std. dev.
    """
    var = std**2
    total_var = var+1e-3
    loss = (y-loc)**2 / total_var + K.log(total_var)
    return K.mean(loss)

def gaussian_loss(y,loc,std,noise_std,reg_weight):
    """ Gaussian loss function
        Model noisy data using a Gaussian prior and Gaussian noise model
        Parameters:
            y: noisy input image
            loc: prior mean
            std: prior std. dev.
            noise_std: noise std. dev.
            reg_weight: strength of regularization on prior std. dev.
    """
    var = std**2
    noise_var = noise_std**2
    total_var = var+noise_var
    loss = (y-loc)**2 / total_var + K.log(total_var)
    reg = reg_weight * K.abs(std)
    return K.mean(loss+reg)

def gaussian_posterior_mean(y,loc,std,noise_std):
    """ Gaussian posterior mean
        Given noisy observation (y), compute optimal estimate for denoised image 
            y: noisy input image
            loc: prior mean
            std: prior std. dev.
            noise_std: noise std. dev.
    """
    var = std**2
    noise_var = noise_std**2
    total_var = var+noise_var
    return (loc*noise_var + var*y)/total_var
  
def _conv(x, num_filters, name):
  """ 2d convolution """
  filter_size = [3,3]

  x = Conv2D(filters=num_filters, kernel_size=filter_size, padding='same', kernel_initializer='he_normal', name=name)(x)
  x = LeakyReLU(0.1)(x)

  return x

def _vshifted_conv(x, num_filters, name):
  """ Vertically shifted convolution """
  filter_size = [3,3]
  k = filter_size[0]//2

  x = ZeroPadding2D([[k,0],[0,0]])(x)
  x = Conv2D(filters=num_filters, kernel_size=filter_size, padding='same', kernel_initializer='he_normal', name=name)(x)
  x = LeakyReLU(0.1)(x)
  x = Cropping2D([[0,k],[0,0]])(x)

  return x

def _pool(x):
  """ max pooling"""
  x = MaxPooling2D(pool_size=2,strides=2,padding='same')(x)

  return x

def _vshifted_pool(x):
  """ Vertically shifted max pooling"""
  x = ZeroPadding2D([[1,0],[0,0]])(x)
  x = Cropping2D([[0,1],[0,0]])(x)

  x = MaxPooling2D(pool_size=2,strides=2,padding='same')(x)

  return x

def _vertical_blindspot_network(x):
  """ Blind-spot network; adapted from noise2noise GitHub
    Each row of output only sees input pixels above that row
  """
  skips = [x]

  n = x
  n = _vshifted_conv(n, 48, 'enc_conv0')
  n = _vshifted_conv(n, 48, 'enc_conv1')
  n = _vshifted_pool(n)
  skips.append(n)

  n = _vshifted_conv(n, 48, 'enc_conv2')
  n = _vshifted_pool(n)
  skips.append(n)

  n = _vshifted_conv(n, 48, 'enc_conv3')
  n = _vshifted_pool(n)
  skips.append(n)

  n = _vshifted_conv(n, 48, 'enc_conv4')
  n = _vshifted_pool(n)
  skips.append(n)

  n = _vshifted_conv(n, 48, 'enc_conv5')
  n = _vshifted_pool(n)
  n = _vshifted_conv(n, 48, 'enc_conv6')

  #-----------------------------------------------
  n = UpSampling2D(2)(n)
  n = Concatenate(axis=3)([n, skips.pop()])
  n = _vshifted_conv(n, 96, 'dec_conv5')
  n = _vshifted_conv(n, 96, 'dec_conv5b')

  n = UpSampling2D(2)(n)
  n = Concatenate(axis=3)([n, skips.pop()])
  n = _vshifted_conv(n, 96, 'dec_conv4')
  n = _vshifted_conv(n, 96, 'dec_conv4b')

  n = UpSampling2D(2)(n)
  n = Concatenate(axis=3)([n, skips.pop()])
  n = _vshifted_conv(n, 96, 'dec_conv3')
  n = _vshifted_conv(n, 96, 'dec_conv3b')

  n = UpSampling2D(2)(n)
  n = Concatenate(axis=3)([n, skips.pop()])
  n = _vshifted_conv(n, 96, 'dec_conv2')
  n = _vshifted_conv(n, 96, 'dec_conv2b')

  n = UpSampling2D(2)(n)
  n = Concatenate(axis=3)([n, skips.pop()])
  n = _vshifted_conv(n, 96, 'dec_conv1a')
  n = _vshifted_conv(n, 96, 'dec_conv1b')

  # final pad and crop for blind spot
  n = ZeroPadding2D([[1,0],[0,0]])(n)
  n = Cropping2D([[0,1],[0,0]])(n)

  return n

def blindspot_network(inputs):
  b,h,w,c = K.int_shape(inputs)
  #if h != w:
    #raise ValueError('input shape must be square')
  if h % 32 != 0 or w % 32 != 0:
    raise ValueError('input shape (%d x %d) must be divisible by 32'%(h,w))

  # make vertical blindspot network
  vert_input = Input([h,w,c])
  vert_output = _vertical_blindspot_network(vert_input)
  vert_model = Model(inputs=vert_input,outputs=vert_output)

  # run vertical blindspot network on rotated inputs
  stacks = []
  for i in range(4):
      rotated = Lambda(lambda x: tf.image.rot90(x,i))(inputs)
      if i == 0 or i == 2:
          rotated = Reshape([h,w,c])(rotated)
      else:
          rotated = Reshape([w,h,c])(rotated)
      out = vert_model(rotated)
      out = Lambda(lambda x:tf.image.rot90(x,4-i))(out)
      stacks.append(out)

  # concatenate outputs
  x = Concatenate(axis=3)(stacks)

  # final 1x1 convolutional layers
  x = Conv2D(384, 1, kernel_initializer='he_normal', name='conv1x1_1')(x)
  x = LeakyReLU(0.1)(x)

  x = Conv2D(96, 1, kernel_initializer='he_normal', name='conv1x1_2')(x)
  x = LeakyReLU(0.1)(x)
  
  return x

def gaussian_blindspot_network(input_shape,mode,reg_weight=0):
    """ Create a variant of the Gaussian blindspot newtork.
        input_shape: Shape of input image
        mode: mse, uncalib, global, perpixel, poisson
              mse              -- regress to expected value using mean squared error loss 
              uncalib          -- model prior and noise together with single Gaussian at each pixel
              gaussian         -- Gaussian noise
              poisson          -- Poisson noise
              poissongaussian  -- Poisson-Gaussian noise
        reg_weight: strength of regularization on prior std. dev.
    """ 
    # create input layer
    inputs = Input(input_shape)
  
    # run blindspot network
    x = blindspot_network(inputs)
    
    # get prior parameters
    loc = Conv2D(1, 1, kernel_initializer='he_normal', name='loc')(x)
    if mode != 'mse':
        std = Conv2D(1, 1, kernel_initializer='he_normal', name='std')(x)
    
    # get noise variance
    if mode == 'mse':
        pass
    elif mode == 'uncalib':
        pass
    elif mode == 'gaussian':
        noise_std = GaussianLayer()([])
    elif mode == 'poisson':
        noise_std = PoissonLayer()(loc)
    elif mode == 'poissongaussian':
        noise_std = PoissonGaussianLayer()(loc)
    else:
        raise ValueError('unknown mode %s'%mode)
    
    # get outputs
    if mode == 'mse':
        outputs = loc
    elif mode == 'uncalib':
        outputs = [loc,std]
    else:
        outputs = Lambda(lambda x:gaussian_posterior_mean(*x))([inputs,loc,std,noise_std])
  
    # create model
    model = Model(inputs=inputs,outputs=outputs)
  
    # create loss function
    # input is evaluated against output distribution
    if mode == 'mse':
        loss = mse_loss(inputs,loc)
    elif mode == 'uncalib':
        loss = uncalib_gaussian_loss(inputs,loc,std)
    else:
        loss = gaussian_loss(inputs,loc,std,noise_std,reg_weight)
    model.add_loss(loss)
  
    return model

