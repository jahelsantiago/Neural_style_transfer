import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from utils import *
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg

def transfer_style(content_image, style_image, save_preimage = False, num_iterations= 200):
    """
    content_image- numpy array as image (300,400,3)
    style_image- numpy array as image of shape (300,400,3)
    save_preimage- boolean save the step images
    num_iterations- integer, number of iterations
    
    generated_image- numpy array as image (300,400,3)    
    """
    STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]
    
    def compute_content_cost(a_C, a_G):
        """
        a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
        a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

        J_content -- scalar result of compute content formula
        """
        m, n_H, n_W, n_C = a_G.get_shape().as_list()

        # Reshape a_C and a_G (desenrrollar)
        a_C_unrolled = tf.reshape(a_C, shape = [m, -1, n_C])
        a_G_unrolled = tf.reshape(a_G, shape = [m, -1, n_C])

        # compute the cost
        J_content = tf.reduce_sum((a_C_unrolled - a_G_unrolled)**2)*(1/(4*n_H*n_W*n_C))    

        return J_content   

    def gram_matrix(A):
        """
        Argument:
        A -- matrix of shape (n_C, n_H*n_W)

        Returns:
        GA -- Gram matrix of A, of shape (n_C, n_C)
        """
        GA = tf.matmul(A,tf.transpose(A))
        return GA
    def compute_layer_style_cost(a_S, a_G):
        """
        Arguments:
        a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
        a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

        Returns: 
        J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
        """
        m, n_H, n_W, n_C = a_G.get_shape().as_list() #traer las dimensiones de la activacion

        a_S = tf.transpose(tf.reshape(a_S, shape = [n_H*n_W,n_C])) #cambiar las dimensiones a (n_C, n_H*n_W)
        a_G = tf.transpose(tf.reshape(a_G, shape = [n_H*n_W,n_C]))

        GS = gram_matrix(a_S) #creamos la matriz de style 
        GG = gram_matrix(a_G)

        J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS, GG)), axis = None) / ((2*n_H*n_W*n_C)**2) #calcular la funcion loss

        return J_style_layer

    def compute_style_cost(model, STYLE_LAYERS):
        """        
        Arguments:
        model -- our tensorflow model
        STYLE_LAYERS -- A python list containing:
                            - the names of the layers we would like to extract style from
                            - a coefficient for each of them

        Returns: 
        J_style -- tensor representing a scalar value, style cost defined above by equation (2)
        """
        J_style = 0
        for layer_name, coeff in STYLE_LAYERS:
            out = model[layer_name] #seleccionamos solo la layer requerida
            a_S = sess.run(out) #corremos la sesion en esta para obtener sus activaciones        
            a_G = out

            J_style_layer = compute_layer_style_cost(a_S, a_G) #calculamos el style cost 

            J_style += coeff * J_style_layer #los multiplocamos por los coeficientes

        return J_style
    
    def total_cost(J_content, J_style, alpha = 10, beta = 40):
        """        
        Arguments:
        J_content -- content cost coded above
        J_style -- style cost coded above
        alpha -- hyperparameter weighting the importance of the content cost
        beta -- hyperparameter weighting the importance of the style cost

        Returns:
        J -- total cost as defined by the formula above.
        """
        J = alpha*J_content + beta*J_style
        return J
    
    def model_nn(sess, input_image, num_iterations):    
        sess.run(tf.global_variables_initializer()) #inicializar variables globales
        sess.run(model["input"].assign(input_image)) #corremos la imagen inicial a traves del modelo    

        for i in range(num_iterations):            
            sess.run(train_step) #correr la sesion en el train_step para minimizar                                
            generated_image = sess.run(model['input']) #computar la imagen generada 

            if i%20 == 0 and save_preimage == True:
                Jt, Jc, Js = sess.run([J, J_content, J_style])
                save_image_2("output/" + str(i) + ".png", generated_image) #guardar la imagen del avance parcial            

        save_image_2('output/generated_image.jpg', generated_image) #guardar la imagen definitiva
        return generated_image
    
    
    tf.reset_default_graph() #resetear las graph
    sess = tf.InteractiveSession() #iniciar la sesion
    
    content_image = reshape_and_normalize_image(content_image) #normalizar la imagen content
    style_image = reshape_and_normalize_image(style_image) #normalizar la imagen style
    
    generated_image = generate_noise_image(content_image) #generamos la imagen nueva
    
    path = r"imagenet-vgg-verydeep-19.mat" #cargar el modelo
    model = load_vgg_model(path)
    
    sess.run(model["input"].assign(content_image)) #asignamos la content image para ser el input del modelo vgg-19
    out = model['conv4_2'] #seleccionamos el output
    a_C = sess.run(out) #asignams el output de la imagen content
    a_G = out #asignamos el output de la imagen generada
    
    J_content = compute_content_cost(a_C, a_G) #calculamos la funcion loss para el content
    
    sess.run(model['input'].assign(style_image)) #asignamos el input como la imagen generada
    J_style = compute_style_cost(model, STYLE_LAYERS) #calcular el style cost
    
    
    J = total_cost(J_content, J_style, alpha = 10, beta = 40) #asigamos el costo total
    
    #asignar el optimizador
    optimizer = tf.train.AdamOptimizer(2.0) #usamos adam con learning rate = 2
    train_step = optimizer.minimize(J) #definimos el objetivo
    
    final_image = model_nn(sess, generated_image, num_iterations)
    return final_image
    
