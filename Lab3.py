import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import random
#from scipy.misc import imsave, imresize
from PIL import Image
from scipy.optimize import fmin_l_bfgs_b   # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from tensorflow.keras import metrics
import warnings
import winsound
import sys

random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

tf.compat.v1.disable_eager_execution()
#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CONTENT_IMG_PATH = "germany.jpg"
STYLE_IMG_PATH = "StyleImg.jpg"

CONTENT_IMG_H = 500
CONTENT_IMG_W = 500

STYLE_IMG_H = 500
STYLE_IMG_W = 500

IMG_CHANNELS = 3

CONTENT_WEIGHT = 0.7    # Alpha weight.
STYLE_WEIGHT = 0.3      # Beta weight.
TOTAL_WEIGHT = 8.5e-5

TRANSFER_ROUNDS = 3



#=============================<Helper Fuctions>=================================
'''
TODO: implement this.
This function should take the tensor and re-convert it to an image.
'''
def deprocessImage(img):
    img = img.reshape((CONTENT_IMG_H, CONTENT_IMG_W, IMG_CHANNELS))
    img = img[:, :, ::-1]
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = np.clip(img, 0, 255).astype("uint8")
    return Image.fromarray(img)



def gramMatrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))  #/ x.get_shape().num_elements()
    return gram



#========================<Loss Function Builder Functions>======================

def styleLoss(style, gen):
    styleG = gramMatrix(style)
    genG = gramMatrix(gen)
    M_l = STYLE_IMG_H * STYLE_IMG_W
    return K.sum(K.square(styleG - genG)) / (4.0 * (32^2) * (M_l ^ 2))
    #print(f"genG = {str(genG.shape)}, styleG = {str(styleG.shape)}")
    #return metrics.mse(styleG, genG)

def contentLoss(content, gen):
    return K.sum(K.square(gen - content))


def totalLoss(x):
    return None   #TODO: implement.





#=========================<Pipeline Functions>==================================

def getRawData():
    print("   Loading images.")
    print("      Content image URL:  \"%s\"." % CONTENT_IMG_PATH)
    print("      Style image URL:    \"%s\"." % STYLE_IMG_PATH)
    cImg = load_img(CONTENT_IMG_PATH)
    tImg = cImg.copy()
    sImg = load_img(STYLE_IMG_PATH)
    print("      Images have been loaded.")
    return ((cImg, CONTENT_IMG_H, CONTENT_IMG_W), (sImg, STYLE_IMG_H, STYLE_IMG_W), (tImg, CONTENT_IMG_H, CONTENT_IMG_W))



def preprocessData(raw):
    img, ih, iw = raw
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #imresize is deprecated so we're just gonna yeet this
        #img = imresize(img, (ih, iw, 3))
        img = img.resize((ih, iw))
    img = img_to_array(img)
    np.reshape(img, (ih, iw, 3))
    img = img.astype("float64")
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


'''
TODO: Allot of stuff needs to be implemented in this function.
First, make sure the model is set up properly.
Then construct the loss function (from content and style loss).
Gradient functions will also need to be created, or you can use K.Gradients().
Finally, do the style transfer with gradient descent.
Save the newly generated and deprocessed images.
'''
def styleTransfer(cData, sData, tData):
    print("   Building transfer model.")
    #Returns the input as an instantiated variable with keras metadata

    print(f"BLAH: {tf.executing_eagerly()}")

    contentTensor = K.variable(cData)   
    styleTensor = K.variable(sData)

    #A placeholder tensor
    genTensor = K.placeholder((1, CONTENT_IMG_H, CONTENT_IMG_W, 3)) 

    #concats all these tensors along axis specified
    inputTensor = K.concatenate([contentTensor, styleTensor, genTensor], axis=0) 
    
    #TODO: look into more input options for this like pooling type
    model = vgg19.VGG19(include_top=False, weights="imagenet", input_tensor=inputTensor)
    outputDict = dict([(layer.name, layer.output) for layer in model.layers])
    print("   VGG19 model loaded.")

    loss = 0.0
    styleLayerNames = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
    contentLayerName = "block5_conv2"
    styleWeights = [0.2, 0.3, 0.3, 0.3, 0.4]

    print("   Calculating content loss.")
    contentLayer = outputDict[contentLayerName]
    contentOutput = contentLayer[0, :, :, :]
    genOutput = contentLayer[2, :, :, :]

    closs = contentLoss(contentOutput, genOutput)
    loss += CONTENT_WEIGHT * closs

    sloss = 0
    print("   Calculating style loss.")
    for layerName, w in zip(styleLayerNames,styleWeights):
        styleLayer = outputDict[layerName]
        styleOutput = styleLayer[1, :, :, :]
        genOutput = styleLayer[2, :, :, :]
        #potentially divvy up these
        sloss += w * styleLoss(styleOutput, genOutput)

    loss += STYLE_WEIGHT * sloss

    loss *= TOTAL_WEIGHT
    # Setup gradients or use K.gradients().
    gradients = K.gradients(loss, genTensor)
    outputs = [loss]
    outputs.append(gradients)

    #outputs += gradients
    kGradientFunction = K.function([genTensor], outputs)
    optimizerfunc = optimizerFunction(kGradientFunction)

    imgToOpt = tData
    #print(f"tData: {type(tData)}")

    print("   Beginning transfer.")
    for i in range(TRANSFER_ROUNDS):
        print("   Step %d." % i)
        #Perform gradient descent using fmin_l_bfgs_b.
        imgToOpt, tLoss, info = fmin_l_bfgs_b(optimizerfunc.lossFunc, imgToOpt.flatten(), fprime = optimizerfunc.gradFunc, maxfun = 35)
        print("      Loss: %f." % tLoss)

        imgToSave = deprocessImage(imgToOpt)
        saveFile = "iteration_" + str(i) + ".png"

        imgToSave.save(saveFile)
        winsound.Beep(500, 250)
        print("      Image saved to \"%s\"." % saveFile)
        print("   Transfer complete.")

class optimizerFunction:
    def __init__(self, kfunc):
        self.kfunc = kfunc

    def lossFunc(self, img):
        img = img.reshape((1, CONTENT_IMG_H, CONTENT_IMG_W, IMG_CHANNELS))
        loss, self.gradients = self.kfunc([img])
        return loss

    def gradFunc(self, img): 
        self.gradients = np.asarray(self.gradients)
        return self.gradients.flatten().astype("float64")
        


#=========================<Main>================================================

def main():
    print("Starting style transfer program.")
    raw = getRawData()
    cData = preprocessData(raw[0])   # Content image.
    sData = preprocessData(raw[1])   # Style image.
    tData = preprocessData(raw[2])   # Transfer image.
    styleTransfer(cData, sData, tData)
    print("Done. Goodbye.")



if __name__ == "__main__":
    main()
