#!/usr/bin/env python
#!/usr/local/bin/python
#!/usr/bin/env PYTHONIOENCODING="utf-8" python
# -*- coding: utf-8 -*-

#importing useful libraries
import os 
#THIS IS NEEDED BECAUSE MAC DOESNT HAVE GPU SUPPORT
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
import tflearn
import speech_data as data
import tensorflow as tf

#CREATING A FUNCTION FOR SPEAKER CLASSIFIER BECAUSE IT HAS TO BE CALLED BY THE WEB SCRIPT
def speaker_classifier(name):
    demo_file = name
    speakers = data.get_speakers()
    number_classes=len(speakers)
    batch=data.wave_batch_generator(batch_size=1000, source=data.Source.DIGIT_WAVES, target=data.Target.speaker)
    X,Y=next(batch)

    #CREATING A NEURAL NETWORK GRAPH
    tflearn.init_graph(num_cores=8, gpu_memory_fraction=1)

    #SETTING UP INPUT DATA AND CREATING A FULLY CONNECTED NEURAL NETWORK
    net = tflearn.input_data(shape=[None, 8192]) #Two wave chunks
    net = tflearn.fully_connected(net, 64)
    net = tflearn.dropout(net, 0.5)
    net = tflearn.fully_connected(net, number_classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')
    
    
    #SAVING THE NEURAL MODEL CREATED
    model = tflearn.DNN(net)
    if(os.path.isfile('./model.tfl.index')):
        model.load('./model.tfl')
    else :
        model.fit(X, Y, n_epoch=100, show_metric=True, snapshot_step=100)
        model.save('model.tfl')
    demo=data.load_wav_file(demo_file)

    #PREDICTING THE VOICE ACROSSS THE MODEL GENERATED
    result=model.predict([demo])
    result=data.one_hot_to_item(result,speakers)
    return result
