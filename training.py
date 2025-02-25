################################################################
#This is the sample code for the single qubit ground state preparation, two ancilla qubits are used for measurement and feedback while another two ancilla qubits are traced out
#Package info: Python 3.9.18 Numpy 1.24.3 Tensorflow 2.13.1 Pennylane 0.38.1
#After training, the weights of the model are save in './weights/checkpoint'
################################################################
import tensorflow as tf
import keras
from keras import mixed_precision
from keras.layers import Concatenate, concatenate
from keras.models import Model
from functools import partial
import pennylane as qml
import numpy as np
from tensorflow.python.ops.numpy_ops import np_config
from tensorflow.experimental import numpy as tnp
import random
import sys
import math
import time

tf.get_logger().setLevel('ERROR')
np_config.enable_numpy_behavior()
DEFAULT_TENSOR_TYPE = tf.float64
################################################################
#Parameters
################################################################
qbit = 5
num_epoch = 100000
QLL = 1
Len = 18
LL = QLL*Len
TT = 5
sam = 10
osam = 10
################################################################
#Data preparation
################################################################
initial_data = [[1.0+0.000000000000000000e+00j,  0.0+0.000000000000000000e+00j],
 [0.0+0.000000000000000000e+00j,  0.0+0.000000000000000000e+00j]]
initial_data = tf.convert_to_tensor(initial_data, dtype=tf.complex64)
initial_dataset = tf.data.Dataset.from_tensors(initial_data)
target_data = [[1.0+0.000000000000000000e+00j,  0.0+0.000000000000000000e+00j],
 [0.0+0.000000000000000000e+00j,  0.0+0.000000000000000000e+00j]]
target_data = tf.convert_to_tensor(target_data, dtype=tf.complex64)
target_dataset = tf.data.Dataset.from_tensors(target_data)
D_set = tf.data.Dataset.zip(initial_dataset,target_dataset)
Htx = [[0.0+0.000000000000000000e+00j,  1.0+0.000000000000000000e+00j],
 [1.0+0.000000000000000000e+00j,  0.0+0.000000000000000000e+00j]]
Hty = [[0.0+0.000000000000000000e+00j,  0.0-1.000000000000000000e+00j],
 [0.0+1.000000000000000000e+00j,  0.0+0.000000000000000000e+00j]]
Htz = [[1.0+0.000000000000000000e+00j,  0.0+0.000000000000000000e+00j],
 [0.0+0.000000000000000000e+00j,  -1.0+0.000000000000000000e+00j]]
Htx = tf.convert_to_tensor(Htx, dtype=tf.complex128)
Hty = tf.convert_to_tensor(Hty, dtype=tf.complex128)
Htz = tf.convert_to_tensor(Htz, dtype=tf.complex128)
upz = tf.constant([[1.+0.j, 0.+0.j], [0.+0.j, 0.+0.j]], dtype = tf.complex128)
downz = tf.constant([[0.+0.j, 0.+0.j], [0.+0.j, 1.+0.j]], dtype = tf.complex128)
Iz = tf.constant([[1.+0.j, 0.+0.j], [0.+0.j, 1.+0.j]], dtype = tf.complex128)
op_up1 = tf.experimental.numpy.kron(Iz,upz)
op_up1 = tf.experimental.numpy.kron(op_up1,Iz)
op_up1 = tf.experimental.numpy.kron(op_up1,Iz)
op_up1 = tf.experimental.numpy.kron(op_up1,Iz)
op_down1 = tf.experimental.numpy.kron(Iz,downz)
op_down1 = tf.experimental.numpy.kron(op_down1,Iz)
op_down1 = tf.experimental.numpy.kron(op_down1,Iz)
op_down1 = tf.experimental.numpy.kron(op_down1,Iz)
op_up2 = tf.experimental.numpy.kron(Iz,Iz)
op_up2 = tf.experimental.numpy.kron(op_up2,upz)
op_up2 = tf.experimental.numpy.kron(op_up2,Iz)
op_up2 = tf.experimental.numpy.kron(op_up2,Iz)
op_down2 = tf.experimental.numpy.kron(Iz,Iz)
op_down2 = tf.experimental.numpy.kron(op_down2,downz)
op_down2 = tf.experimental.numpy.kron(op_down2,Iz)
op_down2 = tf.experimental.numpy.kron(op_down2,Iz)
def initial_prep():
    rr = 1
    zz = np.random.uniform(-1,1)
    cos_zz = zz
    sin_zz = 1 - zz**2
    phi = np.random.uniform(0,2*np.pi)
    a = 0.5*(1+rr*cos_zz) + 0.j
    b = 1 - a
    c = rr/2*sin_zz*(np.cos(phi) - np.sin(phi)*1.j)
    c_ = rr/2*sin_zz*(np.cos(phi) + np.sin(phi)*1.j)
    state = np.array([[a,c],[c_,b]])
    state = tf.convert_to_tensor(state, dtype=tf.complex128)
    iupx = tf.random.uniform(shape=[1,1],minval=-1,maxval=1,dtype=DEFAULT_TENSOR_TYPE)
    iupy = tf.random.uniform(shape=[1,1],minval=-1,maxval=1,dtype=DEFAULT_TENSOR_TYPE)
    iupz = tf.random.uniform(shape=[1,1],minval=-1,maxval=1,dtype=DEFAULT_TENSOR_TYPE)
    iupxyz = tf.concat([tf.concat([iupx, iupy], 1), iupz], 1)
    return state, iupxyz
def initial_prep2():
    ini_theta1 = tf.random.uniform(shape=[2],minval=0,maxval=2*tnp.pi,dtype=DEFAULT_TENSOR_TYPE)
    state1 = inicuit(ini_theta1)
    state1 = tf.reshape(state1,[1,2,2])
    ini_theta2 = tf.random.uniform(shape=[2],minval=0,maxval=2*tnp.pi,dtype=DEFAULT_TENSOR_TYPE)
    state2 = inicuit(ini_theta2)
    state2 = tf.reshape(state2,[1,2,2])
    ini_theta3 = tf.random.uniform(shape=[2],minval=0,maxval=2*tnp.pi,dtype=DEFAULT_TENSOR_TYPE)
    state3 = inicuit(ini_theta3)
    state3 = tf.reshape(state3,[1,2,2])
    ini_theta4 = tf.random.uniform(shape=[2],minval=0,maxval=2*tnp.pi,dtype=DEFAULT_TENSOR_TYPE)
    state4 = inicuit(ini_theta4)
    state4 = tf.reshape(state4,[1,2,2])
    ini_theta5 = tf.random.uniform(shape=[2],minval=0,maxval=2*tnp.pi,dtype=DEFAULT_TENSOR_TYPE)
    state5 = inicuit(ini_theta5)
    state5 = tf.reshape(state5,[1,2,2])
    ip = tf.random.uniform(shape=[5],minval=0,maxval=1,dtype=DEFAULT_TENSOR_TYPE)
    ip_norm = ip[0]
    for i in range(4):
        ip_norm = ip_norm + ip[i+1]
    ip = ip/ ip_norm
    state = state1*ip[0] + state2*ip[1] + state3*ip[2] + state4*ip[3] + state5*ip[4]
    iup = tf.random.uniform(shape=[1,3],minval=-1,maxval=1,dtype=DEFAULT_TENSOR_TYPE)
    return state, iup
################################################################
#Quantumm circuit
################################################################
dev2 = qml.device("default.qubit", wires=1)
@qml.qnode(dev2, interface='tf')
def inicuit(theta):
    qml.RX(theta[0], wires=0)
    qml.RZ(theta[1], wires=0)
    return qml.density_matrix([0])
dev = qml.device("default.mixed", wires=qbit)
@qml.qnode(dev, interface='tf')
def selecircuit(inputs,theta):
    qml.QubitDensityMatrix(inputs, wires=0)
    for i in range(QLL):
        qml.RX(theta[i*Len+0], wires=0)
        qml.RX(theta[i*Len+1], wires=1)
        qml.RX(theta[i*Len+2], wires=2)
        qml.RX(theta[i*Len+3], wires=3)
        qml.RX(theta[i*Len+4], wires=4)
        qml.RY(theta[i*Len+5], wires=0)
        qml.RY(theta[i*Len+6], wires=1)
        qml.RY(theta[i*Len+7], wires=2)
        qml.RY(theta[i*Len+8], wires=3)
        qml.RY(theta[i*Len+9], wires=4)
        qml.CRX(theta[i*Len+10],wires=[0,1])
        qml.CRX(theta[i*Len+11],wires=[0,2])
        qml.CRX(theta[i*Len+12],wires=[0,3])
        qml.CRX(theta[i*Len+13],wires=[0,4])
        qml.CRX(theta[i*Len+14],wires=[1,0])
        qml.CRX(theta[i*Len+15],wires=[2,0])
        qml.CRX(theta[i*Len+16],wires=[3,0])
        qml.CRX(theta[i*Len+17],wires=[4,0])
    return qml.density_matrix([0,1,2,3,4])
@qml.qnode(dev, interface='tf')
def fcircuit(inputs,theta):
    qml.QubitDensityMatrix(inputs, wires=0)
    for i in range(QLL):
        qml.RX(theta[i*Len+0], wires=0)
        qml.RX(theta[i*Len+1], wires=1)
        qml.RX(theta[i*Len+2], wires=2)
        qml.RX(theta[i*Len+3], wires=3)
        qml.RX(theta[i*Len+4], wires=4)
        qml.RY(theta[i*Len+5], wires=0)
        qml.RY(theta[i*Len+6], wires=1)
        qml.RY(theta[i*Len+7], wires=2)
        qml.RY(theta[i*Len+8], wires=3)
        qml.RY(theta[i*Len+9], wires=4)
        qml.CRX(theta[i*Len+10],wires=[0,1])
        qml.CRX(theta[i*Len+11],wires=[0,2])
        qml.CRX(theta[i*Len+12],wires=[0,3])
        qml.CRX(theta[i*Len+13],wires=[0,4])
        qml.CRX(theta[i*Len+14],wires=[1,0])
        qml.CRX(theta[i*Len+15],wires=[2,0])
        qml.CRX(theta[i*Len+16],wires=[3,0])
        qml.CRX(theta[i*Len+17],wires=[4,0])
    return qml.density_matrix([0])
def measurecuit(dm):
    ##################### measuring the first ancilla qubit
    prob_up = tf.linalg.trace(tf.linalg.matmul(op_up1,dm))
    prob_down = tf.linalg.trace(tf.linalg.matmul(op_down1,dm))
    state_up = tf.linalg.matmul(tf.linalg.matmul(op_up1,dm),op_up1)/prob_up
    state_down = tf.linalg.matmul(tf.linalg.matmul(op_down1,dm),op_down1)/prob_down
    prob_up, prob_down = tf.reshape(prob_up,[1,1]), tf.reshape(prob_down,[1,1])
    prob_1 = tf.concat([prob_up, prob_down], 1)
    prob_1 = tf.math.real(prob_1)
    q_measure1 = tf.random.categorical(tf.math.log(prob_1),1,dtype=tf.int32)
    ps1 = tf.reshape(q_measure1, [])
    dm = if1(ps1,state_up,state_down)
    ##################### measuring the second ancilla qubit
    prob_up = tf.linalg.trace(tf.linalg.matmul(op_up2,dm))
    prob_down = tf.linalg.trace(tf.linalg.matmul(op_down2,dm))
    state_up = tf.linalg.matmul(tf.linalg.matmul(op_up2,dm),op_up2)/prob_up
    state_down = tf.linalg.matmul(tf.linalg.matmul(op_down2,dm),op_down2)/prob_down
    prob_up, prob_down = tf.reshape(prob_up,[1,1]), tf.reshape(prob_down,[1,1])
    prob_2 = tf.concat([prob_up, prob_down], 1)
    prob_2 = tf.math.real(prob_2)
    q_measure2 = tf.random.categorical(tf.math.log(prob_2),1,dtype=tf.int32)
    ps2 = tf.reshape(q_measure2, [])
    dm = if1(ps2,state_up,state_down)
    return ps1, ps2, qml.math.partial_trace(dm,indices = [1,2,3,4])
def if1(ps,state_up,state_down):
    pred = ps == 0
    true_fn =  lambda: state_up
    false_fn = lambda: state_down
    return tf.cond(pred, true_fn, false_fn)
def if2(q_measure):
    pred = q_measure == 0
    true_fn =  lambda: +1
    false_fn = lambda: -1
    return tf.cond(pred, true_fn, false_fn)
################################################################
#Loss function
################################################################
def sdloss(initial,iupxyz):
    iupxyz = tf.reshape(iupxyz,[3])
    iupx = tf.slice(iupxyz,begin=[0],size=[1])
    iupy = tf.slice(iupxyz,begin=[1],size=[1])
    iupz = tf.slice(iupxyz,begin=[2],size=[1])
    iupx, iupy, iupz = tf.cast(iupx, tf.complex128), tf.cast(iupy, tf.complex128), tf.cast(iupz, tf.complex128)
    iupx, iupy, iupz = tf.reshape(iupx,[]), tf.reshape(iupy,[]), tf.reshape(iupz,[])
    Hamiltonian = tf.scalar_mul(iupx, Htx) + tf.scalar_mul(iupy, Hty) + tf.scalar_mul(iupz, Htz)
    inp = tf.linalg.matmul(initial,Hamiltonian)
    inp = tf.linalg.trace(inp)
    inp = inp + tf.math.sqrt(iupx**2 + iupy**2 + iupz**2) - 1
    return inp
# ################################################################
#Custom Model
################################################################
class RNNModel(keras.Model):
    def __init__(self, model1, model2, model3, model4, model5):
        super().__init__()
        self.model1,self.model2,self.model3,self.model4,self.model5 = model1,model2,model3,model4,model5
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.m1_tracker = keras.metrics.Mean(name="m1")
        self.m2_tracker = keras.metrics.Mean(name="m2")
        self.m3_tracker = keras.metrics.Mean(name="m3")
        self.m4_tracker = keras.metrics.Mean(name="m4")
        self.m5_tracker = keras.metrics.Mean(name="m5")
        self.m6_tracker = keras.metrics.Mean(name="m6")
        self.m7_tracker = keras.metrics.Mean(name="m7")
        self.m8_tracker = keras.metrics.Mean(name="m8")
        self.m9_tracker = keras.metrics.Mean(name="m9")
        self.m10_tracker = keras.metrics.Mean(name="m10")
    def train_step(self, data):
        qs, target = data
        with tf.GradientTape() as tape:
            initial_state, iupxyz = initial_prep2()
            #1
            m_iupxyz0 = tf.reshape(iupxyz,[1,1,3])
            h_state11,c_state11 = self.model1(m_iupxyz0,None,None, training=True)
            dm1,measure1 = self.data_flowing(initial_state,h_state11,1)
            #2
            m_iupxyz1 = tf.reshape(tf.concat([measure1, iupxyz], 1),[1,1,5])
            h_state21,c_state21 = self.model2(m_iupxyz1,h_state11,c_state11, training=True)
            dm2,measure2 = self.data_flowing(dm1,h_state21,2)
            #3
            m_iupxyz2 = tf.reshape(tf.concat([measure2, iupxyz], 1),[1,1,5])
            h_state31,c_state31 = self.model3(m_iupxyz2,h_state21,c_state21, training=True)
            dm3,measure3 = self.data_flowing(dm2,h_state31,3)
            #4
            m_iupxyz3 = tf.reshape(tf.concat([measure3, iupxyz], 1),[1,1,5])
            h_state41,c_state41 = self.model4(m_iupxyz3,h_state31,c_state31, training=True)
            dm4,measure4 = self.data_flowing(dm3,h_state41,4)
            #5
            m_iupxyz4 = tf.reshape(tf.concat([measure4, iupxyz], 1),[1,1,5])
            h_state51,c_state51 = self.model5(m_iupxyz4,h_state41,c_state41, training=True)
            dm5 = self.data_flowing(dm4,h_state51,5)
            # print('dataset1',initial_state,dm1,dm2,dm3,dm4,dm5)
            # print('measure1',measure1,measure2,measure3,measure4)
            # with open("demo51_output.txt","a") as o:
            #     o.write(str(tf.linalg.trace(tf.linalg.matmul(initial_state,Htz)))+" "+str(tf.linalg.trace(tf.linalg.matmul(dm5,Htz)))+" "+measure1+" "+str(measure2)+" "+str(measure3)+" "+str(measure4)+" ")
            loss = sdloss(dm5,iupxyz)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.loss_tracker.update_state(loss)
        measure1,measure2,measure3,measure4 = tf.reshape(measure1,[2]),tf.reshape(measure2,[2]),tf.reshape(measure3,[2]),tf.reshape(measure4,[2])
        self.m1_tracker.update_state(tf.linalg.trace(tf.linalg.matmul(initial_state,Htz)))
        self.m2_tracker.update_state(tf.linalg.trace(tf.linalg.matmul(dm5,Htz)))
        self.m3_tracker.update_state(tf.slice(measure1,begin=[0],size=[1]))
        self.m4_tracker.update_state(tf.slice(measure1,begin=[1],size=[1]))
        self.m5_tracker.update_state(tf.slice(measure2,begin=[0],size=[1]))
        self.m6_tracker.update_state(tf.slice(measure2,begin=[1],size=[1]))
        self.m7_tracker.update_state(tf.slice(measure3,begin=[0],size=[1]))
        self.m8_tracker.update_state(tf.slice(measure3,begin=[1],size=[1]))
        self.m9_tracker.update_state(tf.slice(measure4,begin=[0],size=[1]))
        self.m10_tracker.update_state(tf.slice(measure4,begin=[1],size=[1]))
        return {"loss": self.loss_tracker.result(), "m1": self.m1_tracker.result(), "m2": self.m2_tracker.result(), "m3": self.m3_tracker.result(), "m4": self.m4_tracker.result(), "m5": self.m5_tracker.result(), "m6": self.m6_tracker.result(), "m7": self.m7_tracker.result(), "m8": self.m8_tracker.result(), "m9": self.m9_tracker.result(), "m10": self.m10_tracker.result()}
    def data_flowing(self,idm,h_state,nn):
        istate = idm
        h_state = tf.reshape(h_state, [LL])
        theta = tf.slice(h_state,begin=[0],size=[LL])
        theta = theta*tnp.pi
        if nn == 5:
            odm = fcircuit(istate,theta)
            return odm
        else:
            odm = selecircuit(istate,theta)
            q_measure1, q_measure2, odm = measurecuit(odm)
            q_measure1 = if2(q_measure1)
            q_measure2 = if2(q_measure2)
            o_measure1 = tf.reshape(q_measure1,[1,1])
            o_measure2 = tf.reshape(q_measure2,[1,1])
            o_measure = tf.concat([o_measure1, o_measure2], 1)
            o_measure = tf.cast(o_measure,dtype=DEFAULT_TENSOR_TYPE)
            return odm,o_measure
    @property
    def metrics(self):
        return [self.loss_tracker, self.m1_tracker, self.m2_tracker, self.m3_tracker, self.m4_tracker, self.m5_tracker, self.m6_tracker, self.m7_tracker, self.m8_tracker, self.m9_tracker, self.m10_tracker]
#######################################################################
#LTSM
#######################################################################
class sModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.RNN1 = tf.keras.layers.LSTM(LL, return_state=True)
        self.RNN2 = tf.keras.layers.LSTM(LL, return_state=True)

    def call(self, input,state1h,state1c):
        if state1c == None and state1h == None:
            RNN_11,RNN_12,RNN_13 = self.RNN1(input, training=True)
        else:
            RNN_11,RNN_12,RNN_13 = self.RNN1(input, initial_state=[state1c,state1h], training=True)
        RNN_11 = tf.reshape(RNN_11,[1,LL])
        # RNN_11 = tf.reshape(RNN_11,[1,1,LL])
        # if state2c == None and state2h == None:
        #     RNN_21,RNN_22,RNN_23 = self.RNN2(RNN_11, training=True)
        # else:
        #     RNN_21,RNN_22,RNN_23 = self.RNN2(RNN_11, initial_state=[state2c,state2h], training=True)
        # RNN_11 = tf.reshape(RNN_11,[1,LL])
        return RNN_11,RNN_13
################################################################
#Call back
################################################################
class haltCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.lossb = tf.zeros([num_epoch+1])
        self.stop = 0
        self.start = 0
        self.time = 0
        self.ave = 0
        self.begin = 0
    def on_epoch_begin(self, epoch, logs = None):
        self.start = time.time()
    def on_epoch_end(self, epoch, logs = None):
        indices = [[epoch]]
        update = logs.get("loss")
        i1,i2,i3,i4,i5,i6,i7,i8,i9,i10 = logs.get("m1"),logs.get("m2"),logs.get("m3"),logs.get("m4"),logs.get("m5"),logs.get("m6"),logs.get("m7"),logs.get("m8"),logs.get("m9"),logs.get("m10"),
        self.time = self.time + time.time() - self.start
        if epoch == 0:
            self.begin = time.time() - self.start
        self.ave = (self.time-self.begin)/(epoch+1)
        tf.print(epoch,' ',update,' ',self.ave,' ',self.begin)
        print(epoch,' ',update)
        # print(model.layers[1].get_weights())
        update = tf.reshape(update,[1])
        self.lossb = tf.tensor_scatter_nd_update(self.lossb,indices,update)
        # if(epoch > 0):
        #     if(abs(self.lossb[epoch]) <= 10**(-4)):
        #         self.stop = self.stop + 1
        #         if(self.stop == 300):
        #             print("stop training at loss = ",self.lossb[epoch-1])
        #             print('####################beigin sampling#########################')
        #             for i in range(osam):
        #                 self.model.sampling(initial_data,target_data)
        #             # tf.keras.callbacks.ModelCheckpoint(filepath="LTSM.ckpt",save_weights_only=True)
        #             self.model.stop_training = True
        if(epoch%1000 == 0):
            self.model.save_weights('./weights/checkpoint')
        if(epoch > 0):
            #if(abs((self.lossb[epoch]-self.lossb[epoch-1])/self.lossb[epoch]) <= 10**(-1)):
            if(abs((self.lossb[epoch]-self.lossb[epoch-1])/self.lossb[epoch]) <= 1*10**(-3)):
                self.stop = self.stop + 1
                if(self.stop == 10):
                    print("stop training at loss (convered)= ",self.lossb[epoch-1])
                    print('####################beigin sampling#########################')
                    self.model.save_weights('./weights/checkpoint')
                    self.model.stop_training = True
            else:
                self.stop = 0
        if epoch == num_epoch-1:
            print("stop training at loss (not converge)= ",self.lossb[epoch-1])
            print('####################beigin sampling#########################')
            self.model.save_weights('./weights/checkpoint')
            self.model.stop_training = True
            # if(abs((self.lossb[epoch]-self.lossb[epoch-1])/self.lossb[epoch]) <= 10**(-2)):
            #     self.stop = self.stop + 1
            #     if(self.stop == 3):
            #         print("stop training at loss = ",self.lossb[epoch-1])
            #         tf.keras.callbacks.ModelCheckpoint(filepath="LTSM.ckpt",save_weights_only=True)
            #         self.model.stop_training = True
            # else:
            #     self.stop = 0
################################################################
#Excuting 
################################################################
model1, model2, model3, model4, model5 = sModel(),sModel(),sModel(),sModel(),sModel()
model = RNNModel(model1, model2, model3, model4, model5)
model.compile(optimizer="Adam")
# model.run_eagerly = True
model.fit(x = D_set, batch_size=None, epochs=num_epoch+1, verbose = 0, callbacks=[haltCallback()])
