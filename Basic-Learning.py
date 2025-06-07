"""
this is for a basic neural network.
so, don't expect much.
-------------------------------------------------------------------
#NOTE:- this framework is going to get a rework in july, 2025. STAY TUNED!!!!!!
-------------------------------------------------------------------
Basic-Learning name isn't a good name in python as trying to import it shows error, do this instead:-

import importlib.util
import sys

path = "/path/to/the/file"
name = "filename"

spec = importlib.util.spec_from_file_location(name, path)
alias = importlib.util.module_from_spec(spec)
sys.modules[name] = alias
spec.loader.exec_module(alias)

"""
import math
"""
I'm going to use math, tho, I think numpy would've been better, switching between numpy and lists give me errors, which I don't want'
"""
import random
"""
random is going to be used to randomly generate weights and biases
"""
import pickle
"""
I'm going to use pickle to save and load
"""
import statistics
"""
this is going to be used to calculate mean and median in the batch training
"""
from copy import deepcopy
"""
this is going to be used to copy a network object
"""
def identity(x):
    """
    in case you don't want any activation function'
    """
    return x
def d_identity(x):
    """
    always 1
    """
    return 1
def sigmoid(x):
    """
    the good old sigmoid, with vanishing gradients. IT'S OUTDATED(why're you using it?)
    """
    try:
        """
        I had to add this try-except block to stop the sigmoid from blowing up
        """
        return 1/(1+math.exp(-x))
    except:
        return 0 if x < 0 else 1
def d_sigmoid(x):
    """
    for backpropogation, we need the partial derivative of the activation functions.
    """
    a = sigmoid(x)
    return a*(1-a)
def relu(x):
    """
    Relu is my least faviorite, I hate, why's it even used? it's not even diffentiable at x = 0
    """
    return max(0, x)
def d_relu(x):
    """
    Relu's derivative is either 1 or 0 at most of the time, for x = 0, I'm going to assume it's 0.
    """
    if x > 0:
        return 1
    else:
        return 0
def tanh(x):
    """
    tanh is better than sigmoid, but, still the same problem'
    """
    try:
        return (math.exp(x) - math.exp(-x))/(math.exp(x) + math.exp(-x))
    except:
        return -1 if x < 0 else 1
def d_tanh(x):
    """sigmoid and tanh have their own function in it, which is good.
    """
    a = tanh(x)
    return 1- a**2
def swish(x):
    """
    "what is love?", and the answer is swish, just joking, but swish is very good.
    """
    return x*sigmoid(x)
def d_swish(x):
    """
    the derivative of swish
    """
    s = sigmoid(x)
    return s + x*s*(1-s)
def leakyrelu(x, a = 0.01):
    """
    LeakyRelu is Relu , but better.
    """
    if x > 0:
        return x
    else:
        return x*a
def d_leakyrelu(x, a = 0.01):
    """
    don't forget to keep the input same for the derivatives and the functions, or else it's going to learn something else.
    """
    if x > 0:
        return 1
    else:
        return a
        
class forwardpass:
    """
    I'm going to explain neural networks, listen closely
    """
    def __init__(s, inputs, fhlw, fhlb ,olw, olb, weights, biases, activation_function = swish, fla = swish, ola = swish):
        """
        give it arguments. to have flexiblity, the first hidden layer and output layer doesn't have to have the same amount of neurons, inputs, weights, biases etc.
        """
        s.i = inputs
        s.fw = fhlw
        s.fb = fhlb
        s.ow = olw
        s.ob = olb
        s.w = weights
        s.b = biases
        s.a = activation_function
        s.f = fla
        s.o = ola
    @staticmethod
    def nrn(i, w, bias, act_func = swish):
        """
        in simple words, a neuron takes a bunch of inputs, multiplies them by a weight for each, sums that, adds a bias and applies a activation function
        """
        assert len(i) == len(w), f"{len(i)} doesn't match {len(w)}"
        return act_func(sum([i[a]*w[a] for a in range(len(i))]) + bias)
    @staticmethod
    def lyr(inputs, weights, bias, act_func = swish):
        """
        a layer consists of multiple neurons, they all get the same input, but they don't give the same output'
        """
        nrn = forwardpass.nrn
        output = []
        for a in range(len(bias)):
            output.append(nrn(inputs, weights[a], bias[a], act_func))
        return output
    def nn(s):
        """
        a it's just the forwardpass of neural network, backpropagation or backwardpass is still remaining, without it, the net won't learn
        """
        lyr = forwardpass.lyr
        first = lyr(s.i, s.fw, s.fb, s.f)
        a = 0
        ins = [first]
        i = first 
        while a < len(s.b):
            w = s.w[a]
            b = s.b[a]
            i = lyr(i, w, b, s.a)
            ins.append(i)
            a += 1
        output = lyr(i, s.ow, s.ob, s.o)
        last = ins.pop()
        return [s.i, ins, last, output]
def _sign(x):
    """
    this function is used for this module, I don't think you need to use this function.'
    """
    if x < 0:
        return -1
    elif x > 0:
        return 1
    else:
        return 0
        
def mae(x, y):
    """
    it's just straight up the difference'
    """
    assert len(x) == len(y)
    return sum([abs(x[a] - y[a]) for a in range(len(x))])/len(x)
def d_mae(x, y):
    """
    derivatives are for single output_neuron
    """
    assert len(x) == len(y)
    return [[_sign(x[a] - y[a])/len(x)] for a in range(len(x))]
    
def mse(x, y):
    """
    mse is good
    """
    assert len(x) == len(y)
    return sum([(x[a] - y[a])**2 for a in range(len(x))])/len(x)
    
def d_mse(x, y):
    """
    it's derivative is good too
    """
    assert len(x) == len(y), f"{len(x)} doesn't match {len(y)}"
    return [[2*(x[a] - y[a])/len(x)] for a in range(len(x))]
    
def _clip(x, y, z):
    """
    this function is just a clip function implemented.
    you may not have any use of it.
    """
    if x < y:
        return y
    elif x > z:
        return z
    else:
        return x
        
def cel(x, y):
    """
    log term needs cliping, which I don't like.'
    """
    assert len(x) == len(y)
    eps = math.exp(-37)
    y = [_clip(i , eps, 1-eps) for i in y]
    return -sum([x[a]*math.log(y[a])+(1-x[a])*math.log(1-y[a]) for a in range(len(x))])/len(x)
    
def d_cel(x, y):
    """
    it's good for binary .'
    """
    assert len(x) == len(y)
    eps = math.exp(-37)
    y = [_clip(i , eps, 1-eps) for i in y]
    return [[((x[a] - y[a])/(y[a]*(1-y[a])))/len(x)] for a in range(len(x))]
    
def up(x, dx, lr):
    """
    this is to calculate the new weight and bias according to the gradient.
    """    
    return x - dx*lr
def up0(x, dx, lr):
    """
    this is to update the weights of a neuron or the biases of a layer.
    """    
    assert len(x) == len(dx)
    nx = []
    for a in range(len(x)):
        nx.append(up(x[a], dx[a], lr))
    return nx
def up1(x, dx, lr):
    """
    this is to update the weights of a layer or the biases of the network.
    """       
    assert len(x) == len(dx)
    nx = []
    for a in range(len(x)):
        nx.append(up0(x[a], dx[a], lr))
    return nx
def up2(x, dx, lr):
    """
    this is to update the weights of a network
    """       
    assert len(x) == len(dx)
    nx = []
    for a in range(len(x)):
        nx.append(up1(x[a], dx[a], lr))
    return nx
    
def _trans(x):
    """
    this is the transpose function, it'll come in use later, for me. I don't think you'll need it.'
    """
    a = []
    for b in range(len(x[0])):
        c = []
        for d in range(len(x)):
            c.append(x[d][b])
        a.append(c)
    return a
    
class backwardpass:
    """
    the part where the network "learns"! is backpropagation. give it the outputs, I'm going to explain the backpropagation, listen carefully'
    """
    def __init__(s, output , first, ins, last, fw, fb, ow, ob,w, b, desired, a = d_swish, fa = d_swish, la = d_swish, ef = d_mse, lr = 0.1):
        """
        btw, don't forget to give them inputs correctly! cause there is no errorhandeling done by me, I'm too lazy.
        """
        s.output = output 
        s.first = first
        s.ins = ins
        s.last = last
        s.fw = fw
        s.fb = fb
        s.ow = ow
        s.ob = ob
        s.w = w
        s.b = b
        s.desired = desired
        s.a = a
        s.fa = fa
        s.la = la
        s.ef = ef
        s.lr = lr
    @staticmethod
    def neuron(i, w, bias, previous, a_func = d_swish):
        """
        so, to explain , listen carefully.
        for backpropagation/backpass , we use the chainrule.
        let's think of how a neuron would update.
        let's assume we took the partial derivative of the error according the output and desired output.
        next, we'll update the output layer, let's take a neuron of it.
        the inputs in the arguments is the input it got when it was forward passing.
        the weights are the weights of the connections it has.
        bias is it's bias.
        a_func is it's activation function.
        lr is the learning rate of the network.
        so, first to get the new bias:-
        1. we sum all of the previous(since for the outputlayer, the previous is list containing the error function derivative according to the neurons output and the desired output(not the networks desired output)).
        2. multiply it by the derivative of activation function of bias + sum(inputs*weights)
       and the result is the gradient of bias.
        for weight:-
       1. multiply it by the inputs to get a gradient for each weight, I'm assuming you didn't give the array of input different length than the array of weights, which is likely if you are dumb.
       to calculate the gradients of the previous layer, we'll need to pass some info. it's just gradient of bias * the new weights(which is old_weights - gradient of weights * learning_rate)
       for the next/previous layer(whether you would like to call it). due to it having affect on all the outputs, we sum the weighted gradients.
        """
        assert len(i) == len(w)
        z = sum([i[a]*w[a] for a in range(len(i))]) + bias
        b = a_func(z)*sum(previous)
        return [[b*i[a] for a in range(len(i))], b, [b*w[a] for a in range(len(w))]]
    @staticmethod
    def layer(i, w, b, previous, a_func = d_swish):
        """
        to find the gradients of a layer,  do what's done to a neuron for each neuron.
        """
        assert len(w) == len(b)
        p = previous
        assert len(b) == len(p)
        z = [backwardpass.neuron(i, w[a], b[a], p[a], a_func) for a in range(len(b))]
        nw = []
        nb = []
        np = []
        for e in z:
            nw.append(e[0])
            nb.append(e[1])
            np.append(e[2])
        np = _trans(np)   
        return [nw, nb, np]
    def net(s):
        """
        for the net, first calculate it for the last/output layer. and go in reverse and then, atlast calculate it for the first layer'
        """
        lyr = backwardpass.layer
        error = s.ef(s.output, s.desired)
        last = lyr(s.last , s.ow, s.ob, error, s.la)
        a = 1
        lw = last[0]
        lb = last[1]
        p = last[2]
        nw = []
        nb = []
        while a <= len(s.ins):
            num = len(s.ins) - a
            w = s.w[num]
            b = s.b[num]
            ins = s.ins[num]
            c = lyr(ins, w, b, p, s.a)
            nw.append(c[0])
            nb.append(c[1])
            p = c[2]
            a = a + 1
        first = lyr(s.first, s.fw, s.fb, p, s.fa)
        fw = first[0]
        fb = first[1]
        nw = nw[::-1]
        nb = nb[::-1]
        fw = up1(s.fw, fw, s.lr)
        fb = up0(s.fb, fb, s.lr)
        nw = up2(s.w, nw, s.lr)
        nb = up1(s.b, nb, s.lr)
        lw = up1(s.ow, lw, s.lr)
        lb = up0(s.ob, lb, s.lr)
        return [fw, fb, nw, nb, lw, lb]
        
def avg(list, func = statistics.mean):
        """
        used for batch processing of the weights of first and output layer.
        """
        l = _trans(list)
        return [func(a) for a in l]
        
def avg0(list, func = statistics.mean):
    """
    used for batch processing of the weights of the first and output layer. and the biases of the other layers
    """
    l = _trans(list)
    return [avg(a, func) for a in l]
    
def avg1(list, func = statistics.mean):
    """
    used for batch processing of the weights of the other layers
    """
    l = _trans(list)
    return [avg0(a, func) for a in l]
    
class network:
    """
    you don't like to manually use forwardpass and then backpropagation.
    you'd want to just give some things and make the model learn.
    so, this network class is for that
    """
    def __init__(self, inputs, fw, fb ,ow, ob, weights, biases, desired, a = swish,ad = d_swish, fa = swish,fad = d_swish, oa = swish,oad = d_swish, ef = mse, efd = d_mse, lr = 0.1):
        """
        give it everything it needs
        """
        self.i = inputs
        self.fw = fw
        self.fb = fb
        self.ow = ow
        self.ob = ob
        self.w = weights
        self.b = biases
        self.d = desired
        self.a = a
        self.ad = ad
        self.fa = fa
        self.fad = fad
        self.oa = oa
        self.oad = oad
        self.ef = ef
        self.efd = efd
        self.lr = lr
    def view(self):
        """
        in case you want to copy the network
        """
        return network(self.i, self.fw, self.fb, self.ow, self.ob, self.w, self.b, self.d, self.a, self.ad, self.fa, self.fad, self.oa, self.oad, self.ef, self.efd, self.lr)
        
    def copy(self):
        i = deepcopy(self.i)
        fw = deepcopy(self.fw)
        fb = deepcopy(self.fb)
        ow = deepcopy(self.ow)
        ob = deepcopy(self.ob)
        w = deepcopy(self.w)
        b = deepcopy(self.b)
        d = deepcopy(self.d)
        a = deepcopy(self.a)
        ad = deepcopy(self.ad)
        fa = deepcopy(self.fa)
        fad = deepcopy(self.fad)
        oa = deepcopy(self.oa)
        oad = deepcopy(self.oad)
        ef = deepcopy(self.ef)
        efd = deepcopy(self.efd)
        lr = deepcopy(self.lr)
        return network(i, fw, fb, ow, ob, w, b, d, a, ad, fa, fad, oa, oad, ef, efd, lr)
        
    def learn(self, show_error = True, lrd = 0):
        """
        do both forward and backward pass
        """
        forward = forwardpass(self.i, self.fw, self.fb, self.ow, self.ob , self.w , self.b , self.a, self.fa, self.oa)
        forresult = forward.nn()
        _ , ins, last, output = forresult
        error = self.ef(output, self.d)
        if show_error:
            print(error)
        backward = backwardpass(output , self.i, ins, last, self.fw, self.fb, self.ow, self.ob, self.w, self.b, self.d, self.ad, self.fad, self.oad, self.efd , self.lr)
        backresult = backward.net()
        self.lr = up(self.lr, lrd, self.lr)
        return (backresult, output, error)
            
    def train(self, show_error = True, lrd = 0):
        """
        train for a single epoch
        """
        backresult, _, err = self.learn(show_error, lrd)
        self.fw = backresult[0]
        self.fb = backresult[1]
        self.w = backresult[2]
        self.b = backresult[3]
        self.ow = backresult[4]
        self.ob = backresult[5]
        return _, err
    
    def single(self, epochs = 100, show_error = 1, lrd = 0):
        """
        single input set learning
        """
        for a in range(epochs):
            if show_error not in (None, False, 0):
                se = True if a%show_error == 0 else False
            else:
                se = False
            _, err = self.train(se , lrd)
            
    def multi(self, epochs = 100, show_error = True, lrd = 0):
        """
        learn for a dataset.
        this version may overfit for each input-desired set
        better to use itertive
        """
        a = self.i
        b = self.d
        e = 1
        for c, d in zip(a, b):
            self.i = c
            self.d = d
            self.single(epochs, show_error, lrd)
            print(f"{e} completed")
            e += 1
        self.i = a
        self.d = b
        
    def iterative(self, epochs = 100, show_error = True, lrd = 0):
        """
        best learning style I have Implemented.
        tho, batch processing would be better
        l'm too lazy to make batch processing
        #NOTE:- I'VE IMPLEMENTED BATCH PROCESSING NOW!!! YIPEEE!!!!!!!!!!!!
        so, batch processing is the best learning style now
        """
        se = True if show_error in (True, "all") else False
        for a in range(epochs):
            b = self.i
            c = self.d
            err = []
            for d, e in zip(b, c):
                self.i = d
                self.d = e
                _, error = self.train(se, lrd)
                err.append(error)
            if show_error != "all":
                if a%show_error == 0:
                    print(statistics.mean(err))
            self.i = b
            self.d = c
            
    def batch(self, epochs = 100, show_error = True, lrd = 0, func = statistics.mean):
        """
        truly the best learning style I have Implemented.
        you average the gradients for multiple dataset
        """
        se = True if show_error == "all" else False
        for a in range(epochs):
            b = self.i
            c = self.d
            gfw = []
            gfb = []
            gw = []
            gb = []
            gow = []
            gob = []
            err = []
            for d, e in zip(b, c):
                self.i = d
                self.d = e
                rs , output, error = self.learn(se, lrd)
                gfw.append(rs[0])
                gfb.append(rs[1])
                gw.append(rs[2])
                gb.append(rs[3])
                gow.append(rs[4])
                gob.append(rs[5])
                err.append(error)
            self.fw = avg0(gfw, func)
            self.fb = avg(gfb, func)
            self.w = avg1(gw, func)
            self.b = avg0(gb, func)
            self.ow = avg0(gow, func)
            self.ob = avg(gob, func)
            if show_error not in ("all", 0):
                if a%show_error == 0:
                    print(statistics.mean(err))
            self.i = b
            self.d = c
           
    def save(self, name):
        """
        in case you fight want to save your model
        """
        with open(name + ".NN", "wb") as f:
            pickle.dump(self, f)
    @staticmethod
    def load(name):
        """
        in case you want to load a saved model
        """
        with open(name + ".NN", "rb") as f:
            NN = pickle.load(f)
        return NN
        
class initialize:
    """
    who wants to manually write the the weights and biases(unless you're a masochist).
    use this to make your life easier
    """
    def __init__(self, i, n, l, on):
        """
        btw, don't forget to give it some info
        """
        self.i = i
        self.n = n
        self.l = l
        self.on = on

    def constant(self, fw=1, fb=0, w=1, b=0, ow=1, ob=0):
        """
        make your weights and biases some constents.
        ---NOT RECOMMENDED---
        """
        fwl = [[fw for _ in range(self.i)] for _ in range(self.n)]
        fbl = [fb for _ in range(self.n)]
        wl = [[[w for _ in range(self.n)] for _ in range(self.n)] for _ in range(self.l)]
        bl = [[b for _ in range(self.n)] for _ in range(self.l)]
        owl = [[ow for _ in range(self.n)] for _ in range(self.on)]
        obl = [ob for _ in range(self.on)]
        return (fwl, fbl, wl, bl, owl, obl)

    def random(self, fwmin=-1, fwmax=1, fbmin=0, fbmax=1, wmin=-1, wmax=1, bmin=0, bmax=1, owmin=-1, owmax=1, obmin=0, obmax=1, fwfunc=random.uniform, fbfunc=random.uniform, wfunc=random.uniform, bfunc=random.uniform, owfunc=random.uniform, obfunc=random.uniform):
        """
        you can use this to randomly intialize your weights and biases
        xavier or he would be better
        but I'm too lazy
        I may add them later'
        """           
        fwl = [[fwfunc(fwmin, fwmax) for _ in range(self.i)] for _ in range(self.n)]
        fbl = [fbfunc(fbmin, fbmax) for _ in range(self.n)]
        wl = [[[wfunc(wmin, wmax) for _ in range(self.n)] for _ in range(self.n)] for _ in range(self.l)]
        bl = [[bfunc(bmin, bmax) for _ in range(self.n)] for _ in range(self.l)]
        owl = [[owfunc(owmin, owmax) for _ in range(self.n)] for _ in range(self.on)]
        obl = [obfunc(obmin, obmax) for _ in range(self.on)]
        return (fwl, fbl, wl, bl, owl, obl)
        
    def xavier(self):
        """
        xavier is way easier to implement using normal(gaussian) distribution.
        so, I'm going to use that'
        """
        fw = [[random.gauss(0, math.sqrt(2/(self.i + self.n))) for a in range (self.i)] for b in range(self.n)]
        fb = [random.gauss(0, math.sqrt(2/(self.i + self.n))) for a in range(self.n)]
        w = [[[random.gauss(0, math.sqrt(1/self.n)) for a in range(self.n)] for b in range(self.n)] for c in range(self.l)]
        b = [[random.gauss(0, math.sqrt(1/self.n)) for a in range(self.n)] for b in range(self.l)]
        ow = [[random.gauss(0, math.sqrt(2/(self.n + self.on))) for a in range(self.n)] for b in range(self.on)]
        ob = [random.gauss(0,  math.sqrt(2/(self.n + self.on))) for a in range(self.on)]
        return (fw, fb, w, b, ow, ob)
        
    def he(self):
        """
        he is also way easier to implement using normal(gaussian) distribution.
        so, I'm going to use that for he too
        """
        fw = [[random.gauss(0, math.sqrt(2/(self.i))) for a in range (self.i)] for b in range(self.n)]
        fb = [random.gauss(0, math.sqrt(2/(self.i))) for a in range(self.n)]
        w = [[[random.gauss(0, math.sqrt(2/self.n)) for a in range(self.n)] for b in range(self.n)] for c in range(self.l)]
        b = [[random.gauss(0, math.sqrt(2/self.n)) for a in range(self.n)] for b in range(self.l)]
        ow = [[random.gauss(0, math.sqrt(2/(self.n))) for a in range(self.n)] for b in range(self.on)]
        ob = [random.gauss(0,  math.sqrt(2/(self.n))) for a in range(self.on)]
        return (fw, fb, w, b, ow, ob)
        
    def lecun(self):
        """
        it feels like everything is way easier to implement using normal(gaussian) distribution.
        so, I'm going to use that for this too
        """
        fw = [[random.gauss(0, math.sqrt(1/(self.i))) for a in range (self.i)] for b in range(self.n)]
        fb = [random.gauss(0, math.sqrt(1/(self.i))) for a in range(self.n)]
        w = [[[random.gauss(0, math.sqrt(1/self.n)) for a in range(self.n)] for b in range(self.n)] for c in range(self.l)]
        b = [[random.gauss(0, math.sqrt(1/self.n)) for a in range(self.n)] for b in range(self.l)]
        ow = [[random.gauss(0, math.sqrt(1/(self.n))) for a in range(self.n)] for b in range(self.on)]
        ob = [random.gauss(0,  math.sqrt(1/(self.n))) for a in range(self.on)]
        return (fw, fb, w, b, ow, ob)
        
"""
Thanks for using!!!
it's quite basic, hence the name "Basic-Learning"
and if it doesn't work, it's not my fault, I'm just 14.
you can use this neural network framework(if you can call it that) to learn about neural networks, use it to create neural networks(may suck or not work), or make a better neural network framework.
choice is yours.
btw, this framework is going to get a full rework and a new name in july, 2025. So, stay tuned.
with that said, have a good day.
                                        -----Samar Jyoti Pator-----
"""       



