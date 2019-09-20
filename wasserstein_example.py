import tensorflow as tf

import numpy as np
from scipy.special import lambertw

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def C_matrix(n,p=2, kernel_size=3, local_transport= False):
    N=n**2
    if local_transport:
        if kernel_size % 2 != 1:
            raise ValueError("Need odd kernel size")

        center = kernel_size // 2
        C = np.zeros([kernel_size,kernel_size])
        for i in range(kernel_size):
            for j in range(kernel_size):
                C[i,j] = (abs(i-center)**2 + abs(j-center)**2)**(p/2)
        return tf.constant(C)
    else:
        C = np.zeros([N,N])
        for i in range(N):
            for j in range(N):
                ii,ij=divmod (i,n)
                ji,jj=divmod (j,n)
                C[i,j]=(abs(ii-ji)**2 + abs(ij-jj)**2)**(p/2)
        return tf.constant(C)

def tensor_to_array(X):
    return tf.compat.v1.Session().run(X)


def plot_tensor(X,d=28):
    plt.imshow(np.resize(tensor_to_array(X),[d,d]))
    plt.axis('off')
    plt.show()



def wasserstein_example(X,lam=100,epsilon=3,p=2,verbose=0,deltalimit=1e-15):
    x= tf.convert_to_tensor(tensor_to_array(X))
    [d1,d2]=x.shape

    n=int(np.sqrt(d1.value))
    N=d1.value
    assert (d2==1)&(N==n**2) , 'input error'
    #x=np.random.random((N,1))
    #x=tf.convert_to_tensor(x/np.sum(x))
    #plot_tensor(x)
    w=np.random.random((N,1))
    w=tf.convert_to_tensor(w/np.sum(w))
    C=C_matrix(n)
    temp=tf.constant(np.ones([N,1])*np.log(1/N))
    alpha=temp;
    beta=temp;

    #psi=tf.convert_to_tensor(1,dtype=tf.float64)
    psi=1.
    u=tf.exp(alpha)
    v=tf.exp(beta)

    calpha=alpha
    cbeta=beta
    cpsi=psi
    iterate_time=0

    converge_flag=False
    while not converge_flag:

        K= tf.exp(-psi*C-1)

        alpha=tf.math.log(x)-tf.math.log(tf.matmul(K,v))
        u=tf.exp(alpha)
        #plot_tensor(alpha)

        temp_fw_array=tensor_to_array(tf.multiply(tf.transpose(tf.matmul(tf.transpose(u),K)),lam* tf.exp(lam*w)))
        beta=lam*w-tf.convert_to_tensor(np.real(lambertw(temp_fw_array)))
        v=tf.exp(beta)

        a=1.0

        g= -epsilon + tf.matmul(tf.matmul(tf.transpose(u), tf.multiply(C,K)),v)
        h= - tf.matmul(tf.matmul(tf.transpose(u),tf.multiply(tf.multiply(C,C),K)),v)
        b=tensor_to_array(tf.divide(g, h))

        while psi-a*b<0:
            a=a/2
            #print('haha')

            #print (a,psi-a*b)
        psi=psi-a*b


        delta=tf.reduce_max(tf.abs(calpha-alpha))+tf.reduce_max(tf.abs(cbeta-beta))+np.abs(cpsi-psi)

        delta=tensor_to_array(delta)
        print (delta)
        print (delta<deltalimit)
        if delta<deltalimit:
            converge_flag=True
        calpha=alpha
        #print(delta)
        cbeta=beta
        cpsi=psi
        iterate_time=iterate_time+1

        if verbose>0:
            if np.mod(iterate_time,verbose)==0:
                plot_tensor(w-(1./lam)*beta)
                print (iterate_time,delta)
    return w-(1./lam)*beta
