# <codecell>
#%%
import numpy as np
import theano
import theano.tensor as T

from collections import OrderedDict

import matplotlib.pyplot as plt
%matplotlib inline

# <markdowncell>
# # Introduction to Theano
#
# Theano is a library for doing symbolic computaion. To understand this a comparison
# with numpy as probably best...
#
# Lets create to matrices and calculate their product..

# <codecell>
#%%
x = np.random.rand(2,3)
y = np.random.rand(3,5)

z = np.dot(x,y)
print(z)

# <markdowncell>
# So what happened there? We made two matrices and in the process space was allocated
# and filled with some random numbers. We then calculated their product, some more
# space was allocated for the resulut and then the actual computaion took place!
#
# Lets do the same but using Theano...

# <codecell>
#%%
x = T.matrix()
y = T.matrix()

z = T.dot(x,y)
print(z)

# <markdowncell>
# It looks similar but is in fact very different. No space was allocated no numbers
# exist within the matrices, no computation took place. The two variables x,y are just
# boxes which at some point in the future can contain data - the size of the matrices has
# not yet been specified. The result z will take the contents of the boxes and at some
# later point calculate the result, usually on a GPU. Theano is lazy...
#
# How to actualy do the computation? We use a theano function...

# <codecell>
#%%
f = theano.function([x,y],z)

# <markdowncell>
# The first argument is a list of input variables - the boxes we created. The second
# argument is the output, in this case z. To use this function we need to specify some
# data...

# <codecell>
#%%
x_ = np.random.rand(2,3).astype('float32')
y_ = np.random.rand(3,6).astype('float32')

# <markdowncell>
# Its really important that we convert to 32 bit float since Theano is used on GPU's
# and most do not have support for 64 bit arithmetic...
#
# We can now use this function...we should get a matrix of size 2x6

# <codecell>
#%%
print(f(x_,y_))

# <markdowncell>
# This function will work with matrices of any size as long as the dimensions
# agree (the last dim of x should be the same as first dim of y)

# <codecell>
#%%
x_ = np.random.rand(200,300).astype('float32')
y_ = np.random.rand(300,600).astype('float32')

print(f(x_,y_))

# <codecell>
#%%
x_ = np.random.rand(200,30).astype('float32')
y_ = np.random.rand(300,600).astype('float32')

print(f(x_,y_))

# <markdowncell>
# We get an error due to dimension mismatch...

# <markdowncell>
# ## Scalars, Vectors and Matrices
# Theano can manipulate various data types. Lets do some more complex things...

# <codecell>
#%%
a = T.scalar()
a_ = 2
b = T.scalar()
b_ = 56

u = T.vector()
u_ = np.random.rand(5).astype('float32')
v = T.vector()
v_ = np.ones(5).astype('float32')

A = T.matrix()
A_= np.eye(5).astype('float32')
B = T.matrix()
B_ = np.random.rand(5,5).astype('float32')

f1 = a + b
f2 = T.sin(a)

f3 = T.dot(u,v)
f4 = a*v

f5 = A + B
f6 = b*A
f7 = T.dot(B,u)

f1_ = theano.function([a,b],f1)
f2_ = theano.function([a],f2)
f3_ = theano.function([u,v],f3)
f4_ = theano.function([a,v],f4)
f5_ = theano.function([A,B],f5)
f6_ = theano.function([b,A],f6)
f7_ = theano.function([B,u],f7)

print(f1_(a_,b_))
print(f2_(a_))
print(f3_(u_,v_))
print(f4_(a_,v_))
print(f5_(A_,B_))
print(f6_(b_,A_))
print(f7_(B_,u_))

# <markdowncell>
# We can form large chains of computation with various intermediate steps...

# <codecell>
#%%
a = T.scalar()
A = T.matrix()
B = T.matrix()

Z = a*A - a*B
Y = T.dot(A,-B)

W = Z + Y

res = theano.function([a,A,B],W)

a_ = 10
A_ = np.random.rand(5,5).astype('float32')
B_ = np.random.rand(5,5).astype('float32')

print(res(a_,A_,B_))

# <markdowncell>
# ## Derivatives
# Theano does derivatives for us, no need for algebra!
#
# Lets find the derivative of the following;
# $$ y = sin(x) $$
# $$ \frac{dy}{dx} = cos(x) $$

# <codecell>
#%%
x = T.scalar()
y = T.sin(x)
y_ = theano.function([x],y)

dydx = T.grad(y,wrt=x)
dydx_ = theano.function([x],dydx)

# <markdowncell>
# Lets plot the results...

# <codecell>
#%%

xs = np.linspace(0,2*np.pi).astype('float32')

ys = np.array([float(y_(k)) for k in xs])
dydxs = np.array([float(dydx_(k)) for k in xs])

plt.plot(xs,ys,label='ys')
plt.plot(xs,dydxs,label='dydxs')
plt.legend(loc=0)

# <markdowncell>
# We can do much more complicated derivatives...
# $$ F(A,\mathbf{x},\mathbf{y}) = \frac{1}{2} ||\mathbf{y} - A\mathbf{x}||^2 = \frac{1}{2}(\mathbf{y} - A\mathbf{x})^T (\mathbf{y}  - A\mathbf{x}) $$
# $$ \frac{\partial F}{\partial \mathbf{x}} = (A\mathbf{x} - \mathbf{y})^T A $$

# <codecell>
#%%
A = T.matrix()
x = T.vector()
y = T.vector()

r = y - T.dot(A,x)
f = 0.5*T.dot(r,r)

dfdx = T.grad(f,wrt=x)
dfdx_ = theano.function([A,x,y],dfdx)

A_ = np.random.rand(5,5).astype('float32')
x_ = np.random.rand(5).astype('float32')
y_ = np.random.rand(5).astype('float32')

print(dfdx_(A_,x_,y_))
print(np.dot(np.dot(A_,x_) - y_,A_))


# <markdowncell>
# Theano makes updating variables, the weights in the NN, really easy...But we need
# to understand shared variables first...These are variable that we need to initialise
# with data and which we are able to change...

# <codecell>
#%%
x = theano.shared(5)
print(x.get_value())

x.set_value(24)
print(x.get_value())

x = theano.shared(np.random.rand(2,3).astype('float32'))
print(x.get_value())

x.set_value(np.eye(4).astype('float32'))
print(x.get_value())

# <markdowncell>
# We can update variable in the following way...

# <codecell>
#%%
x = theano.shared(0.0)

updates = OrderedDict()
updates[x] = x + 0.1

f = theano.function([],updates=updates)

print(x.get_value())
f()
print(x.get_value())
f()
print(x.get_value())

# <markdowncell>
# So each time we call the function f the updates are applied to the shared variable.
# We can update multiple variables at the same time...we can evan have a parameter that
# controls the update - so it's not always the same update!

# <codecell>
#%%
x = theano.shared(10.0)
y = theano.shared(23.0)
a  = T.scalar()

updates = OrderedDict()
updates[x] = x + a
updates[y] = y*a

f = theano.function([a],updates=updates)

f(2)
print(x.get_value())
print(y.get_value())

f(-1)
print(x.get_value())
print(y.get_value())

# <markdowncell>
# ## Putting it all together..
# We are going to use theano to solve the following problem;
# $$ \underset{x,y}{\text{minimize}} \; x^2 + y^2 + xy -2x $$
# We are looking for the values of $x,y$ that makes $x^2 + y^2 + xy - 2x$ as small as
# possible...The general method is very similar to what we will do when teaching a neural
# network. Find the gradient at a particuar position and then change our initial guess by moving
# in the opposite direction...

# <codecell>
#%%

# initial guess at solution
np.random.seed(26988)
x0 = 100*np.random.rand()
y0 = 100*np.random.rand()
x = theano.shared(x0)
y = theano.shared(y0)

# the function we want to minimise
f = x**2 + y**2 + x*y - 2*x

# gradients
dfdx = T.grad(f,wrt=x)
dfdy = T.grad(f,wrt=y)

#updates
l_rate = 0.1
updates = OrderedDict()
updates[x] = x - l_rate*dfdx
updates[y] = y - l_rate*dfdy

update_step = theano.function([],updates=updates)
f_val = theano.function([],f)


# <markdowncell>
# The model is all set up now to solve the optimisation problem, all that remains is
# to repeatedly call our update function until we reach the minimum. According to wolframalpha
# the solution should be $x = \frac{4}{3}, y = -\frac{2}{3}$. Usually we would stop calling the update
# function once the gradient gets small (when dfdx and dfdy are small) but for now we will just do
# 100 iterations...

# <codecell>
#%%
#%%
for i in range(100):
    update_step()
    if i % 10 == 0:
        print('iter {}:\n @ x={},y={}\n f_val = {}\n'.format(i,
                                                           x.get_value(),
                                                           y.get_value(),
                                                           f_val()))
