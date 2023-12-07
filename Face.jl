
D:
cd Julia-1.7.2\bin

set JULIA_NUM_THREADS=4
SET JULIA_DEPOT_PATH=D:\JULIA_ASSETS\JULIA_PACKAGE_DIRECTORY

julia



using DataFrames
using Images
using ImageView
using StableRNGs
using Random
using MLJ

path = "C:\\Users\\user\\Desktop\\face\\UTKFace"
images = []
age = []
gender = []

for img in readdir(path)

push!(images,channelview(float32.(Gray.(Images.load(string(path,"\\",img))))))
push!(age,split(img, "_")[1])
push!(gender,split(img, "_")[2])

end


B_up=images
images=B_up
images=[imresize(var, (200, 200)) for var in images]
y=gender

perm = randperm(length(y))
rng=StableRNG(perm[1])
train,test=partition(eachindex(y),0.75, shuffle=true ,rng=rng)


xtrain=images[train]
xtest=images[test]
ytrain=y[train]
ytest=y[test]

xtrain = reduce(hcat, xtrain);
xtest = reduce(hcat, xtest);
xtrain = reshape(xtrain, 200, 200, 1,:);
xtest = reshape(xtest, 200, 200,1, :);


using SimpleChains
lenet = SimpleChain(
  (static(200), static(200), static(1)),
  SimpleChains.Conv(SimpleChains.relu, (5, 5), 6),
  SimpleChains.MaxPool(2, 2),
  SimpleChains.Conv(SimpleChains.relu, (5, 5), 16),
  SimpleChains.MaxPool(2, 2),
  Flatten(3),
  TurboDense(SimpleChains.relu, 120),
  TurboDense(SimpleChains.relu, 84),
  TurboDense(identity, 2),
)






@time p = SimpleChains.init_params(lenet);

@time lenet(xtrain, p)

ytrain= parse.(UInt32, ytrain)
ytest= parse.(UInt32, ytest)

lenetloss = SimpleChains.add_loss(lenet, LogitCrossEntropyLoss(ytrain));

g = similar(p);
@time valgrad!(g, lenetloss, xtrain, p)

G = SimpleChains.alloc_threaded_grad(lenetloss);

#Training Starts

@time SimpleChains.train_batched!(G, p, lenetloss, xtrain, SimpleChains.ADAM(3e-4), 10);

SimpleChains.accuracy_and_loss(lenetloss, xtrain, p),
SimpleChains.accuracy_and_loss(lenetloss, xtest, ytest, p)

# SimpleChains.init_params!(lenet, p);
@time SimpleChains.train_batched!(G, p, lenetloss, xtrain, SimpleChains.ADAM(3e-4), 10);
SimpleChains.accuracy_and_loss(lenetloss, xtrain, p),
SimpleChains.accuracy_and_loss(lenetloss, xtest, ytest, p)



lenet.memory .= 0;
SimpleChains.init_params!(lenet, p);
@time SimpleChains.train_batched!(G, p, lenetloss, xtrain, SimpleChains.ADAM(3e-4), 10);
SimpleChains.accuracy_and_loss(lenetloss, xtrain, p),
SimpleChains.accuracy_and_loss(lenetloss, xtest, ytest, p)
SimpleChains.init_params!(lenet, p);
@time SimpleChains.train_batched!(G, p, lenetloss, xtrain, SimpleChains.ADAM(3e-4), 10);
SimpleChains.accuracy_and_loss(lenetloss, xtrain, p),
SimpleChains.accuracy_and_loss(lenetloss, xtest, ytest, p)



g0 = similar(g);
g1 = similar(g);
lenetloss.memory .= 0xff;
@time valgrad!(g0, lenetloss, xtrain, p)
lenetloss.memory .= 0x00;
@time valgrad!(g1, lenetloss, xtrain, p)
g0 == g1
lenet.memory .= 0;
SimpleChains.init_params!(lenet, p);
@time SimpleChains.train_batched!(G, p, lenetloss, xtrain, SimpleChains.ADAM(3e-4), 10);
SimpleChains.accuracy_and_loss(lenetloss, xtrain, p),
SimpleChains.accuracy_and_loss(lenetloss, xtest, ytest, p)
SimpleChains.init_params!(lenet, p);
@time SimpleChains.train_batched!(G, p, lenetloss, xtrain, SimpleChains.ADAM(3e-4), 10);
SimpleChains.accuracy_and_loss(lenetloss, xtrain, p),
SimpleChains.accuracy_and_loss(lenetloss, xtest, ytest, p)
