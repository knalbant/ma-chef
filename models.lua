require 'torch'
require 'nn'

-- Inputs : Number of features
-- Output : Number of classes
-- Layers : Number of hidden layers
-- Hidden : Number of nodes in each hidden layer
-- Activator : Activation function
-- Drouput : Perform dropout or not
function buildmodel(inputs, output, layers, hidden, activator, dropout)
    dropout = dropout or 0

    net = nn.Sequential()
    net:add(nn.Linear(inputs, hidden))
    if dropout ~= 0 then
        net:add(nn.Dropout())
    end
    net:add(activator)

    for i=1,layers do
        net:add(nn.Linear(hidden, hidden))
        if dropout ~= 0 then
            net:add(nn.Dropout())
        end
        net:add(activator)
    end

    net:add(nn.Linear(hidden,output))
    net:add(nn.LogSoftMax())
    return net, nn.ClassNLLCriterion()
end
