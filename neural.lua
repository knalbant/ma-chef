require 'torch'
require 'nn'
require 'csvigo'
require 'models.lua'
require 'utils.lua'

results_dir = "./results"

hidden_layers = 2
hidden_nodes = 40
learning_rate = 0.1
activation = nn.Sigmoid()

-- Load data
features = torch.Tensor(csvigo.load{path="features.csv", header=false, mode="raw"})
labels = torch.Tensor(csvigo.load{path="labels.csv", header=false, mode="raw"})

training, testing = trainTestSplit(features, labels, .7)

net, criterion = buildmodel(features:size(2),
                            torch.max(labels),
                            hidden_layers,
                            hidden_nodes,
                            activation)
trainer = nn.StochasticGradient(net, criterion)
trainingError, testingError, meanAccuracies = {}, {}, {}
trainer.testing = testing
trainer.learningRate = 0.1

trainer.hookIteration = function(self, iteration)
    trainingError[iteration] = self.criterion.output
    testingError[iteration] = calculateError(self.testing.data, self.testing.label, self.module, self.criterion) 
    meanAccuracies[iteration] = calculateAccuracy(self.testing.data, self.testing.label, self.module, self.criterion)
end

trainer:train(training)

writeTable(results_dir .. "training_error_hidden_" .. hidden_layers .. "_" .. hidden_nodes .. "_" .. tostring(activation) .. "_" .. learning_rate, trainingError)
writeTable(results_dir .. "testing_error_hidden_" .. hidden_layers .. "_" .. hidden_nodes .. "_" .. tostring(activation) .. "_" .. learning_rate, testingError)
writeTable(results_dir .. "mean_accuracies_" .. hidden_layers .. "_" .. hidden_nodes .. "_" .. tostring(activation) .. "_" .. learning_rate, meanAccuracies)
