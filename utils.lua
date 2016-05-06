require 'nn'
require 'math'


function trainTestSplit(data,labels, trainpercent)
  local size      = data:size()[1] 
  local trainSize = math.ceil(size * trainpercent)
  local data, labels = permute(data,labels)
  local trainset = makeDataTable(data[{ {1,trainSize}, {} }],labels[ { {1,trainSize}, {} }])
  local testset  = makeDataTable(data[ { {trainSize + 1, size}, {} } ],labels[ { {trainSize + 1, size}, {} } ])

  trainset = makeTrainAble(trainset)

  return trainset, testset
end 

function makeTrainAble(trainset)
   
  --sets an indexing method for the trainset for use with stochastic gradient descent
  setmetatable(trainset, {__index = function(t,i) return {t.data[i], t.label[i]} end }) 

  --also for SGD, returns size of the dataset 
  function trainset:size()
    return self.data:size(1)
  end 

  return trainset
  
end 

function makeDataTable(data,labels)

  local datatable = {}
  datatable.data   = data
  datatable.label = labels

  return datatable

end 


function permute(data, labels)
  local shuffled = torch.randperm(data:size()[1]):long() --make the permutation vector 
  return data:index(1,shuffled), labels:index(1,shuffled) --return the permuted matrices 
end 

--convenience function which convert a vector to a string
function tensorToString(tensor)
  local size = tensor:size(1)
  local str = ''

  for i =1, size do
    str = str .. tostring(tensor[i]) .. ' '
  end 
  return str
end 

--convenience function to write table to file
function writeTable(filename, table)

  local f = io.open(filename, 'w')
   
  for _, s in ipairs(table) do
    f:write(tostring(s) .. '\n')
  end 
  f:close()

end 

--convenience function to write the weight and output arrays to a file 
function writeTableTensor(filename, tensorTable)
  local f = io.open(filename, 'w')
  
  for _,t in ipairs(tensorTable) do
    s = tensorToString(t)
    f:write(s .. '\n')
  end 

  f:close()
end 

function calculateAccuracy(data, labels, net, loss)
 
  local class_performance = {}
  local class_accuracies  = {}

  for i=1,torch.max(labels) do
    class_performance[i] = 0
    class_accuracies[i]  = 0
  end 


  for i=1,data:size(1) do
    local groundtruth = labels[i]
    local prediction  = net:forward(data[i])
    local confidences, indices = torch.sort(prediction, true) --sort in descending order

    if groundtruth == indices[1] then
      class_performance[groundtruth] = class_performance[groundtruth] + 1
    end
  end 

  
  for i =1,#class_accuracies do
    class_accuracies[i] = 100 * class_performance[i] / torch.sum(labels:eq(i))
  end 
  print(class_performance)

  return torch.sum(torch.Tensor(class_accuracies)) / #class_accuracies, class_accuracies, class_performance

end 



function calculateError(data, labels, net, loss)

  local size = data:size(1)
  local err  = 0 



  for i = 1,size do
    err = err + loss:forward(net:forward(data[i]), labels[i])
  end 

  return err / size
end
