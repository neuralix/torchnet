tnt = require 'torchnet'

mnist = require 'mnist'

traindataset = mnist['traindataset']()
traindataset.data = traindataset.data:reshape(traindataset.data:size(1),
            traindataset.data:size(2) * traindataset.data:size(3)):double()

testdataset = mnist['testdataset']()
testdataset.data = testdataset.data:reshape(testdataset.data:size(1),
            testdataset.data:size(2) * testdataset.data:size(3)):double()

function getTrainIterator()
   return tnt.DatasetIterator{ 
      dataset = tnt.ListDataset{
         list = torch.range(1, traindataset.data:size(1)):long(),            
         load = function(idx)
            return {
               input  = traindataset.data,
               target = traindataset.label + 1,
            }
         end,
      }
   }
end

function getTestIterator()
   return tnt.DatasetIterator{ 
      dataset = tnt.ListDataset{
         list = torch.range(1, testdataset.data:size(1)):long(),            
         load = function(idx)
            return {
               input  = testdataset.data,
               target = testdataset.label + 1,
            }
         end,
      }
   }
end

net = nn.Sequential():add(nn.Linear(784, 10))
crit = nn.CrossEntropyCriterion()
engine = tnt.SGDEngine()
meter = tnt.AverageValueMeter()

engine.hooks.onForwardCriterion = function(state)
   meter:add(state.criterion.output)
   if state.training then
      print(string.format('%2.2f', meter:value()))
   end
end


engine:train{
   network = net,
   criterion = crit,
   iterator = getTrainIterator(),
   lr = 0.2,
   maxepoch = 1,
}

engine:test{
   network = net,
   criterion = crit,
   iterator = getTestIterator(),
}
