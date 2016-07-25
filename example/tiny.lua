local tnt = require 'torchnet'

local function getIterator(mode)
   return tnt.ParallelDatasetIterator{
      nthread = 1,
      init = function() require 'torchnet' end,
      closure = function()
         mnist = require 'mnist'
         dataset = mnist[mode .. 'dataset']()
         dataset.data = dataset.data:reshape(dataset.data:size(1),
            dataset.data:size(2) * dataset.data:size(3)):double()

         return tnt.BatchDataset{ 
            batchsize = 1000,
            dataset = tnt.ListDataset{
               list = torch.range(1, dataset.data:size(1)):long(),            
               load = function(idx)
                  return {
                     input  = dataset.data[idx],
                     target = torch.LongTensor{dataset.label[idx] + 1},
                  }
               end,
            }
         }
      end,
   }
end

local net = nn.Sequential():add(nn.Linear(784, 10))
local crit = nn.CrossEntropyCriterion()
local engine = tnt.SGDEngine()
local meter = tnt.AverageValueMeter()
local clerr = tnt.ClassErrorMeter({topk={1}})

engine.hooks.onStartEpoch = function(state)
   meter:reset()
   clerr:reset()
end
  
iter = 0
epoch = 0
engine.hooks.onForwardCriterion = function(state)
   iter = iter + 1
   meter:add(state.criterion.output)
   clerr:add(state.network.output, state.sample.target)
   if state.training then
      dummy = 1
      print(string.format('%d:%2.2f [iter:loss]', iter, meter:value()))
   end
end

engine.hooks.onEndEpoch = function(state)
   epoch = epoch + 1
   meter:add(state.criterion.output)
   clerr:add(state.network.output, state.sample.target)
   if state.training then
      engine:test{
         network = net,
         criterion = crit,
         iterator = getIterator('test'),
      }
      print(string.format('%d:%2.2f [epoch:validation error]\n', epoch,clerr:value{k = 1}))
   end
end

require 'cunn'
net = net:cuda()
crit = crit:cuda()
engine.hooks.onSample = function(state)
   state.sample.input = torch.CudaTensor():
      resize(state.sample.input:size()):
      copy(state.sample.input)
   state.sample.target = torch.CudaTensor():
      resize(state.sample.target:size()):
      copy(state.sample.target)
end

engine:train{
   network = net,
   criterion = crit,
   iterator = getIterator('train'),
   lr = 0.2,
   maxepoch = 3,
}

engine:test{
   network = net,
   criterion = crit,
   iterator = getIterator('test'),
}
