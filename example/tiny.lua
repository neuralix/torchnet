local tnt = require 'torchnet'

flag_mnist = false 

local function getIterator(mode)
   return tnt.ParallelDatasetIterator{
      nthread = 1,
      init = function() require 'torchnet' end,
      closure = function()
         if flag_mnist then
            mnist = require 'mnist'
            dataset = mnist[mode .. 'dataset']()
            dataset.data = dataset.data:reshape(dataset.data:size(1),
               dataset.data:size(2) * dataset.data:size(3)):double()
         else
            if not paths.dirp('cifar-10-batches-t7') then
               print '==> downloading dataset'
               tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar10.t7.tgz'
               os.execute('wget ' .. tar)
               os.execute('tar xvf ' .. paths.basename(tar))
            end
            if mode == 'train' then
               size = 50000
               dataset = {
                  data = torch.Tensor(size, 3*32*32),
                  labels = torch.Tensor(size)
               }
               dataset.data = torch.Tensor(size, 3*32*32)
               dataset.labels = torch.Tensor(size)
               for i = 0,4 do
                  subset = torch.load('cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')
                  dataset.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
                  dataset.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
               end
               dataset.labels = dataset.labels + 1
            else
               size = 2000
               dataset = {
                  data = torch.Tensor(size, 3*32*32),
                  labels = torch.Tensor(size)
               }
               subset = torch.load('cifar-10-batches-t7/test_batch.t7', 'ascii')
               dataset.data = subset.data:t():double()
               dataset.labels = subset.labels[1]:double()
               dataset.labels = dataset.labels + 1
            end
            -- resize dataset (if using small version)
            dataset.data = dataset.data[{ {1,size} }]
            dataset.labels = dataset.labels[{ {1,size} }]
         end

         return tnt.BatchDataset{ 
            batchsize = 1000,
            dataset = tnt.ListDataset{
               list = torch.range(1, dataset.data:size(1)):long(),            
               load = function(idx)
                  return {
                     input  = dataset.data[idx],
                     target = torch.LongTensor{dataset.labels[idx] + 1},
                  }
               end,
            }
         }
      end,
   }
end

local net = nil
if flag_mnist then
   net = nn.Sequential():add(nn.Linear(784, 10))
else
   channels = 3
   net = nn.Sequential()
   net:add(nn.Reshape(3*32*32))
   net:add(nn.Linear(3*32*32, 2048))
   net:add(nn.Tanh())
   net:add(nn.Linear(2048,10))
end
local crit = nn.CrossEntropyCriterion()
local engine = tnt.SGDEngine()
local meter = tnt.AverageValueMeter()
local clerr = tnt.ClassErrorMeter({topk={1}})

iter = 0
old_iter = 0
epoch = 0

engine.hooks.onStartEpoch = function(state)
   iter = old_iter
   meter:reset()
   clerr:reset()
end
  
engine.hooks.onForwardCriterion = function(state)
   iter = iter + 1
   meter:add(state.criterion.output)
   clerr:add(state.network.output, state.sample.target)
   if state.training then
      dummy = 1
      --print(string.format('%d:%2.2f [iter:loss]', iter, meter:value()))
   end
end

engine.hooks.onEndEpoch = function(state)
   old_iter = iter
   epoch = epoch + 1
   meter:add(state.criterion.output)
   clerr:add(state.network.output, state.sample.target)
   if state.training then
      engine:test{
         network = net,
         criterion = crit,
         iterator = getIterator('test'),
      }
      print(string.format('%d:%2.2f [epoch:validation error]', epoch,clerr:value{k = 1}))

      file = io.open('trend.log', 'a+')
      file:write(string.format('%2.2f\n', clerr:value{k = 1}))
      file:close()
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

file = io.open('trend.log', 'a+')
file:write(string.format('last\n'))
file:close()

engine:train{
   network = net,
   criterion = crit,
   iterator = getIterator('train'),
   lr = 0.01,
   maxepoch = 100000,
}

