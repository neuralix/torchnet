--[[
Copyright (c) 2016-present, Facebook, Inc.
All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.
]]--

-- load torchnet:
local tnt = require 'torchnet'

local flag_mnist = false
local use_gpu = true

-- function that sets of dataset iterator:
local function getIterator(mode)
   return tnt.ParallelDatasetIterator{
      nthread = 1,
      init    = function() require 'torchnet' end,
      closure = function()

         local data = nil
         if flag_mnist then
            -- load MNIST dataset:
            local mnist = require 'mnist'
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
               dataset.label = torch.Tensor(size)
               for i = 0,4 do
                  subset = torch.load('cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')
                  dataset.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
                  dataset.label[{ {i*10000+1, (i+1)*10000} }] = subset.labels -- CAUTION: labels and label
               end
               dataset.label = dataset.label + 1
            else
               size = 2000
               dataset = {
                  data = torch.Tensor(size, 3*32*32),
                  label = torch.Tensor(size)
               }
               subset = torch.load('cifar-10-batches-t7/test_batch.t7', 'ascii')
               dataset.data = subset.data:t():double()
               dataset.label = subset.labels[1]:double() -- CAUTION: labels and label
               dataset.label = dataset.label + 1
            end
            -- resize dataset (if using small version)
            dataset.data = dataset.data[{ {1,size} }]
            dataset.label = dataset.label[{ {1,size} }]
         end
   

         -- return batches of data:
         return tnt.BatchDataset{
            batchsize = 128,
            dataset = tnt.ListDataset{  -- replace this by your own dataset
               list = torch.range(1, dataset.data:size(1)):long(),
               load = function(idx)
                  return {
                     input  = dataset.data[idx],
                     target = torch.LongTensor{dataset.label[idx] + 1},
                  }  -- sample contains input and target
               end,
            }
         }
      end,
   }
end

require 'cunn'
-- set up logistic regressor:
local net = nil
if flag_mnist then
   net = nn.Sequential():add(nn.Linear(784,10))
else
   net = nn.Sequential():add(nn.Linear(3*32*32,10))
end
local criterion = nn.CrossEntropyCriterion()

local dpt = nn.DataParallelTable(1)
--dpt:add(net, {1,2,3,4})
dpt:add(net, {1})
net = dpt:cuda()
crit = criterion:cuda()

-- set up training engine:
local engine = tnt.SGDEngine()
local meter  = tnt.AverageValueMeter()
local clerr  = tnt.ClassErrorMeter{topk = {1}}

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
      --print(string.format('avg. loss: %2.4f; avg. error: %2.4f',
      --   meter:value(), clerr:value{k = 1}))
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

      local modelFile = 'net_' .. epoch .. '.t7'
      local stateFile = 'state_' .. epoch .. '.t7'

      if epoch % 100 == 0 then
         torch.save(modelFile, state.network)
      end
   end
end

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

-- set up GPU training:
if usegpu then

   -- copy model to GPU:
   net       = net
   criterion = crit

   -- copy sample to GPU buffer:
   local igpu, tgpu = torch.CudaTensor(), torch.CudaTensor()
   engine.hooks.onSample = function(state)
      igpu:resize(state.sample.input:size() ):copy(state.sample.input)
      tgpu:resize(state.sample.target:size()):copy(state.sample.target)
      state.sample.input  = igpu
      state.sample.target = tgpu
   end  -- alternatively, this logic can be implemented via a TransformDataset
end

-- train the model:
engine:train{
   network   = net,
   iterator  = getIterator('train'),
   criterion = crit,
   lr        = 0.2,
   maxepoch  = 2,
}

print(string.format('test loss: %2.4f; test error: %2.4f',
   meter:value(), clerr:value{k = 1}))
