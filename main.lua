--[[
Copyright (c) 2016-present, Facebook, Inc.
All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.
]]--

-- load torchnet:
require 'paths'
tnt = require 'torchnet'

local flag_mnist = false 
local use_gpu = true

-- function that sets of dataset iterator:
local function getIterator_with_filename_without_threads(mode)
   local function findClasses(dir)
      local dirs = paths.dir(dir)
      table.sort(dirs)

      local classList = {}
      local classToIdx = {}
      for _ ,class in ipairs(dirs) do
         if not classToIdx[class] and class ~= '.' and class ~= '..' then
            table.insert(classList, class)
            classToIdx[class] = #classList
         end
      end

      -- assert(#classList == 1000, 'expected 1000 ImageNet classes')
      return classList, classToIdx
   end

   local function findImages(dir, classToIdx)
      ----------------------------------------------------------------------
      -- Options for the GNU and BSD find command
      local extensionList = {'jpg', 'png','JPG','PNG','JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
      local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
      for i=2,#extensionList do
         findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
      end

      -- Find all the images using the find command
      local f = io.popen('find -L ' .. dir .. findOptions)

      local maxLength = -1
      local imagePaths = {}
      local imageClasses = {}

      -- Generate a list of all the images and their class
      while true do
         local line = f:read('*line')
         if not line then break end

         local className = paths.basename(paths.dirname(line))
         local filename = paths.basename(line)
         local path = className .. '/' .. filename

         local classId = classToIdx[className]
         assert(classId, 'class not found: ' .. className)

         table.insert(imagePaths, path)
         table.insert(imageClasses, classId)

         maxLength = math.max(maxLength, #path + 1)
      end

      f:close()

      -- Convert the generated list to a tensor for faster loading
      local nImages = #imagePaths

      return imagePaths, imageClasses
   end

   local trainDir = paths.concat('/mnt/disk1/Data/Etc/ImageNet/for_facebook_resnet', 'train')
   local valDir = paths.concat('/mnt/disk1/Data/Etc/ImageNet/for_facebook_resnet', 'val')
   assert(paths.dirp(trainDir), 'train directory not found: ' .. trainDir)
   assert(paths.dirp(valDir), 'val directory not found: ' .. valDir)
   print(trainDir)
   print(valDir)

   print("=> Generating list of images")
   local classList, classToIdx = findClasses(trainDir)

   print(" | finding all validation images")
   local valImagePaths, valImageClasses = findImages(valDir, classToIdx)

   print(" | finding all training images")
   local trainImagePaths, trainImageClasses = findImages(trainDir, classToIdx)

   local info = {
      basedir = '.',
      classList = classList,
      train = {
         filename = trainImagePaths,
         label = trainImageClasses,
      },
      val = {
         filename = valImagePaths,
         label = valImageClasses,
      },
   }

   local cacheFile = './list_of_images.t7'
   print(" | saving list of images to " .. cacheFile)
   torch.save(cacheFile, info)



   info = torch.load(cacheFile)
   local data = {} 
   label = {} 
   mode = 'train'
   if mode == 'train' then
      for i=1,10 do --#info.train.filename do
         --table.insert(data, info.train.filename[i])
         table.insert(data, info.train.filename[i])
         table.insert(label, info.train.label[i])
      end
   else
      for i=1,10 do --#info.val.filename do
         --table.insert(data, info.val.filename[i])
         table.insert(data, info.val.filename[i])
         table.insert(label, info.val.label[i])
      end
   end
   print('1. type of list input: ' .. type(data))


   return tnt.ParallelDatasetIterator{
      nthread = 1,
      init = function(threadid)
         tnt = require 'torchnet'
         require 'image'
      end,
      closure = function()
         classes = {}
         for i=1,5 do
            print('type of list input: ' .. type(data) .. ', ' .. #data)
            local list = tnt.ListDataset{
               list = data,
               load = function(im)
                  return {
                     input = image.load(paths.concat('/mnt/disk1/Data/Etc/ImageNet/for_facebook_resnet/train',im)):float(),
                     target = torch.LongTensor{1},
                  }
               end,
            }
            classes[#classes+1] = mode == 'train' and list:shuffle(2,true) or list:shuffle() -- class balancing
         end
         return tnt.ConcatDataset{datasets = classes}:shuffle():batch(3,'skip-last')
      end,
   }
end


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
               1, dataset.data:size(2), dataset.data:size(3)):double()
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
            dataset.data = dataset.data:reshape(size, 3, 32, 32)
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
net = nn.Sequential()

local function ConvBNReLU(nInputPlane, nOutputPlane)
  net:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  net:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  net:add(nn.ReLU(true))
  return net 
end


if flag_mnist then
   net = nn.Sequential()
   --net:add(nn.MulConstant(0.00390625))
   net:add(nn.SpatialConvolution(1,20,5,5,1,1,0,0)) -- 1*28*28 -> 20*24*24
   net:add(nn.SpatialMaxPooling(2,2,2,2)) -- 20*24*24 -> 20*12*12
   net:add(nn.SpatialConvolution(20,50,5,5,1,1,0,0)) -- 20*12*12 -> 50*8*8
   net:add(nn.SpatialMaxPooling(2,2,2,2)) --  50*8*8 -> 50*4*4
   --net:add(nn.SpatialConvolution(1,20,5,5,1,1,0,0)) -- channels*28*28 -> 20*24*24
   --net:add(nn.Linear(20*24*24,10))
   --net = nn.Sequential()
   --net:add(nn.MulConstant(0.00390625))
   --net:add(nn.SpatialConvolution(channels,20,5,5,1,1,0)) -- channels*28*28 -> 20*24*24
   --net:add(nn.SpatialMaxPooling(2, 2, 2, 2)) -- 20*24*24 -> 20*12*12
   --net:add(nn.SpatialConvolution(20,50,5,5,1,1,0)) -- 20*12*12 -> 50*8*8
   --net:add(nn.SpatialMaxPooling(2,2,2,2)) --  50*8*8 -> 50*4*4
   --net:add(nn.View(-1):setNumInputDims(3))  -- 50*4*4 -> 800
   --net:add(nn.Linear(800,500))  -- 800 -> 500
   --net:add(nn.ReLU())
   --net:add(nn.Linear(500, 10))  -- 500 -> nclasses
   --net:add(nn.LogSoftMax())
   net:add(nn.View(-1):setNumInputDims(3))  -- 50*4*4 -> 800
   net:add(nn.Linear(50*4*4, 10))  -- 500 -> nclasses
   net:add(nn.LogSoftMax())
else
   local MaxPooling = nn.SpatialMaxPooling
   ConvBNReLU(3,64) -- 32*32 -> 30*30
   --ConvBNReLU(64,64)
   net:add(MaxPooling(2,2,2,2))
   ConvBNReLU(64,128)
   --ConvBNReLU(128,128)
   net:add(MaxPooling(2,2,2,2))
   ConvBNReLU(128,256)
   --ConvBNReLU(256,256)
   --ConvBNReLU(256,256)
   net:add(MaxPooling(2,2,2,2))
   ConvBNReLU(256,512)
   --ConvBNReLU(512,512)
   --ConvBNReLU(512,512)
   net:add(MaxPooling(2,2,2,2))
   --ConvBNReLU(512,512)
   --ConvBNReLU(512,512)
   --ConvBNReLU(512,512)
   --net:add(MaxPooling(2,2,2,2))
   --net:add(nn.View(512))
   net:add(nn.View(-1):setNumInputDims(3))
   --net:add(nn.Linear(512,512))
   --net:add(nn.ReLU(true))
   net:add(nn.Linear(512*(32/2/2/2/2)*(32/2/2/2/2),10)) 
   -- initialization from MSR
   local function MSRinit(net)
     local function init(name)
       for k,v in pairs(net:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         v.bias:zero()
       end
     end
     -- have to do for both backends
     --init'nn.SpatialConvolution'
   end
   MSRinit(net)
end


local function ConvInit(name)
      for k,v in pairs(net:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         v.bias = nil
         v.gradBias = nil
      end
end


ConvInit('nn.SpatialConvolution')



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
         iterator = getIterator_with_filename_without_threads('test'),
      }
      print(string.format('%d:%2.2f [epoch:validation error]', epoch,clerr:value{k = 1}))

      file = io.open('trend.log', 'a+')
      file:write(string.format('%2.2f\n', clerr:value{k = 1}))
      file:close()

      local modelFile = 'net_' .. epoch .. '.t7'
      local stateFile = 'state_' .. epoch .. '.t7'

      if epoch % 1 == 0 then
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
   iterator  = getIterator_with_filename_without_threads('train'),
   criterion = crit,
   lr        = 0.2,
   maxepoch  = 10,
}

print(string.format('test loss: %2.4f; test error: %2.4f',
   meter:value(), clerr:value{k = 1}))
