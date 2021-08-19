require 'xlua'
require 'optim'
require 'cunn'
require 'image'
require 'gnuplot'
require'lfs'

local c = require 'trepl.colorize'

opt = lapp[[
    --model                 (default 'split_win19_dep9')             model name
    -g, --gpuid             (default 0)                  gpu id
    --feat_dim              (default 64)

    --data_version          (default 'kitti2015')   
    --data_root             (default '/ais/gobi3/datasets/kitti/scene_flow')
    --perm_fn               (default '')

    --model_param           (default '')    weight file
    --bn_meanstd            (default '')

    --saveDir               (default 'results')   folder for intermediate prediction result
    
    --sub_width             (default 2000)

    --start_id              (default 161)
    --n                     (default 1)

    --disp_range            (default 201)
    --savebin               (default 0)
    
    --postprocess           (default 1)
    --cost_agg              (default 0)
    --cost_agg_2            (default 0)
    --cost_w                (default 5)
    --cost_h                (default 5)

    --sgm                   (default 1)
    --post                  (default 1)
    --nyu_cost_agg_1        (default 1)
    --nyu_cost_agg_2        (default 1)
    --confLoc               (default 0)
    --thres                 (default 23)

    --small                 (default 0)
    --koi_sgm               (default 0)
    --koi_sps               (default 0)
    --unary_scale           (default 0)
]]
print(opt)

d = 201
h = 375
w= 1242
img_h = 375
img_w = 1242
function fromfile(fname)
   local size = io.open(fname):seek('end')
   local x = torch.FloatTensor(torch.FloatStorage(fname, false, size / 4))
   local nan_mask = x:ne(x)
   x[nan_mask] = 1e38
   return x
end

function Split(s, delimiter)
  result = {};
  for match in (s..delimiter):gmatch("(.-)"..delimiter) do
      table.insert(result, match);
  end
  return result;
end

left_cost_url = './results/left_cost/'
right_cost_url = './results/right_cost/'
left_disp_url = './results/left_disp/'
right_disp_url = './results/right_disp/'

for file in lfs.dir(left_cost_url) do
  if string.find(file, ".bin") then
    left_cost_path = file
    split_string = Split(file, "_")
    tail_path = Split(split_string[6], "%.")
    right_cost_path = 'right' .. '_' .. split_string[2] .. '_' .. split_string[3] .. '_' .. split_string[4] .. '_' .. split_string[5] .. '_' .. split_string[6]
    left_disp_path = split_string[1] .. '_' .. split_string[2] .. '_10.png'
    right_disp_path = 'right' .. '_' .. split_string[2]  .. '_10.png'

    img_h, img_w, d = tonumber(split_string[4]), tonumber(split_string[5]), tonumber(tail_path[1])
    file_id = tonumber(split_string[2])
    unary_vol = fromfile(left_cost_url .. left_cost_path):view(img_h, img_w,d)
    right_unary_vol = fromfile(right_cost_url .. right_cost_path):view(img_h, img_w,d)
    local l_img = image.load(left_disp_url .. left_disp_path,1, 'byte'):cuda()
    local r_img = image.load(right_disp_url .. right_disp_path, 1, 'byte'):cuda()
    l_img:add(-l_img:mean()):div(l_img:std())
    r_img:add(-r_img:mean()):div(r_img:std())

    if opt.postprocess == 1 then
      require 'smooth'
      if opt.cost_agg > 0 then
        print('cost agg..')
        local tic = torch.tic()
        cost_vol = unary_vol:permute(3,1,2):clone():cuda()
        local pad_w, pad_h = (opt.cost_w-1)/2, (opt.cost_h-1)/2
        local agg_model = nn.SpatialAveragePooling(opt.cost_w, opt.cost_h, 1, 1, pad_w, pad_h):cuda()
        agg_model:setCountExcludePad()
        for i = 1, opt.cost_agg do
          cost_vol = agg_model:forward(cost_vol):clone()
        end
        paths.mkdir(paths.concat(opt.saveDir, 'cost_img'))
        score,pred = cost_vol:max(1)
        if opt.confLoc == 1 then
            pred[score:lt(opt.thres)] = 256
        end
        pred = pred:view(img_h, img_w) - 1 -- disp range: [0,...,128]
        image.save(string.format('%s/cost_img/%06d_10.png', opt.saveDir, file_id), pred:byte())
      end
      if opt.nyu_cost_agg_1 > 0 then
        print('nyu cost agg..')
        local tic = torch.tic()
        lu = unary_vol:view(1,img_h,img_w,opt.disp_range):permute(1,4,2,3):clone():cuda()
        ru = right_unary_vol:view(1,img_h,img_w,opt.disp_range):permute(1,4,2,3):clone():cuda()
        lu,ru = smooth.nyu.cross_agg(l_img:view(1,1,img_h,img_w), r_img:view(1,1,img_h,img_w), lu, ru, opt.nyu_cost_agg_1)
        print('nyu cost agg tmr.. ' .. torch.toc(tic))
      
        paths.mkdir(paths.concat(opt.saveDir, 'nyu_cost_img'))
        _,pred = lu:max(2)
        pred = pred:view(img_h, img_w) - 1 -- disp range: [0,...,128]
        image.save(string.format('%s/nyu_cost_img/%06d_10.png', opt.saveDir, file_id), pred:byte())
      
        paths.mkdir(paths.concat(opt.saveDir, 'right_nyu_cost_img'))
        _,pred = ru:max(2)
        pred = pred:view(img_h, img_w) - 1 -- disp range: [0,...,128]
        image.save(string.format('%s/right_nyu_cost_img/%06d_10.png', opt.saveDir, file_id), pred:byte())
        print('writing NYU cost image done..')
      else
        lu = cost_vol:view(1, opt.disp_range, img_h, img_w)
        ru = right_cost_vol:view(1, opt.disp_range, img_h, img_w)
      end
      
      if opt.confLoc == 1 then
        lu[lu:lt(opt.thres)] = 0
        ru[ru:lt(opt.thres)] = 0
      end
      
      print('cost mean: ' .. c.cyan(lu:mean()) .. ' cost max: ' .. c.cyan(lu:max()) .. ' std: ' .. c.cyan(lu:std()))
      -- torch.save('debug.t7', lu:float())
      lu = lu / lu:std()
      ru = ru / ru:std()
      
      if opt.sgm == 1 then
        print('nyu sgm..')
        local tic = torch.tic()
        lu:mul(-1)
        ru:mul(-1)
        lu = smooth.nyu.sgm(l_img, r_img, lu, -1)
        ru = smooth.nyu.sgm(l_img, r_img, ru, 1)
        print('nyu sgm tmr.. ' .. torch.toc(tic))
      
        paths.mkdir(paths.concat(opt.saveDir, 'nyu_sgm_img'))
        _,pred = lu:min(2)
        pred = pred:view(img_h, img_w) - 1 -- disp range: [0,...,128]
        image.save(string.format('%s/nyu_sgm_img/%06d_10.png', opt.saveDir, file_id), pred:byte())
      
        paths.mkdir(paths.concat(opt.saveDir, 'right_nyu_sgm_img'))
        _,pred = ru:min(2)
        pred = pred:view(img_h, img_w) - 1 -- disp range: [0,...,128]
        image.save(string.format('%s/right_nyu_sgm_img/%06d_10.png', opt.saveDir, file_id), pred:byte())
        -- lu: 1 x disp x h x w
        print('writing SGM image done..')
      end
      
      if opt.nyu_cost_agg_2 > 0 then
        print('nyu cost agg 2..')
        local tic = torch.tic()
        lu,ru = smooth.nyu.cross_agg(l_img:view(1,1,img_h,img_w), r_img:view(1,1,img_h,img_w), lu, ru, opt.nyu_cost_agg_2)
        print('nyu cost agg tmr.. ' .. torch.toc(tic))
      
        paths.mkdir(paths.concat(opt.saveDir, 'nyu_cost_img_2'))
        _,pred = lu:min(2)
        pred = pred:view(img_h, img_w) - 1 -- disp range: [0,...,128]
        image.save(string.format('%s/nyu_cost_img_2/%06d_10.png', opt.saveDir, file_id), pred:byte())
      
        paths.mkdir(paths.concat(opt.saveDir, 'right_nyu_cost_img_2'))
        _,pred = ru:min(2)
        pred = pred:view(img_h, img_w) - 1 -- disp range: [0,...,128]
        image.save(string.format('%s/right_nyu_cost_img_2/%06d_10.png', opt.saveDir, file_id), pred:byte())
      end
    
      if opt.cost_agg_2 > 0 then
        print('cost agg..')
        local tic = torch.tic()
        lu = lu:view(opt.disp_range, img_h, img_w)
        local pad_w, pad_h = (opt.cost_w-1)/2, (opt.cost_h-1)/2
        local agg_model = nn.SpatialAveragePooling(opt.cost_w, opt.cost_h, 1, 1, pad_w, pad_h):cuda()
        agg_model:setCountExcludePad()
    
        for i = 1, opt.cost_agg_2 do
            lu = agg_model:forward(lu):clone()
        end
    
        ru = ru:view(opt.disp_range, img_h, img_w)
        for i = 1, opt.cost_agg_2 do
            ru = agg_model:forward(ru):clone()
        end
        print('post cost agg tmr.. ' .. torch.toc(tic))
    
        paths.mkdir(paths.concat(opt.saveDir, 'post_cost_img'))
        _,pred = lu:min(1)
        -- if opt.confLoc == 1 then
        --     pred[score:lt(opt.thres)] = 256
        -- end
        pred = pred:view(img_h, img_w) - 1 -- disp range: [0,...,128]
        image.save(string.format('%s/post_cost_img/%06d_10.png', opt.saveDir, file_id), pred:byte())
    
        paths.mkdir(paths.concat(opt.saveDir, 'right_post_cost_img'))
        _,pred = ru:max(1)
        pred = pred:view(img_h, img_w) - 1 -- disp range: [0,...,128]
        image.save(string.format('%s/right_post_cost_img/%06d_10.png', opt.saveDir, file_id), pred:byte())
    
        lu = lu:view(1, opt.disp_range, img_h, img_w)
        ru = ru:view(1, opt.disp_range, img_h, img_w)
      end
      -- more nyu postprocess
      if opt.post == 1 then
        -- lu: 1 x disp x h x w
        disp = {}
        _, pred = lu:min(2)
        disp[1] = (pred - 1):cuda()
        _, pred = ru:min(2)
        disp[2] = (pred - 1):cuda()
    
        print('nyu post..')
        local tic = torch.tic()
    
        final_pred, outlier = smooth.nyu.post(disp, lu)
        print('nyu post tmr.. ' .. torch.toc(tic))
    
        paths.mkdir(paths.concat(opt.saveDir, 'nyu_post'))
        image.save(string.format('%s/nyu_post/%06d_10.png', opt.saveDir, file_id), final_pred:view(img_h, img_w):byte())
    
        paths.mkdir(paths.concat(opt.saveDir, 'outlier'))
        image.save(string.format('%s/outlier/%06d_10.png', opt.saveDir, file_id), (outlier*127):view(img_h, img_w):byte())
        print('writing NYU post image done..')
      end
    
      if opt.koi_sgm == 0 and opt.post == 1 and opt.koi_sps == 1 then
        -- use koi smooth only
        paths.mkdir(opt.saveDir..'/nyu_koi_final')
        local png_fn = string.format('%s/nyu_post/%06d_10.png', opt.saveDir, file_id)
        print('koi sps post..')
        local tic = torch.tic()
        smooth.koi.sps(l_fn, r_fn, png_fn, opt.saveDir..'/nyu_koi_final')
        print('koi sps post tmr.. ' .. torch.toc(tic))
        print('writing SPS post image done..')
      end
    end

  end
end

