smooth.nyu = {}

smooth.nyu.L1 = 5 -- cost window size
smooth.nyu.tau1 = 0.03 -- pixel intensity
-- smooth.nyu.cbca_i1 = 2
-- smooth.nyu.cbca_i2 = 4
smooth.nyu.sgm_i = 1 -- sgm iter??

smooth.nyu.pi1 = 2.3 -- sgm p1
smooth.nyu.pi2 = 24.25 -- sgm p2
smooth.nyu.tau_so = 0.08  -- pixel intensity threshold for changing pi1/pi2
smooth.nyu.alpha1 = 1.75 --?? haha
smooth.nyu.sgm_q1 = 3 -- factor 1 used for changing pi1/pi2
smooth.nyu.sgm_q2 = 2 -- factor 2 used for changing pi1/pi2

-- post
smooth.nyu.occlusion = 1
smooth.nyu.subpixel = 1
smooth.nyu.median = 1
smooth.nyu.bilateral = 1
smooth.nyu.blur_sigma = 5.99
smooth.nyu.blur_t = 5

function smooth.nyu.cross_agg( l_img, r_img, left_vol, right_vol, iter)
	-- left_vol: 1 x disp_range x h x w
	-- l_img: 1 x 3 x h x w
	local img_h, img_w = left_vol:size(3), left_vol:size(4)
	local disp_range = left_vol:size(2)
    -- cross computation
    local x0c, x1c
    x0c = torch.CudaTensor(1, 4, img_h, img_w)
    x1c = torch.CudaTensor(1, 4, img_h, img_w)
    adcensus.cross(l_img, x0c, smooth.nyu.L1, smooth.nyu.tau1)
    adcensus.cross(r_img, x1c, smooth.nyu.L1, smooth.nyu.tau1)
    local tmp_cbca = torch.CudaTensor(1, disp_range, img_h, img_w)
    for i = 1,iter do
        adcensus.cbca(x0c, x1c, left_vol, tmp_cbca, -1)
        left_vol:copy(tmp_cbca)
        adcensus.cbca(x0c, x1c, right_vol, tmp_cbca, 1)
        right_vol:copy(tmp_cbca)
    end
    tmp_cbca = nil
    collectgarbage()
    return left_vol, right_vol
end

function smooth.nyu.sgm( l_img, r_img, vol, direction)
	-- direction: -1 for left, 1 for right
	-- l_img: 3 x h x w
	-- vol: 1 x disp x h x w
   	local img_h, img_w = l_img:size(2), l_img:size(3)
    local vol = vol:permute(1,3,4,2):clone()
    local max_disp = vol:size(4)

	local out = torch.CudaTensor(1, img_h, img_w, max_disp)
	local tmp = torch.CudaTensor(img_w, max_disp)
	for _ = 1,smooth.nyu.sgm_i do
	   out:zero()
	   adcensus.sgm2(l_img, r_img, vol, out, tmp, smooth.nyu.pi1, smooth.nyu.pi2, smooth.nyu.tau_so,
	      smooth.nyu.alpha1, smooth.nyu.sgm_q1, smooth.nyu.sgm_q2, direction)
	   vol:copy(out):div(4)
	end
	vol:resize(1, max_disp, img_h, img_w)
	vol:copy(out:permute(1,4,2,3)):div(4) -- vol: 1 x max_disp x h x w

	--            local out = torch.CudaTensor(4, vol:size(2), vol:size(3), vol:size(4))
	--            out:zero()
	--            adcensus.sgm3(x_batch[1], x_batch[2], vol, out, smooth.nyu.pi1, smooth.nyu.pi2, smooth.nyu.tau_so,
	--               smooth.nyu.alpha1, smooth.nyu.sgm_q1, smooth.nyu.sgm_q2, direction)
	--            vol:mean(out, 1)
	--            vol = vol:transpose(3, 4):transpose(2, 3):clone()
	collectgarbage()

	return vol
end

local function gaussian(sigma)
   local kr = math.ceil(sigma * 3)
   local ks = kr * 2 + 1
   local k = torch.Tensor(ks, ks)
   for i = 1, ks do
      for j = 1, ks do
         local y = (i - 1) - kr
         local x = (j - 1) - kr
         k[{i,j}] = math.exp(-(x * x + y * y) / (2 * sigma * sigma))
      end
   end
   return k
end

function smooth.nyu.post( disp, left_vol)
	-- disp[1]: left prediction, disp: 0 .. max-1
	local max_disp = left_vol:size(2)
	local outlier = torch.CudaTensor():resizeAs(disp[1]):zero()
	adcensus.outlier_detection(disp[1], disp[2], outlier, max_disp)

	if smooth.nyu.occlusion == 1 then
		disp[1] = adcensus.interpolate_occlusion(disp[1], outlier)
		disp[1] = adcensus.interpolate_mismatch(disp[1], outlier)
	end

	if smooth.nyu.subpixel == 1 then
		disp[1] = adcensus.subpixel_enchancement(disp[1], left_vol, max_disp)
	end

	if smooth.nyu.median == 1 then
		disp[1] = adcensus.median2d(disp[1], 5)
	end

	if smooth.nyu.bilateral == 1 then
		disp[1] = adcensus.mean2d(disp[1], gaussian(smooth.nyu.blur_sigma):cuda(), smooth.nyu.blur_t)
	end

	return disp[1], outlier
end
