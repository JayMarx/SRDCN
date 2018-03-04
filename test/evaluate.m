addpath('./evaluation_func/');
ori_img1 = imread('./pic/1.png');
sr_img1 = imread('./pic/1_x4.png');

ori_ycbcr = ori_img1;
sr_ycbcr = sr_img1;

if size(ori_img1,3)>1
    ori_ycbcr = rgb2ycbcr(ori_img1);
    sr_ycbcr = rgb2ycbcr(sr_img1);
end

ori_y = ori_ycbcr(:,:,1);
sr_y = sr_ycbcr(:,:,1);

%ori_y_gnd = modcrop(ori_y, 4);

%im_y_gnd1 = shave(ori_y_gnd, [4, 4]);

psnr = compute_psnr(ori_y,sr_y)
ssim = ssim_index(ori_y,sr_y)