clear all;
clc;

%???????????.
tic;


%---------%only information here needs to be provided by user.----------%
noFiles = 100;
nTimesteps_train = 70;
nTimesteps_test = noFiles - nTimesteps_train;   %30.
firstTimesteps = nTimesteps_train;
fileStartVal = 1;
fileIncrement = 1;
constVal = 1;
xDim = 168;
yDim = 224;
zDim = 160;
singleDataType = 'single';
name_epoch1 = 1199;     %3499; %6999; %349;   
name_epoch2 = 2799;     %6999; %13999;    %699; 
name_epoch3 = 4399;     %10499;    %20999;  
name_epoch4 = 7199;     %13999;    %27999;    %1049; 
name_epoch5 = 9199;     %17499;    %34999;    %1399; 
name_epoch6 = 11999;    %20999;    %41999;    %1749; 
name_epoch7 = 12799;    %24499;    %48999;    %2099; 
name_epoch8 = 14399;    %27999;    %55999;    %2449; 
name_epoch9 = 15999;    %31499;    %62999;    %2799; 
name_epoch10 = 17599;   %34999;   %69999;    %3149;
name_epoch11 = 19199;   %38499;   %76999;   %3499;    
name_epoch12 = 20399;   %41999;   %3849;   
name_epoch13 = 21999;   %45499;   %4199;    
name_epoch14 = 23199;   %48999;   %4549;   
name_epoch15 = 24799;   %52499;  %4899;    
name_epoch16 = 25199;   %55999;  %5249;    
name_epoch17 = 26399;   %59499;  %5599;    
name_epoch18 = 27199;   %62999;  %5949;    
name_epoch19 = 15199;   %66499;  %6299;   
name_epoch20 = 27999;   %69999;  %6649;

name_epoch1 = 3499;     %1399;
name_epoch2 = 6999;     %2799;
name_epoch3 = 10499;    %4199;
name_epoch4 = 13999;    %5599;
name_epoch5 = 17499;    %6999;
name_epoch6 = 20999;    %8399;
name_epoch7 = 24499;    %9799;
name_epoch8 = 27999;    %11199;
name_epoch9 = 31499;    %12599;
name_epoch10 = 34999;   %13999;
name_epoch11 = 38499;   %15399;
name_epoch12 = 41999;   %16799;
name_epoch13 = 45499;   %18199;
name_epoch14 = 48999;   %19599;
name_epoch15 = 52499;   %20999;
name_epoch16 = 55999;   %22399;
name_epoch17 = 59499;   %23799;
name_epoch18 = 62999;   %25199;
name_epoch19 = 66499;   %26599;
name_epoch20 = 69999;   %27999;


dataSourcePath_GT = 'D: pythonprojects\Multimodalsyn\ctmri_brain\';
dataSourcePath_epoch = 'C:\pythonprojects\MultimodalSyn\ctmri\brain\crop=0.5\v9.WGAN_InNorm_enContrast_1MoreTB_finalPick.py\epochs=1000\';

dataSourcePath_GT = 'C:\Files\#IT_lessons\Visualiation\MRI2CT\DCGAN\dataSet1\';
dataSourcePath_epoch = 'C:\Files\#IT_lessons\Visualiation\MRI2CT\DCGAN\testResults\trial07\';

timestepAxis = firstTimesteps:noFiles-1;    %timestepAxis: [70, 99].
batchSize = 1;
numOfBatches = ceil(nTimesteps_train / batchSize);  %70.
%---------%only information here needs to be provided by user.----------%


for t = firstTimesteps:noFiles-1    %t: [71,100].
    %at each timestep t:
    %1. read gt mr.
    file = sprintf('%snorm_mr_enContrast.%.3d.raw', dataSourcePath_GT, (fileStartVal + t * fileIncrement) / constVal);
    fid = fopen(file, 'r');
    oneColumnMatrix = fread(fid, xDim * yDim * zDim, singleDataType);
    fclose(fid);
    GT = reshape(oneColumnMatrix, [xDim, yDim, zDim]);  %[width, height, depth]. The order here is very important.
    clear oneColumnMatrix;
    
    
    %2. read predict results from different epochs.
    %(2.1)predict result from epoch1.
    file = sprintf('%sDCGAN_%d\\DCGAN_pred_norm_mr_%.4d.raw', dataSourcePath_epoch, name_epoch1, (fileStartVal + t * fileIncrement) / constVal);
    fid = fopen(file, 'r');
    oneColumnMatrix = fread(fid, xDim * yDim * zDim, singleDataType);
    fclose(fid);
    epoch1 = reshape(oneColumnMatrix, [xDim, yDim, zDim]);  %[width, height, depth]. The order here is very important.
    clear oneColumnMatrix;
    
    
    %(2.2)predict from epoch2.
    file = sprintf('%sDCGAN_%d\\DCGAN_pred_norm_mr_%.4d.raw', dataSourcePath_epoch, name_epoch2, (fileStartVal + t * fileIncrement) / constVal);
    fid = fopen(file, 'r');
    oneColumnMatrix = fread(fid, xDim * yDim * zDim, singleDataType);
    fclose(fid);
    epoch2 = reshape(oneColumnMatrix, [xDim, yDim, zDim]);  %[width, height, depth]. The order here is very important.
    clear oneColumnMatrix;
    
    
    %(2.3)predict from epcoch3.
    file = sprintf('%sDCGAN_%d\\DCGAN_pred_norm_mr_%.4d.raw', dataSourcePath_epoch, name_epoch3, (fileStartVal + t * fileIncrement) / constVal);
    fid = fopen(file, 'r');
    oneColumnMatrix = fread(fid, xDim * yDim * zDim, singleDataType);
    fclose(fid);
    epoch3 = reshape(oneColumnMatrix, [xDim, yDim, zDim]);  %[width, height, depth]. The order here is very important.
    clear oneColumnMatrix;
    
    
    %(2.4)predict from epoch4.
    file = sprintf('%sDCGAN_%d\\DCGAN_pred_norm_mr_%.4d.raw', dataSourcePath_epoch, name_epoch4, (fileStartVal + t * fileIncrement) / constVal);
    fid = fopen(file, 'r');
    oneColumnMatrix = fread(fid, xDim * yDim * zDim, singleDataType);
    fclose(fid);
    epoch4 = reshape(oneColumnMatrix, [xDim, yDim, zDim]);  %[width, height, depth]. The order here is very important.
    clear oneColumnMatrix;
    
    
    %(2.5)predict from epoch5.
    file = sprintf('%sDCGAN_%d\\DCGAN_pred_norm_mr_%.4d.raw', dataSourcePath_epoch, name_epoch5, (fileStartVal + t * fileIncrement) / constVal);
    fid = fopen(file, 'r');
    oneColumnMatrix = fread(fid, xDim * yDim * zDim, singleDataType);
    fclose(fid);
    epoch5 = reshape(oneColumnMatrix, [xDim, yDim, zDim]);  %[width, height, depth]. The order here is very important.
    clear oneColumnMatrix;
    
    
    %(2.6)predict from epoch6.
    file = sprintf('%sDCGAN_%d\\DCGAN_pred_norm_mr_%.4d.raw', dataSourcePath_epoch, name_epoch6, (fileStartVal + t * fileIncrement) / constVal);
    fid = fopen(file, 'r');
    oneColumnMatrix = fread(fid, xDim * yDim * zDim, singleDataType);
    fclose(fid);
    epoch6 = reshape(oneColumnMatrix, [xDim, yDim, zDim]);  %[width, height, depth]. The order here is very important.
    clear oneColumnMatrix;
    
    
    %(2.7)predict from epoch7.
    file = sprintf('%sDCGAN_%d\\DCGAN_pred_norm_mr_%.4d.raw', dataSourcePath_epoch, name_epoch7, (fileStartVal + t * fileIncrement) / constVal);
    fid = fopen(file, 'r');
    oneColumnMatrix = fread(fid, xDim * yDim * zDim, singleDataType);
    fclose(fid);
    epoch7 = reshape(oneColumnMatrix, [xDim, yDim, zDim]);  %[width, height, depth]. The order here is very important.
    clear oneColumnMatrix;
    
    
    %(2.8)predict from epoch8.
    file = sprintf('%sDCGAN_%d\\DCGAN_pred_norm_mr_%.4d.raw', dataSourcePath_epoch, name_epoch8, (fileStartVal + t * fileIncrement) / constVal);
    fid = fopen(file, 'r');
    oneColumnMatrix = fread(fid, xDim * yDim * zDim, singleDataType);
    fclose(fid);
    epoch8 = reshape(oneColumnMatrix, [xDim, yDim, zDim]);  %[width, height, depth]. The order here is very important.
    clear oneColumnMatrix;
    
    
    %(2.9)predict from epoch9.
    file = sprintf('%sDCGAN_%d\\DCGAN_pred_norm_mr_%.4d.raw', dataSourcePath_epoch, name_epoch9, (fileStartVal + t * fileIncrement) / constVal);
    fid = fopen(file, 'r');
    oneColumnMatrix = fread(fid, xDim * yDim * zDim, singleDataType);
    fclose(fid);
    epoch9 = reshape(oneColumnMatrix, [xDim, yDim, zDim]);  %[width, height, depth]. The order here is very important.
    clear oneColumnMatrix;
    
    
    %(2.10)predict from epoch10.
    file = sprintf('%sDCGAN_%d\\DCGAN_pred_norm_mr_%.4d.raw', dataSourcePath_epoch, name_epoch10, (fileStartVal + t * fileIncrement) / constVal);
    fid = fopen(file, 'r');
    oneColumnMatrix = fread(fid, xDim * yDim * zDim, singleDataType);
    fclose(fid);
    epoch10 = reshape(oneColumnMatrix, [xDim, yDim, zDim]);  %[width, height, depth]. The order here is very important.
    clear oneColumnMatrix;
    
    
    %(2.11)predict from epoch11.
    file = sprintf('%sDCGAN_%d\\DCGAN_pred_norm_mr_%.4d.raw', dataSourcePath_epoch, name_epoch11, (fileStartVal + t * fileIncrement) / constVal);
    fid = fopen(file, 'r');
    oneColumnMatrix = fread(fid, xDim * yDim * zDim, singleDataType);
    fclose(fid);
    epoch11 = reshape(oneColumnMatrix, [xDim, yDim, zDim]);  %[width, height, depth]. The order here is very important.
    clear oneColumnMatrix;
    
   
    %(2.12)predict from epoch12.
    file = sprintf('%sDCGAN_%d\\DCGAN_pred_norm_mr_%.4d.raw', dataSourcePath_epoch, name_epoch12, (fileStartVal + t * fileIncrement) / constVal);
    fid = fopen(file, 'r');
    oneColumnMatrix = fread(fid, xDim * yDim * zDim, singleDataType);
    fclose(fid);
    epoch12 = reshape(oneColumnMatrix, [xDim, yDim, zDim]);  %[width, height, depth]. The order here is very important.
    clear oneColumnMatrix;
    
    
    %(2.13)predict from epoch13.
    file = sprintf('%sDCGAN_%d\\DCGAN_pred_norm_mr_%.4d.raw', dataSourcePath_epoch, name_epoch13, (fileStartVal + t * fileIncrement) / constVal);
    fid = fopen(file, 'r');
    oneColumnMatrix = fread(fid, xDim * yDim * zDim, singleDataType);
    fclose(fid);
    epoch13 = reshape(oneColumnMatrix, [xDim, yDim, zDim]);  %[width, height, depth]. The order here is very important.
    clear oneColumnMatrix;
    
    
    %(2.14)predict from epoch14.
    file = sprintf('%sDCGAN_%d\\DCGAN_pred_norm_mr_%.4d.raw', dataSourcePath_epoch, name_epoch14, (fileStartVal + t * fileIncrement) / constVal);
    fid = fopen(file, 'r');
    oneColumnMatrix = fread(fid, xDim * yDim * zDim, singleDataType);
    fclose(fid);
    epoch14 = reshape(oneColumnMatrix, [xDim, yDim, zDim]);  %[width, height, depth]. The order here is very important.
    clear oneColumnMatrix;
    
    
    %(2.15)predict from epoch15.
    file = sprintf('%sDCGAN_%d\\DCGAN_pred_norm_mr_%.4d.raw', dataSourcePath_epoch, name_epoch15, (fileStartVal + t * fileIncrement) / constVal);
    fid = fopen(file, 'r');
    oneColumnMatrix = fread(fid, xDim * yDim * zDim, singleDataType);
    fclose(fid);
    epoch15 = reshape(oneColumnMatrix, [xDim, yDim, zDim]);  %[width, height, depth]. The order here is very important.
    clear oneColumnMatrix;
    
    
    %(2.16)predict from epoch16.
    file = sprintf('%sDCGAN_%d\\DCGAN_pred_norm_mr_%.4d.raw', dataSourcePath_epoch, name_epoch16, (fileStartVal + t * fileIncrement) / constVal);
    fid = fopen(file, 'r');
    oneColumnMatrix = fread(fid, xDim * yDim * zDim, singleDataType);
    fclose(fid);
    epoch16 = reshape(oneColumnMatrix, [xDim, yDim, zDim]);  %[width, height, depth]. The order here is very important.
    clear oneColumnMatrix;
    
    
    %(2.17)predict from epoch17.
    file = sprintf('%sDCGAN_%d\\DCGAN_pred_norm_mr_%.4d.raw', dataSourcePath_epoch, name_epoch17, (fileStartVal + t * fileIncrement) / constVal);
    fid = fopen(file, 'r');
    oneColumnMatrix = fread(fid, xDim * yDim * zDim, singleDataType);
    fclose(fid);
    epoch17 = reshape(oneColumnMatrix, [xDim, yDim, zDim]);  %[width, height, depth]. The order here is very important.
    clear oneColumnMatrix;
    
    
    %(2.18)predict from epoch18.
    file = sprintf('%sDCGAN_%d\\DCGAN_pred_norm_mr_%.4d.raw', dataSourcePath_epoch, name_epoch18, (fileStartVal + t * fileIncrement) / constVal);
    fid = fopen(file, 'r');
    oneColumnMatrix = fread(fid, xDim * yDim * zDim, singleDataType);
    fclose(fid);
    epoch18 = reshape(oneColumnMatrix, [xDim, yDim, zDim]);  %[width, height, depth]. The order here is very important.
    clear oneColumnMatrix;
    
   
    %(2.19)predict from epoch19.
    file = sprintf('%sDCGAN_%d\\DCGAN_pred_norm_mr_%.4d.raw', dataSourcePath_epoch, name_epoch19, (fileStartVal + t * fileIncrement) / constVal);
    fid = fopen(file, 'r');
    oneColumnMatrix = fread(fid, xDim * yDim * zDim, singleDataType);
    fclose(fid);
    epoch19 = reshape(oneColumnMatrix, [xDim, yDim, zDim]);  %[width, height, depth]. The order here is very important.
    clear oneColumnMatrix;
    
    
    %(2.20)predict from epoch20.
    file = sprintf('%sDCGAN_%d\\DCGAN_pred_norm_mr_%.4d.raw', dataSourcePath_epoch, name_epoch20, (fileStartVal + t * fileIncrement) / constVal);
    fid = fopen(file, 'r');
    oneColumnMatrix = fread(fid, xDim * yDim * zDim, singleDataType);
    fclose(fid);
    epoch20 = reshape(oneColumnMatrix, [xDim, yDim, zDim]);  %[width, height, depth]. The order here is very important.
    clear oneColumnMatrix;
    
    
    
    %3. at t, compute metrics.
    %(3.1)compute PSNR, in reference to GT e.g., peaksnr = psnr(A,ref).
    psnr_epoch1(t-nTimesteps_train+1) = psnr(epoch1, GT);
    psnr_epoch2(t-nTimesteps_train+1) = psnr(epoch2, GT);
    psnr_epoch3(t-nTimesteps_train+1) = psnr(epoch3, GT);
    psnr_epoch4(t-nTimesteps_train+1) = psnr(epoch4, GT);
    psnr_epoch5(t-nTimesteps_train+1) = psnr(epoch5, GT);
    psnr_epoch6(t-nTimesteps_train+1) = psnr(epoch6, GT);
    psnr_epoch7(t-nTimesteps_train+1) = psnr(epoch7, GT);
    psnr_epoch8(t-nTimesteps_train+1) = psnr(epoch8, GT);
    psnr_epoch9(t-nTimesteps_train+1) = psnr(epoch9, GT);
    psnr_epoch10(t-nTimesteps_train+1) = psnr(epoch10, GT);
    psnr_epoch11(t-nTimesteps_train+1) = psnr(epoch11, GT);
    psnr_epoch12(t-nTimesteps_train+1) = psnr(epoch12, GT);
    psnr_epoch13(t-nTimesteps_train+1) = psnr(epoch13, GT);
    psnr_epoch14(t-nTimesteps_train+1) = psnr(epoch14, GT);
    psnr_epoch15(t-nTimesteps_train+1) = psnr(epoch15, GT);
    psnr_epoch16(t-nTimesteps_train+1) = psnr(epoch16, GT);
    psnr_epoch17(t-nTimesteps_train+1) = psnr(epoch17, GT);
    psnr_epoch18(t-nTimesteps_train+1) = psnr(epoch18, GT);
    psnr_epoch19(t-nTimesteps_train+1) = psnr(epoch19, GT);
    psnr_epoch20(t-nTimesteps_train+1) = psnr(epoch20, GT);
    
    
    
    
    %??3.2??compute SSIM, in reference to GT e.g., ssimval = ssim(A, ref).
    %(note: A/ref must be grayscale image or volume).
    ssim_epoch1(t-nTimesteps_train+1) = ssim(epoch1, GT);
    ssim_epoch2(t-nTimesteps_train+1) = ssim(epoch2, GT);
    ssim_epoch3(t-nTimesteps_train+1) = ssim(epoch3, GT);
    ssim_epoch4(t-nTimesteps_train+1) = ssim(epoch4, GT);
    ssim_epoch5(t-nTimesteps_train+1) = ssim(epoch5, GT);
    ssim_epoch6(t-nTimesteps_train+1) = ssim(epoch6, GT);
    ssim_epoch7(t-nTimesteps_train+1) = ssim(epoch7, GT);
    ssim_epoch8(t-nTimesteps_train+1) = ssim(epoch8, GT);
    ssim_epoch9(t-nTimesteps_train+1) = ssim(epoch9, GT);
    ssim_epoch10(t-nTimesteps_train+1) = ssim(epoch10, GT);
    ssim_epoch11(t-nTimesteps_train+1) = ssim(epoch11, GT);
    ssim_epoch12(t-nTimesteps_train+1) = ssim(epoch12, GT);
    ssim_epoch13(t-nTimesteps_train+1) = ssim(epoch13, GT);
    ssim_epoch14(t-nTimesteps_train+1) = ssim(epoch14, GT);
    ssim_epoch15(t-nTimesteps_train+1) = ssim(epoch15, GT);
    ssim_epoch16(t-nTimesteps_train+1) = ssim(epoch16, GT);
    ssim_epoch17(t-nTimesteps_train+1) = ssim(epoch17, GT);
    ssim_epoch18(t-nTimesteps_train+1) = ssim(epoch18, GT);
    ssim_epoch19(t-nTimesteps_train+1) = ssim(epoch19, GT);
    ssim_epoch20(t-nTimesteps_train+1) = ssim(epoch20, GT);
    
    
    
    
    %(3.3) compute MSE(mean-squared error) e.g., err = immse(X,Y).
    immse_epoch1(t-nTimesteps_train+1) = immse(epoch1, GT);
    immse_epoch2(t-nTimesteps_train+1) = immse(epoch2, GT);
    immse_epoch3(t-nTimesteps_train+1) = immse(epoch3, GT);
    immse_epoch4(t-nTimesteps_train+1) = immse(epoch4, GT);
    immse_epoch5(t-nTimesteps_train+1) = immse(epoch5, GT);
    immse_epoch6(t-nTimesteps_train+1) = immse(epoch6, GT);
    immse_epoch7(t-nTimesteps_train+1) = immse(epoch7, GT);
    immse_epoch8(t-nTimesteps_train+1) = immse(epoch8, GT);
    immse_epoch9(t-nTimesteps_train+1) = immse(epoch9, GT);
    immse_epoch10(t-nTimesteps_train+1) = immse(epoch10, GT);
    immse_epoch11(t-nTimesteps_train+1) = immse(epoch11, GT);
    immse_epoch12(t-nTimesteps_train+1) = immse(epoch12, GT);
    immse_epoch13(t-nTimesteps_train+1) = immse(epoch13, GT);
    immse_epoch14(t-nTimesteps_train+1) = immse(epoch14, GT);
    immse_epoch15(t-nTimesteps_train+1) = immse(epoch15, GT);
    immse_epoch16(t-nTimesteps_train+1) = immse(epoch16, GT);
    immse_epoch17(t-nTimesteps_train+1) = immse(epoch17, GT);
    immse_epoch18(t-nTimesteps_train+1) = immse(epoch18, GT);
    immse_epoch19(t-nTimesteps_train+1) = immse(epoch19, GT);
    immse_epoch20(t-nTimesteps_train+1) = immse(epoch20, GT);
    
    
    %{
    %add on 2023.7.10.
    %(3.4)compute Feature Similarity Index (FSIM), e.g., [FSIM, FSIMc] =
    %FeatureSIM(img1, img2).
    [FSIM_epoch1(t-nTimesteps_train+1), FSIMc] = FeatureSIM(epoch1, GT);
    [FSIM_epoch2(t-nTimesteps_train+1), FSIMc] = FeatureSIM(epoch2, GT);
    [FSIM_epoch3(t-nTimesteps_train+1), FSIMc] = FeatureSIM(epoch3, GT);
    [FSIM_epoch4(t-nTimesteps_train+1), FSIMc] = FeatureSIM(epoch4, GT);
    [FSIM_epoch5(t-nTimesteps_train+1), FSIMc] = FeatureSIM(epoch5, GT);
    [FSIM_epoch6(t-nTimesteps_train+1), FSIMc] = FeatureSIM(epoch6, GT);
    [FSIM_epoch7(t-nTimesteps_train+1), FSIMc] = FeatureSIM(epoch7, GT);
    [FSIM_epoch8(t-nTimesteps_train+1), FSIMc] = FeatureSIM(epoch8, GT);
    [FSIM_epoch9(t-nTimesteps_train+1), FSIMc] = FeatureSIM(epoch9, GT);
    [FSIM_epoch10(t-nTimesteps_train+1), FSIMc] = FeatureSIM(epoch10, GT);
    [FSIM_epoch11(t-nTimesteps_train+1), FSIMc] = FeatureSIM(epoch11, GT);
    [FSIM_epoch12(t-nTimesteps_train+1), FSIMc] = FeatureSIM(epoch12, GT);
    [FSIM_epoch13(t-nTimesteps_train+1), FSIMc] = FeatureSIM(epoch13, GT);
    [FSIM_epoch14(t-nTimesteps_train+1), FSIMc] = FeatureSIM(epoch14, GT);
    [FSIM_epoch15(t-nTimesteps_train+1), FSIMc] = FeatureSIM(epoch15, GT);
    [FSIM_epoch16(t-nTimesteps_train+1), FSIMc] = FeatureSIM(epoch16, GT);
    [FSIM_epoch17(t-nTimesteps_train+1), FSIMc] = FeatureSIM(epoch17, GT);
    [FSIM_epoch18(t-nTimesteps_train+1), FSIMc] = FeatureSIM(epoch18, GT);
    [FSIM_epoch19(t-nTimesteps_train+1), FSIMc] = FeatureSIM(epoch19, GT);
    [FSIM_epoch20(t-nTimesteps_train+1), FSIMc] = FeatureSIM(epoch20, GT);
    %add on 2023.7.10.
    %}
    
    
end


%4. plot metrics vs. timesteps.
%(4.1)plot PSNR.
figure(1);
plot(timestepAxis, psnr_epoch1, 'DisplayName', strcat('# of epochs=', num2str((name_epoch1+1)/numOfBatches)));

hold on;
plot(timestepAxis, psnr_epoch2, 'o--r', 'DisplayName', strcat('# of epochs=', num2str((name_epoch2+1)/numOfBatches)));

hold on;
plot(timestepAxis, psnr_epoch3, 'o--g', 'DisplayName', strcat('# of epochs=', num2str((name_epoch3+1)/numOfBatches)));

hold on;
plot(timestepAxis, psnr_epoch4, 'o--cyan', 'DisplayName', strcat('# of epochs=', num2str((name_epoch4+1)/numOfBatches)));

hold on;
plot(timestepAxis, psnr_epoch5, 'o--b', 'DisplayName', strcat('# of epochs=', num2str((name_epoch5+1)/numOfBatches)));

hold on;
plot(timestepAxis, psnr_epoch6, 'DisplayName', strcat('# of epochs=', num2str((name_epoch6+1)/numOfBatches)));

hold on;
plot(timestepAxis, psnr_epoch7, 'DisplayName', strcat('# of epochs=', num2str((name_epoch7+1)/numOfBatches)));

hold on;
plot(timestepAxis, psnr_epoch8, 'DisplayName', strcat('# of epochs=', num2str((name_epoch8+1)/numOfBatches)));

hold on;
plot(timestepAxis, psnr_epoch9, 'DisplayName', strcat('# of epochs=', num2str((name_epoch9+1)/numOfBatches)));

hold on;
plot(timestepAxis, psnr_epoch10, 'DisplayName', strcat('# of epochs=', num2str((name_epoch10+1)/numOfBatches)));

hold on;
plot(timestepAxis, psnr_epoch11, 'DisplayName', strcat('# of epochs=', num2str((name_epoch11+1)/numOfBatches)));

hold on;
plot(timestepAxis, psnr_epoch12, 'DisplayName', strcat('# of epochs=', num2str((name_epoch12+1)/numOfBatches)));

hold on;
plot(timestepAxis, psnr_epoch13, 'DisplayName', strcat('# of epochs=', num2str((name_epoch13+1)/numOfBatches)));

hold on;
plot(timestepAxis, psnr_epoch14, 'DisplayName', strcat('# of epochs=', num2str((name_epoch14+1)/numOfBatches)));

hold on;
plot(timestepAxis, psnr_epoch15, 'DisplayName', strcat('# of epochs=', num2str((name_epoch15+1)/numOfBatches)));

hold on;
plot(timestepAxis, psnr_epoch16, 'DisplayName', strcat('# of epochs=', num2str((name_epoch16+1)/numOfBatches)));

hold on;
plot(timestepAxis, psnr_epoch17, 'DisplayName', strcat('# of epochs=', num2str((name_epoch17+1)/numOfBatches)));

hold on;
plot(timestepAxis, psnr_epoch18, 'DisplayName', strcat('# of epochs=', num2str((name_epoch18+1)/numOfBatches)));

hold on;
plot(timestepAxis, psnr_epoch19, 'DisplayName', strcat('# of epochs=', num2str((name_epoch19+1)/numOfBatches)));

hold on;
plot(timestepAxis, psnr_epoch20, 'DisplayName', strcat('# of epochs=', num2str((name_epoch20+1)/numOfBatches)));


xlabel('Time Step');
ylabel('PSNR');
legend('Location', 'Best');
title('brain (ct2mr)');




%(4.2)plot SSIM.
figure(2);
plot(timestepAxis, ssim_epoch1, 'DisplayName', strcat('# of epochs=', num2str((name_epoch1+1)/numOfBatches)));

hold on;
plot(timestepAxis, ssim_epoch2, 'DisplayName', strcat('# of epochs=', num2str((name_epoch2+1)/numOfBatches)));

hold on;
plot(timestepAxis, ssim_epoch3, 'DisplayName', strcat('# of epochs=', num2str((name_epoch3+1)/numOfBatches)));

hold on;
plot(timestepAxis, ssim_epoch4, 'DisplayName', strcat('# of epochs=', num2str((name_epoch4+1)/numOfBatches)));

hold on;
plot(timestepAxis, ssim_epoch5, 'DisplayName', strcat('# of epochs=', num2str((name_epoch5+1)/numOfBatches)));

hold on;
plot(timestepAxis, ssim_epoch6, 'DisplayName', strcat('# of epochs=', num2str((name_epoch6+1)/numOfBatches)));

hold on;
plot(timestepAxis, ssim_epoch7, 'DisplayName', strcat('# of epochs=', num2str((name_epoch7+1)/numOfBatches)));

hold on;
plot(timestepAxis, ssim_epoch8, 'DisplayName', strcat('# of epochs=', num2str((name_epoch8+1)/numOfBatches)));

hold on;
plot(timestepAxis, ssim_epoch9, 'o--r', 'DisplayName', strcat('# of epochs=', num2str((name_epoch9+1)/numOfBatches)));

hold on;
plot(timestepAxis, ssim_epoch10, 'DisplayName', strcat('# of epochs=', num2str((name_epoch10+1)/numOfBatches)));

hold on;
plot(timestepAxis, ssim_epoch11, 'DisplayName', strcat('# of epochs=', num2str((name_epoch11+1)/numOfBatches)));

hold on;
plot(timestepAxis, ssim_epoch12, 'DisplayName', strcat('# of epochs=', num2str((name_epoch12+1)/numOfBatches)));

hold on;
plot(timestepAxis, ssim_epoch13, 'DisplayName', strcat('# of epochs=', num2str((name_epoch13+1)/numOfBatches)));

hold on;
plot(timestepAxis, ssim_epoch14, 'DisplayName', strcat('# of epochs=', num2str((name_epoch14+1)/numOfBatches)));

hold on;
plot(timestepAxis, ssim_epoch15, 'DisplayName', strcat('# of epochs=', num2str((name_epoch15+1)/numOfBatches)));

hold on;
plot(timestepAxis, ssim_epoch16, 'DisplayName', strcat('# of epochs=', num2str((name_epoch16+1)/numOfBatches)));

hold on;
plot(timestepAxis, ssim_epoch17, 'DisplayName', strcat('# of epochs=', num2str((name_epoch17+1)/numOfBatches)));

hold on;
plot(timestepAxis, ssim_epoch18, 'DisplayName', strcat('# of epochs=', num2str((name_epoch18+1)/numOfBatches)));

hold on;
plot(timestepAxis, ssim_epoch19, 'DisplayName', strcat('# of epochs=', num2str((name_epoch19+1)/numOfBatches)));

hold on;
plot(timestepAxis, ssim_epoch20, 'DisplayName', strcat('# of epochs=', num2str((name_epoch20+1)/numOfBatches)));

xlabel('Time Step');
ylabel('SSIM');
legend('Location', 'Best');
title('brain (ct2mr)');



%(4.3)plot MSE.
figure(3);
plot(timestepAxis, immse_epoch1, 'DisplayName', strcat('# of epochs=', num2str((name_epoch1+1)/numOfBatches)));

hold on;
plot(timestepAxis, immse_epoch2, 'o--r', 'DisplayName', strcat('# of epochs=', num2str((name_epoch2+1)/numOfBatches)));

hold on;
plot(timestepAxis, immse_epoch3, 'o--g', 'DisplayName', strcat('# of epochs=', num2str((name_epoch3+1)/numOfBatches)));

hold on;
plot(timestepAxis, immse_epoch4, 'o--cyan', 'DisplayName', strcat('# of epochs=', num2str((name_epoch4+1)/numOfBatches)));

hold on;
plot(timestepAxis, immse_epoch5, 'o--b', 'DisplayName', strcat('# of epochs=', num2str((name_epoch5+1)/numOfBatches)));

hold on;
plot(timestepAxis, immse_epoch6, 'DisplayName', strcat('# of epochs=', num2str((name_epoch6+1)/numOfBatches)));

hold on;
plot(timestepAxis, immse_epoch7, 'DisplayName', strcat('# of epochs=', num2str((name_epoch7+1)/numOfBatches)));

hold on;
plot(timestepAxis, immse_epoch8, 'DisplayName', strcat('# of epochs=', num2str((name_epoch8+1)/numOfBatches)));

hold on;
plot(timestepAxis, immse_epoch9, 'DisplayName', strcat('# of epochs=', num2str((name_epoch9+1)/numOfBatches)));

hold on;
plot(timestepAxis, immse_epoch10, 'DisplayName', strcat('# of epochs=', num2str((name_epoch10+1)/numOfBatches)));

hold on;
plot(timestepAxis, immse_epoch11, 'DisplayName', strcat('# of epochs=', num2str((name_epoch11+1)/numOfBatches)));

hold on;
plot(timestepAxis, immse_epoch12, 'DisplayName', strcat('# of epochs=', num2str((name_epoch12+1)/numOfBatches)));

hold on;
plot(timestepAxis, immse_epoch13, 'DisplayName', strcat('# of epochs=', num2str((name_epoch13+1)/numOfBatches)));

hold on;
plot(timestepAxis, immse_epoch14, 'DisplayName', strcat('# of epochs=', num2str((name_epoch14+1)/numOfBatches)));

hold on;
plot(timestepAxis, immse_epoch15, 'DisplayName', strcat('# of epochs=', num2str((name_epoch15+1)/numOfBatches)));

hold on;
plot(timestepAxis, immse_epoch16, 'DisplayName', strcat('# of epochs=', num2str((name_epoch16+1)/numOfBatches)));

hold on;
plot(timestepAxis, immse_epoch17, 'DisplayName', strcat('# of epochs=', num2str((name_epoch17+1)/numOfBatches)));

hold on;
plot(timestepAxis, immse_epoch18, 'DisplayName', strcat('# of epochs=', num2str((name_epoch18+1)/numOfBatches)));

hold on;
plot(timestepAxis, immse_epoch19, 'DisplayName', strcat('# of epochs=', num2str((name_epoch19+1)/numOfBatches)));

hold on;
plot(timestepAxis, immse_epoch20, 'DisplayName', strcat('# of epochs=', num2str((name_epoch20+1)/numOfBatches)));


xlabel('Time Step');
ylabel('MSE');
legend('Location', 'Best');
title('brain (ct2mr)');
ylim([0, 0.1]);


%{
%(4.4)plot FSIM.
figure(4);
plot(timestepAxis, FSIM_epoch1, 'DisplayName', strcat('# of epochs=', num2str((name_epoch1+1)/numOfBatches)));

hold on;
plot(timestepAxis, FSIM_epoch2, 'DisplayName', strcat('# of epochs=', num2str((name_epoch2+1)/numOfBatches)));

hold on;
plot(timestepAxis, FSIM_epoch3, 'DisplayName', strcat('# of epochs=', num2str((name_epoch3+1)/numOfBatches)));

hold on;
plot(timestepAxis, FSIM_epoch4, 'DisplayName', strcat('# of epochs=', num2str((name_epoch4+1)/numOfBatches)));

hold on;
plot(timestepAxis, FSIM_epoch5, 'DisplayName', strcat('# of epochs=', num2str((name_epoch5+1)/numOfBatches)));

hold on;
plot(timestepAxis, FSIM_epoch6, 'DisplayName', strcat('# of epochs=', num2str((name_epoch6+1)/numOfBatches)));

hold on;
plot(timestepAxis, FSIM_epoch7, 'DisplayName', strcat('# of epochs=', num2str((name_epoch7+1)/numOfBatches)));

hold on;
plot(timestepAxis, FSIM_epoch8, 'DisplayName', strcat('# of epochs=', num2str((name_epoch8+1)/numOfBatches)));

hold on;
plot(timestepAxis, FSIM_epoch9, 'o--b', 'DisplayName', strcat('# of epochs=', num2str((name_epoch9+1)/numOfBatches)));

hold on;
plot(timestepAxis, FSIM_epoch10, 'o--g', 'DisplayName', strcat('# of epochs=', num2str((name_epoch10+1)/numOfBatches)));

hold on;
plot(timestepAxis, FSIM_epoch11, 'o--r', 'DisplayName', strcat('# of epochs=', num2str((name_epoch11+1)/numOfBatches)));

hold on;
plot(timestepAxis, FSIM_epoch12, 'DisplayName', strcat('# of epochs=', num2str((name_epoch12+1)/numOfBatches)));

hold on;
plot(timestepAxis, FSIM_epoch13, 'DisplayName', strcat('# of epochs=', num2str((name_epoch13+1)/numOfBatches)));

hold on;
plot(timestepAxis, FSIM_epoch14, 'DisplayName', strcat('# of epochs=', num2str((name_epoch14+1)/numOfBatches)));

hold on;
plot(timestepAxis, FSIM_epoch15, 'DisplayName', strcat('# of epochs=', num2str((name_epoch15+1)/numOfBatches)));

hold on;
plot(timestepAxis, FSIM_epoch16, 'DisplayName', strcat('# of epochs=', num2str((name_epoch16+1)/numOfBatches)));

hold on;
plot(timestepAxis, FSIM_epoch17, 'DisplayName', strcat('# of epochs=', num2str((name_epoch17+1)/numOfBatches)));

hold on;
plot(timestepAxis, FSIM_epoch18, 'DisplayName', strcat('# of epochs=', num2str((name_epoch18+1)/numOfBatches)));

hold on;
plot(timestepAxis, FSIM_epoch19, 'DisplayName', strcat('# of epochs=', num2str((name_epoch19+1)/numOfBatches)));

hold on;
plot(timestepAxis, FSIM_epoch20, 'DisplayName', strcat('# of epochs=', num2str((name_epoch20+1)/numOfBatches)));

xlabel('Time Step');
ylabel('FSIM');
legend('Location', 'Best');
title('brain (ct2mr)');
%}



%??????????????.
executionTime = toc;
fprintf('metrics_diffEpochs.m execution time = %fs.\n', executionTime);