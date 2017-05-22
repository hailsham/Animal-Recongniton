fid = fopen('data1.txt');
tline = fgets(fid);
class_path_list={};
while ischar(tline) 
    class_path_list{end+1} = tline;
    tline = fgets(fid);
end
fclose(fid); 

file_path= class_path_list{1};%图像文件夹路径
img_path_list = dir(strcat(file_path,'*.jpg'));%获取该文件夹中所有jpg格式的图像
n = length(img_path_list);
m=length(class_path_list);
load('w2c.mat');
trainSize = [128,128];
traindata=zeros(m*n, 128*128*11);
labels=ones(m*n,1);


for i=1:m
    file_path= class_path_list{i};%图像文件夹路径
    img_path_list = dir(strcat(file_path,'*.jpg'));%获取该文件夹中所有jpg格式的图像
    img_num = length(img_path_list);%获取图像总数量
    labels((i-1)*img_num+1:i*img_num,1)=i;             %cat--1  dog--2 chipmunk--3 giraffe--4 squirrel--5 wolf--6
    
% 得到颜色特征
    for j=1:img_num
      image_name = img_path_list(j).name;
      img=imread(strcat(file_path,image_name));
      fprintf('%d %d %s\n',j,strcat(file_path,image_name));
      trainImg=double(imresize(img,trainSize));
      out= im2c(trainImg, w2c,-2);
      traindata((i-1)*img_num+j,:)=out;
    end
end

%--------------------pca----------
[coeff,score,latent,tsquared,explained]=pca(traindata);
contribution=cumsum(latent)./ sum(latent);

%-------------training-------------
%降维之后要根据contribution 选择训练维度， 个人经验是85%-95%左右，但是
%维数不能太少，4类非扩充的情况下可以用100%(63维)
%report中的结果是在   原始图片集6类  CN维数  70维        4类   63维
%                    扩充集    6类         120维       4类  70维
t = templateSVM('KernelFunction','linear','Standardize',false);
SVMmodel=fitcecoc(score(:,1:100),labels,'Learners',t);           
%-------cross validate------------                              
sum=0;
for i=1:10
    CVSVM = crossval(SVMmodel,'KFold',n/2);       %CN特征，K值 在非扩充情况下是n/2，  
    loss = kfoldLoss(CVSVM);                      %扩充情况下， 使用K=2 可以近似估计其在真实的训练集的表现,原因见report
    accuracy = 1-loss;
    sum = accuracy+sum;
end
disp(sum/10)

%------------------test----------------------------
class1={'cat','dog','chipmunk','giraffe','squirrel','wolf'};
class2={'cat','dog','giraffe','wolf'};

test_path='.\testdata\';        %读取测试目录
testimg_path_list = dir(strcat(test_path,'*.jpg'));
testimg_num = length(testimg_path_list);

for n=1:testimg_num
    testimg=imread(testimg_path_list(n).name);
    testdata=double(imresize(testimg,trainSize));
    feature=im2c(testdata, w2c,-2);
    featurePCA=bsxfun(@minus,feature,mean(traindata,1))*coeff;
    
   [label,s] = predict(SVMmodel,featurePCA(:,1:70));      %这里维数（70）要跟前面匹配
   figure; imshow(testimg);
   title(['predictImage:',class2{label}]);      %6类class1   4类class2
end




%%%单张测试
    testimg=imread('0a78053476f64a50887fa0467c59913b.jpg');
    testdata=double(imresize(testimg,trainSize));
    feature=im2c(testdata, w2c,-2);
    featurePCA=bsxfun(@minus,feature,mean(traindata,1))*coeff;
    imshow(testimg)
    [predictIndex,~] = predict(svm,featurePCA(:,1:70));        %这里维数（70）要跟前面匹配
    fprintf('%d' ,predictIndex);

