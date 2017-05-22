fid = fopen('data1.txt');
tline = fgets(fid);
class_path_list={};
while ischar(tline) 
    class_path_list{end+1} = tline;
    tline = fgets(fid);
end
fclose(fid); 

file_path= class_path_list{1};%ͼ���ļ���·��
img_path_list = dir(strcat(file_path,'*.jpg'));%��ȡ���ļ���������jpg��ʽ��ͼ��
n = length(img_path_list);
m=length(class_path_list);
load('w2c.mat');
trainSize = [128,128];
traindata=zeros(m*n, 128*128*11);
labels=ones(m*n,1);


for i=1:m
    file_path= class_path_list{i};%ͼ���ļ���·��
    img_path_list = dir(strcat(file_path,'*.jpg'));%��ȡ���ļ���������jpg��ʽ��ͼ��
    img_num = length(img_path_list);%��ȡͼ��������
    labels((i-1)*img_num+1:i*img_num,1)=i;             %cat--1  dog--2 chipmunk--3 giraffe--4 squirrel--5 wolf--6
    
% �õ���ɫ����
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
%��ά֮��Ҫ����contribution ѡ��ѵ��ά�ȣ� ���˾�����85%-95%���ң�����
%ά������̫�٣�4������������¿�����100%(63ά)
%report�еĽ������   ԭʼͼƬ��6��  CNά��  70ά        4��   63ά
%                    ���伯    6��         120ά       4��  70ά
t = templateSVM('KernelFunction','linear','Standardize',false);
SVMmodel=fitcecoc(score(:,1:100),labels,'Learners',t);           
%-------cross validate------------                              
sum=0;
for i=1:10
    CVSVM = crossval(SVMmodel,'KFold',n/2);       %CN������Kֵ �ڷ������������n/2��  
    loss = kfoldLoss(CVSVM);                      %��������£� ʹ��K=2 ���Խ��ƹ���������ʵ��ѵ�����ı���,ԭ���report
    accuracy = 1-loss;
    sum = accuracy+sum;
end
disp(sum/10)

%------------------test----------------------------
class1={'cat','dog','chipmunk','giraffe','squirrel','wolf'};
class2={'cat','dog','giraffe','wolf'};

test_path='.\testdata\';        %��ȡ����Ŀ¼
testimg_path_list = dir(strcat(test_path,'*.jpg'));
testimg_num = length(testimg_path_list);

for n=1:testimg_num
    testimg=imread(testimg_path_list(n).name);
    testdata=double(imresize(testimg,trainSize));
    feature=im2c(testdata, w2c,-2);
    featurePCA=bsxfun(@minus,feature,mean(traindata,1))*coeff;
    
   [label,s] = predict(SVMmodel,featurePCA(:,1:70));      %����ά����70��Ҫ��ǰ��ƥ��
   figure; imshow(testimg);
   title(['predictImage:',class2{label}]);      %6��class1   4��class2
end




%%%���Ų���
    testimg=imread('0a78053476f64a50887fa0467c59913b.jpg');
    testdata=double(imresize(testimg,trainSize));
    feature=im2c(testdata, w2c,-2);
    featurePCA=bsxfun(@minus,feature,mean(traindata,1))*coeff;
    imshow(testimg)
    [predictIndex,~] = predict(svm,featurePCA(:,1:70));        %����ά����70��Ҫ��ǰ��ƥ��
    fprintf('%d' ,predictIndex);

