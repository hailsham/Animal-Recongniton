%����Ŀ¼�ļ�
fid = fopen('data1.txt');
tline = fgets(fid);
class_path_list={};
while ischar(tline) 
    class_path_list{end+1} = tline;
    tline = fgets(fid);
end
fclose(fid); 

load('w2c.mat');                          %color name ��ӳ�����
file_path= class_path_list{1};
img_path_list = dir(strcat(file_path,'*.jpg'));
n = length(img_path_list);       %�ļ�����ͼƬ����
m=length(class_path_list);          %�ļ��и���
trainSize = [128,128];
featureHog=zeros(m*n, 324);
featureCN=zeros(m*n, 128*128*11);
labels=ones(m*n,1);

 %��ȡHOG����
for i=1:m
    file_path= class_path_list{i};%ͼ���ļ���·��
    img_path_list = dir(strcat(file_path,'*.jpg'));%��ȡ���ļ���������jpg��ʽ��ͼ��
    img_num = length(img_path_list);%��ȡͼ��������
    labels((i-1)*img_num+1:i*img_num,1)=i;
    
    for j=1:img_num
      image_name = img_path_list(j).name;
      img=imread(strcat(file_path,image_name));
      fprintf('%d  %s\n',j,strcat(file_path,image_name));
      
     
      im=imresize(img,trainSize);
      [hog1,visualization] = extractHOGFeatures(im,'CellSize',[32 32],'BlockSize',[2 2]);
      featureHog((i-1)*img_num+j,1:324)=hog1;
      
    end
end

 %��ȡColor Name
 for i=1:m
    file_path= class_path_list{i};
    img_path_list = dir(strcat(file_path,'*.jpg'));
    img_num = length(img_path_list);
   % labels((i-1)*40+1:i*40,1)=i;
    
% �õ���ɫ����
    for j=1:img_num
      image_name = img_path_list(j).name;
      img=imread(strcat(file_path,image_name));
      fprintf('%d %d %s\n',j,strcat(file_path,image_name));
      trainImg=double(imresize(img,trainSize));
      out= im2c(trainImg, w2c,-2);                  %����im2c���CN ����
      featureCN((i-1)*img_num+j,:)=out;
    end
 end

 rng(1);
%--------------����pca��ά----------

[coeff,score,latent,tsquared,explained]=pca(featureCN);
contribution= cumsum(latent)./ sum(latent);         
[coeff1,score1,latent1,tsquared1,explained1]=pca(featureHog);
contribution1= cumsum(latent)./ sum(latent);
%--------------------train---------
%--�����ά��Ҫ����contribution��ѡ�� 
%report�еĽ������   ԭʼͼƬ��6��  CNά��  70ά        4��   63ά
%                    ���伯    6��         120ά       4��  70ά
traindata= zeros(m*n,324+100);                              
traindata(:,1:324) = featureHog;                
traindata(:,325:324+70)=score(:,1:100);             



t = templateSVM('KernelFunction','linear','Standardize',false);
SVMmodel=fitcecoc(traindata,labels,'Learners',t);
%-------cross validate------------
%ȡƽ��  ԭʼͼƬ��  K=n/2  ���伯K=2 ԭ�������
sum=0;
for i=1:10
    CVSVM = crossval(SVMmodel,'KFold',n/2);       
    loss = kfoldLoss(CVSVM);
    accuracy = 1-loss;
    sum = sum + accuracy;
end
disp(sum/10)      %׼ȷ��



%----------------test----------------
class1={'cat','dog','chipmunk','giraffe','squirrel','wolf'};
class2={'cat','dog','giraffe','wolf'};

test_path='.\testdata\';            %�²����ļ���
testimg_path_list = dir(strcat(test_path,'*.jpg'));
testimg_num = length(testimg_path_list);

for n=1:testimg_num
    feature=zeros(1,394);
    testimg=imread(testimg_path_list(n).name);
    testdata=imresize(testimg,trainSize);
    featureHog_test= extractHOGFeatures(testdata,'CellSize',[32 32],'BlockSize',[2 2]);
    %featureHog_PCA = bsxfun(@minus,featureHog_test,mean(featureHog,1))*coeff1;
    
    CNinput=double(testdata);
    featureCN_test=im2c(CNinput, w2c,-2);
    featureCN_PCA=bsxfun(@minus,featureCN_test,mean(featureCN,1))*coeff;
    feature(:,1:324) =  featureHog_test;
    feature(:,325:394)= featureCN_PCA(:,1:70);        %����ά����70��Ҫ��ǰ��ƥ��
    
   [label,score] = predict(SVMmodel,feature);
   figure; imshow(testimg);
   title(['predict result:',class2{label}]);         %���� class2 ���� class1
end






