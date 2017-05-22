%����Ŀ¼�ļ�
fid = fopen('data1_flip.txt');
tline = fgets(fid);
class_path_list={};
while ischar(tline) 
    class_path_list{end+1} = tline;
    tline = fgets(fid);
end
fclose(fid); 

%----------initialization--------

file_path= class_path_list{1};%ͼ���ļ���·��
img_path_list = dir(strcat(file_path,'*.jpg'));%��ȡ���ļ���������jpg��ʽ��ͼ��
n = length(img_path_list);
m=length(class_path_list);
trainSize = [128,128];
traindata=zeros(m*n, 324);
labels=ones(m*n,1);


for i=1:m
    file_path= class_path_list{i};%ͼ���ļ���·��
    img_path_list = dir(strcat(file_path,'*.jpg'));%��ȡ���ļ���������jpg��ʽ��ͼ��
    img_num = length(img_path_list);%��ȡͼ��������
    labels((i-1)*img_num+1:i*img_num,1)=i;
    
% �õ�HOG
    for j=1:img_num
      image_name = img_path_list(j).name;
      img=imread(strcat(file_path,image_name));
      %fprintf('%d',labels((i-1)*40+j,:));
      fprintf('%d  %s\n',j,strcat(file_path,image_name));
      im=imresize(img,trainSize);
      [hog1,visualization] = extractHOGFeatures(im,'CellSize',[32 32],'BlockSize',[2 2]);
      traindata((i-1)*img_num+j,:)=hog1;
    end
end


t = templateSVM('KernelFunction','linear','Standardize',false);
SVMmodel=fitcecoc(score1(:,1:100),labels,'Learners',t);
%-------cross validate------------
sum=0;
for i=1:10
    CVSVM = crossval(SVMmodel,'KFold',n/2);     %HOG ���� K=n/2 ÿ��ÿ��2�Ų���
    loss = kfoldLoss(CVSVM);
    accuracy = 1-loss;
    sum= sum+ accuracy;
end
disp(sum/10)


%-----------------test------------------
%�ļ���
class1={'cat','dog','chipmunk','giraffe','squirrel','wolf'};
class2={'cat','dog','giraffe','wolf'};

test_path='.\testdata\';         %�����ļ���Ŀ¼
testimg_path_list = dir(strcat(test_path,'*.jpg'));
testimg_num = length(testimg_path_list);

for n=1:testimg_num
    testimg=imread(testimg_path_list(n).name);
    testdata=imresize(testimg,trainSize);
    feature = extractHOGFeatures(testdata,'CellSize',[32 32],'BlockSize',[3 3]);
    
   [label,~] = predict(SVMmodel,feature);
   figure; imshow(testimg);
   title(['predictImage:',class{label}]);
end

%------------------���Ų���--------
testimg=imread('0cad278f4ba942d8abd5d021e54ca9b6.jpg');
imshow(testimg)
im=imresize(testimg,trainSize);
feature = extractHOGFeatures(im,'CellSize',[32 32],'BlockSize',[2 2]);
[label,~] = predict(SVMmodel,feature);
figure; imshow(testimg);
title(['predictImage:',class{label}]);

