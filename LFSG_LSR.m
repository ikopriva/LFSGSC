%% LFSG_LSR: Mutual (dis)agreement based hyperparameters tuning for least squares regression (LSR) based self-supervised subspace clustering
% Label-free self-guided LSR algorithm
%
% I. Kopriva 2024-01/04

%%
clear
close all

% Set path to all subfolders
addpath(genpath('.'));

%% parameters
 
hyper_flag = 1; % 1 - max ACC criterion; 2 - nax NMI criterion; 

% Number of evaluations
numit =25;
rel_err_thr = 0.001;
%% Load the data from the chosen dataset

% Please uncomment the dataset that you want to use and comment the other ones
% dataName = 'YaleBCrop025';
% dataName = 'MNIST';
% dataName = 'USPS';
 dataName = 'ORL';
% dataName = 'COIL20'; 
% dataName = 'COIL100';

% Performance metrics are saved in *.mat file named according to:
% save LFSG_LSR_dataset_metric_ACC_or_NMI (depends on the value of hyper_flag)

%% prepare data 
if strcmp(dataName,'YaleBCrop025')
   % Dataset dependent parameters
   dimSubspace=9;   % subspaces dimension
   numIn = 43;  % number of in-sample data
   numOut = 21; % number of out-of-sample data
   ni = 64; % number of images per group
   nc = 38;  % number of groups
   % LSR SC algortithm
    lambda = [1e-5 1e-4 1e-3 1e-2 1e-1 5e-1 1  5 1e1]; 
   % END of dataset dependent parameters

   load YaleBCrop025.mat;
   [i1, i2, ni, nc] = size(I); % rectangular image size: i1 x i2, number of images per person: ni, number of persons: nc
   clear I Ind s ns
   
   N = nc*ni; % number of samples
   X = zeros(i1*i2, N); labels = zeros(N,1); % allocation of space
    
   ns = 0; % sample number counter
   for i=1:nc % person
       for j=1:ni % face image
           ns = ns + 1; % sample index
           X(:,ns) = Y(:,j,i); % sample (columns of X represent vectorized data of rectangular images)
           labels(ns,1) = i;    % to be used for oracle based validation 
       end
   end
   Y = X;
   clear X i j
   
elseif strcmp(dataName,'MNIST')
   % Dataset dependent parameters
   dimSubspace=12;   % subspaces dimension
   numIn = 50;  % number of in-sample data
   numOut = 50; % number of out-of-sample data
   nc = 10;   % number of groups
   lambda = [1e-2 1e-1 5e-1 3 5 1e1]; 
   % END of dataset dependant parameters
    
   images = loadMNISTImages('t10k-images.idx3-ubyte'); % columns of X represent vectorized data of squared images
   % i1 = 28; i2 = 28; N = 10000; % 10000 images of ten digits (each image 28x28 pixels)
   
   labels = loadMNISTLabels('t10k-labels.idx1-ubyte'); % to be used for oracle based validation
   % nc = 10; % ten hand-written digits
   [labelssorted,IX] = sort(labels);
   Y = images(:,IX);  
   labels = labelssorted + 1; 
   clear labelssorted IX images  

elseif strcmp(dataName,'USPS')
   % Dataset dependant parameters
   dimSubspace=12;   % subspaces dimension
   numIn = 50;  % number of in-sample data
   numOut = 50; % number of out-of-sample data
   nc = 10;    % number of groups
   lambda = [1e-2 1e-1 5e-1 3 5 1e1]; 
   % END of dataset dependant parameters
     
   data = load('usps');
   X = data(:,2:end)'; % columns of X represent vectorized data of squared images
   % i1 = 16; i2 = 16; N = 7291; nc = 10; % 1000 images of each of ten digits (each image 16x16 pixels)
    
   labels = data(:,1)-1;  % to be used for oracle based validation
   % nc = 10; % ten hand-written digits
   [labelssorted,IX] = sort(labels);
   Y = X(:,IX);  
   labels = labelssorted + 1; 
   clear data X labelssorted IX

elseif strcmp(dataName,'ORL')  
   % Dataset dependant parameters
   dimSubspace=7;   % subspaces dimension
   numIn = 7;  % number of in-sample data
   numOut = 3; % number of out-of-sample data
   nc = 40; % number of groups
   % END of dataset dependant parameters
   % LSR SC algortithm
   lambda = [1e-5 1e-4 1e-3 1e-2 1e-1 5e-1 1 2 5 1e1]; 

   data = load('ORL_32x32.mat');  
   Y = data.fea'; % columns of X represent vectorized data of squared images
   % i1 = 32; i2 = 32; N = 400; nc = 40; % 400 face images of 40 persons (each image 32x32 pixels)
    
   labels=data.gnd;   % to be used for oracle based validation
   clear data  

elseif strcmp(dataName,'COIL20')
   % Dataset dependant parameters
   dimSubspace=9;   % subspaces dimension
   numIn = 26;  % number of in-sample data
   numOut = 26; % number of out-of-sample data
   nc = 20; % number of groups
   % END of dataset dependant parameters
   lambda = [1e-5 1e-4 1e-3 1e-2 1e-1 3e-1 5e-1 7e-1 1 2 5 1e1];  

   load COIL20.mat
   Y=transpose(fea); % columns of X represent vectorized data of squared images
   % i1=32; i2=32; N=1440; nc=20; % 1440 images of 20 objects (72 images per object) (each image is 32x32 pixels)
   clear fea;
    
   labels=gnd;   % to be used for oracle based validation
   % nc = 20; % twenty objects images
   clear gnd  

elseif strcmp(dataName,'COIL100')
   % Dataset dependant parameters
   dimSubspace=9;   % subspaces dimension
   numIn = 26;  % number of in-sample data
   numOut = 26; % number of out-of-sample data   
   nc = 100; % number of groups
   % END of dataset dependant parameters
   lambda = [1e-5 1e-4 1e-3 1e-2 1e-1 3e-1 5e-1 7e-1 1 2 5 1e1]; 
     
   load COIL100.mat
   Y=double(fea.'); % columns of X represent vectorized data of squared images
   % i1=32; i2=32; N=7200; nc=100; % 7200 images of 100 objects (72 images per object) (each image is 32x32 pixels)
   clear fea;

   labels=gnd;    % to be used for oracle based validation
   % nc = 100; % one hundred objects images
   clear gnd

end

%%
for it=1:numit
    fprintf('Iter %d\n',it);

    % Each category is separately split, to ensure proportional representation
    nIn = 1; nOut = 1;
    for c=1:nc % Through all categories
        ind = (labels == c); % Indices of the chosen category
        Xc = Y(:,ind);       % Samples ...
        numSamples = size(Xc, 2); % Number of samples ...
        ind = randperm(numSamples); % Random permutation of the indices
       % numIn = ceil(numSamples * pIn/100); % In sample size
       % numOut = numSamples - numIn; % Out sample size
        X_in(:,    nIn:nIn+numIn-1 ) = Xc(:, ind(1:numIn)); % Data
        X_out(:, nOut:nOut+numOut-1) = Xc(:, ind(numIn+1:numIn+numOut));
        labels_in(  nIn:nIn + numIn-1) = c; % Labels
        labels_out(nOut:nOut+numOut-1) = c;
        nIn  = nIn  + numIn; % Next indices
        nOut = nOut + numOut;
    end
    X_in( :,   nIn:end) = []; % Cut out the surplus of the allocated space
    X_out(:,  nOut:end) = [];
    labels_in(  nIn:end) = [];
    labels_out(nOut:end) = [];

    X_in = normc(X_in);
    XTX = transpose(X_in)*X_in;

    for i=1:length(lambda)
        Z = pinv(XTX +lambda(i)*eye(size(X_in,2)))*XTX;
        A(i,:) = SpectralClusteringL(abs(Z)+abs(Z'),nc);
        A(i,:) = bestMap(labels_in,A(i,:));
        % A(i,:) = adjacency_matrix_angular_domain(Z, delta);
        CE(i)  = computeCE(A(i,:),labels_in);  % ORACLE
        NMI_o(i) = compute_nmi(A(i,:),labels_in); % ORACLE
        F1(i) = compute_f(A(i,:),labels_in);  % ORACLE
    end

    [cemin imin] = min(CE);
    ACC_star(it) = 1 - cemin;
    lambda_maxacc(it) = lambda(imin);

    [nmimax imax] = max(NMI_o);
    NMI_star(it) = nmimax;
    lambda_nmimax(it)= lambda(imax);

    [f1max imax] = max(F1);
    F1_star(it) = f1max;
    lambda_f1max(it)= lambda(imax);

    if hyper_flag == 1
        Z = pinv(XTX +lambda_maxacc(it)*eye(size(X_in,2)))*XTX; 
    elseif hyper_flag == 2
        Z = pinv(XTX +lambda_nmimax(it)*eye(size(X_in,2)))*XTX; 
    elseif hyper_flag == 3
         Z = pinv(XTX +lambda_f1max(it)*eye(size(X_in,2)))*XTX; 
    end
    labels_star=SpectralClusteringL(abs(Z)+abs(Z'),nc);
    labels_star = bestMap(labels_in,labels_star);
    % estimate bases labels estimated on in-sample data
    [B_star, begB_star, endB_star, mu_star]  = bases_estimation(X_in, labels_star, dimSubspace);

    ACC = zeros(length(lambda),length(lambda));
    NMI = zeros(length(lambda),length(lambda));
    Fscore = zeros(length(lambda),length(lambda));

    for i=1:length(lambda)-1
        for j=i+1:length(lambda)
            ACC(i,j)  = 1 - computeCE(A(i,:),A(j,:));
            NMI(i,j) = compute_nmi(A(i,:),A(j,:));
            Fscore(i,j) = compute_f(A(i,:),A(j,:));
        end
    end

    if hyper_flag == 1
        [amax imax] = max(ACC);
        [amax jmax] = max(amax);
    elseif hyper_flag == 2
        [nmix imax] = max(NMI);
        [nmix jmax] = max(nmix);
    elseif hyper_flag == 3
        [fscx imax] = max(Fscore);
        [fscx jmax] = max(fscx);
    end

    lambda_i = lambda(imax(jmax));
    labels_i = A(imax(jmax),:);
    lambda_j = lambda(jmax);
    labels_j = A(jmax,:);

    lambda_k1 = (2*lambda_i + lambda_j)/3;
    lambda_k2 = (lambda_i + 2*lambda_j)/3;

    rel_err = (lambda_j-lambda_i)/lambda_j;

    while rel_err > rel_err_thr
        Z = pinv(XTX +lambda_k1*eye(size(X_in,2)))*XTX;
        labels_k1 = SpectralClusteringL(abs(Z)+abs(Z'),nc);
        Z = pinv(XTX +lambda_k2*eye(size(X_in,2)))*XTX;
        labels_k2 = SpectralClusteringL(abs(Z)+abs(Z'),nc);   
        if hyper_flag == 1
           labels_i = bestMap(labels_k1,labels_i);
            metric_ik1 = 1 - computeCE(labels_i,labels_k1);
            labels_k1 = bestMap(labels_k2,labels_k1);
            metric_k1k2 = 1 - computeCE(labels_k1,labels_k2);
            labels_j = bestMap(labels_k2,labels_j);
            metric_k2j = 1 - computeCE(labels_k2,labels_j);
        elseif hyper_flag == 2
            labels_i = bestMap(labels_k1,labels_i);
            metric_ik1 = compute_nmi(labels_i,labels_k1);
            labels_k1 = bestMap(labels_k2,labels_k1);
            metric_k1k2 = compute_nmi(labels_k1,labels_k2);
            labels_j = bestMap(labels_k2,labels_j);   
            metric_k2j = compute_nmi(labels_k2,labels_j);
        end

        if (metric_ik1 >= metric_k1k2) && (metric_ik1 >= metric_k2j)
            lambda_j = lambda_k1;
            labels_j = labels_k1;
            lambda_k1 = (2*lambda_i + lambda_j)/3;
            lambda_k2 = (lambda_i + 2*lambda_j)/3;
            rel_err=(lambda_k1-lambda_i)/lambda_k1;
        elseif metric_k1k2 >= metric_k2j
            lambda_i = lambda_k1;
            lambda_j = lambda_k2;
            labels_i = labels_k1;
            labels_j = labels_k2;
            lambda_k1 = (2*lambda_i + lambda_j)/3;
            lambda_k2 = (lambda_i + 2*lambda_j)/3; 
            rel_err=(lambda_k2-lambda_k1)/lambda_k2;
        else
            lambda_i = lambda_k2;
            labels_i = labels_k2;
            lambda_k1 = (2*lambda_i + lambda_j)/3;
            lambda_k2 = (lambda_i + 2*lambda_j)/3; 
            rel_err=(lambda_j-lambda_k2)/lambda_j;
        end
    end

    labels_i = bestMap(labels_k1,labels_i);
    labels_j = bestMap(labels_k2,labels_j);
    if hyper_flag == 1
        metric_ik1 = 1 - computeCE(labels_i,labels_k1);
        metric_ik2 = 1 - computeCE(labels_k2,labels_j);
    elseif hyper_flag == 2
        metric_ik1 = compute_nmi(labels_i,labels_k1);
        metric_ik2 = compute_nmi(labels_k2,labels_j);
    end

    if metric_ik1 >= metric_ik2
        lambda_k = lambda_k1;
    else
        lambda_k = lambda_k2;
    end

    lambda_est(it)=lambda_k;

    Z = pinv(XTX +lambda_k*eye(size(X_in,2)))*XTX;
    labels_est = SpectralClusteringL(abs(Z)+abs(Z'),nc);
    labels_est = bestMap(labels_in,labels_est);
    CE_est  = computeCE(labels_est,labels_in);
    ACC_est(it) = 1 - CE_est;
    NMI_est(it) = compute_nmi(labels_est,labels_in);
    F1_est(it) = compute_f(labels_est,labels_in');

  
    % OUT-OF-SAMPLE DATA
    % estimate bases labels estimated on in-sample data
    [B_x, begB_x, endB_x, mu_X]  = bases_estimation(X_in, labels_est, dimSubspace);
    A0 = labels_out;
    N_out = size(X_out,2);
    X_out = normc(X_out);

    for l=1:nc
        X_outm = X_out - mu_X(:,l);    % make data zero mean for distance calculation
        BB=B_x(:,begB_x(l):endB_x(l));
        Xproj = (BB*BB')*X_outm;
        Dproj = X_outm - Xproj;
        D(l,:) = sqrt(sum(Dproj.^2,1));
    end
    [~, A_x] = min(D);
    clear D

    % Performance on out-of-sample data   
    A_x = bestMap(A0,A_x);    
    ACC_out(it)  = 1 - computeCE(A_x,A0);
    NMI_out(it) = compute_nmi(A0,A_x);
    Fscore_out(it) = compute_f(A0,A_x');
    clear A_x

    for l=1:nc
        X_outm = X_out - mu_star(:,l);    % make data zero mean for distance calculation
        BB=B_star(:,begB_star(l):endB_star(l));
        Xproj = (BB*BB')*X_outm;
        Dproj = X_outm - Xproj;
        D(l,:) = sqrt(sum(Dproj.^2,1));
    end
    [~, A_x] = min(D);
    clear D

    % Performance on out-of-sample data 
    A_x = bestMap(A0,A_x);    
    ACC_out_star(it)  = 1 - computeCE(A_x,A0);
    NMI_out_star(it) = compute_nmi(A0,A_x);
    Fscore_out_star(it) = compute_f(A0,A_x');    
    clear A_x

    save LFSG_LSR_ORL_metric_ACC.mat ACC_star ACC_est ACC_out NMI_star NMI_est NMI_out F1_star F1_est Fscore_out...
    lambda_maxacc lambda_nmimax lambda_f1max ACC_out_star NMI_out_star Fscore_out_star lambda_est 
end

display('********** ACCURACY *****************')
display('ORACLE')
display('Mean')
mean(ACC_star)
display('Std:')
std(ACC_star)

display('ESTIMATED - IN-SAMPLE DATA')
display('Mean')
mean(ACC_est)
display('Std:')
std(ACC_est)

display('ORACLE - OUT-OF-SAMPLE DATA')
display('Mean')
mean(ACC_out_star)
display('Std:')
std(ACC_out_star)

display('ESTIMATED - OUT-OF-SAMPLE DATA')
display('Mean')
mean(ACC_out)
display('Std:')
std(ACC_out)

% ranksum two sided Wilcoxon test of statistical significance
p_acc_in = ranksum(ACC_star,ACC_est)
p_acc_out = ranksum(ACC_out_star,ACC_out)

display('*************** NMI ********************')
display('ORACLE')
display('Mean:')
mean(NMI_star)
display('Std:')
std(NMI_star)

display('ESTIMATED: IN-SMAPLE')
display('Mean')
mean(NMI_est)
display('Std:')
std(NMI_est)

display('ORACLE: OUT-OF-SMAPLE')
display('Mean')
mean(NMI_out_star)
display('Std:')
std(NMI_out_star)

display('ESTIMATED: OUT-OF-SMAPLE')
display('Mean')
mean(NMI_out)
display('Std:')
std(NMI_out)

% ranksum two sided Wilcoxon test of statistical significance
p_nmi_in = ranksum(NMI_star,NMI_est)
p_nmi_out = ranksum(NMI_out_star,NMI_out)

display('*************** F1 score ********************')
display('ORACLE')
display('Mean:')
mean(F1_star)
display('Std:')
std(F1_star)

display('ESTIMATED: IN-SAMPLE')
display('Mean')
mean(F1_est)
display('Std:')
std(F1_est)

display('ORCALE: OUT-OF-SAMPLE')
display('Mean')
mean(Fscore_out_star)
display('Std:')
std(Fscore_out_star)

display('ESTIMATED: OUT-OF-SAMPLE')
display('Mean')
mean(Fscore_out)
display('Std:')
std(Fscore_out)

% ranksum two sided Wilcoxon test of statistical significance
p_F1_in = ranksum(F1_star,F1_est)
p_F1_out = ranksum(Fscore_out_star,Fscore_out)

display('***************** lambda **********************')
display('MAXACC ORACLE:')
mean(lambda_maxacc)
std(lambda_maxacc)
display('MAXNMI ORACLE:')
mean(lambda_nmimax)
std(lambda_nmimax)
display('MAXF1 ORACLE:')
mean(lambda_f1max)
std(lambda_f1max)
display('lambda_EST:')
mean(lambda_est)
std(lambda_est)

% ranksum two sided Wilcoxon test of statistical significance
p_lambda_acc = ranksum(lambda_maxacc,lambda_est)
p_lambda_nmi = ranksum(lambda_nmimax,lambda_est)
p_lambda_F1 = ranksum(lambda_f1max,lambda_est)

save LFSG_LSR_ORL_metric_ACC.mat ACC_star ACC_est ACC_out NMI_star NMI_est NMI_out F1_star F1_est Fscore_out...
    lambda_maxacc lambda_nmimax lambda_f1max p_lambda_F1 p_lambda_nmi p_lambda_acc ACC_out_star...
    NMI_out_star Fscore_out_star lambda_est 
    
