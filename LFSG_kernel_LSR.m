%% LFSG_kernel_LSR: Mutual (dis)agreement based hyperparameters tuning for kernel least squares regression (LSR) based self-supervised subspace clustering
% Label-free self-guided kernel LSR algorithm
%
% I. Kopriva 2024-07

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
% dataName = 'COIL100'; % Color images

 % Performance metrics are saved in *.mat file named according to:
% save LFSG_kernel_LSR_dataset_metric_ACC_or_NMI (depends on the value of hyper_flag)

%% prepare data 
if strcmp(dataName,'YaleBCrop025')
   % Dataset dependent parameters
   dimSubspace=9;   % subspaces dimension
   numIn = 43;  % number of in-sample data
   numOut = 21; % number of out-of-sample data
   ni = 64; % number of images per group
   nc = 38;  % number of groups
   % kernel LSC algorithm
   lambda = [1e-4 1e-3 1e-2 1e-1];
   sig2 = [1 1e1 1e2 5e2 1e3 5e3 1e4];
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
   YY = X;
   clear X i j
   
elseif strcmp(dataName,'MNIST')
   % Dataset dependent parameters
   dimSubspace=12;   % subspaces dimension
   numIn = 50;  % number of in-sample data
   numOut = 50; % number of out-of-sample data
   nc = 10;   % number of groups
   % kernel LSC algorithm
   lambda = [1e-5 1e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1];  
   sig2 = [1 5 1e1 5e1 1e2 5e2 1e3 5e3 1e4 5e4 1e5];
   % END of dataset dependent parameters
    
   images = loadMNISTImages('t10k-images.idx3-ubyte'); % columns of X represent vectorized data of squared images
   % i1 = 28; i2 = 28; N = 10000; % 10000 images of ten digits (each image 28x28 pixels)
   
   labels = loadMNISTLabels('t10k-labels.idx1-ubyte'); % to be used for oracle based validation
   % nc = 10; % ten hand-written digits
   [labelssorted,IX] = sort(labels);
   YY = images(:,IX);  
   labels = labelssorted + 1; 
   clear labelssorted IX images  

elseif strcmp(dataName,'USPS')
   % Dataset dependant parameters
   dimSubspace=12;   % subspaces dimension
   numIn = 50;  % number of in-sample data
   numOut = 50; % number of out-of-sample data
   nc = 10;    % number of groups
   % kernel LSC algorithm
   lambda = [1e-5 1e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1];  
   sig2 = [1 5 1e1 5e1 1e2 5e2 1e3 5e3 1e4 5e4 1e5];
   % END of dataset dependent parameters
     
   data = load('usps');
   X = data(:,2:end)'; % columns of X represent vectorized data of squared images
   % i1 = 16; i2 = 16; N = 7291; nc = 10; % 1000 images of each of ten digits (each image 16x16 pixels)
    
   labels = data(:,1)-1;  % to be used for oracle based validation
   % nc = 10; % ten hand-written digits
   [labelssorted,IX] = sort(labels);
   YY = X(:,IX);  
   labels = labelssorted + 1; 
   clear data X labelssorted IX

elseif strcmp(dataName,'ORL')  
   % Dataset dependant parameters
   dimSubspace=7;   % subspaces dimension
   numIn = 7;  % number of in-sample data
   numOut = 3; % number of out-of-sample data
   nc = 40; % number of groups
   % Kernel LSR algorithm
   lambda = [1e-5 1e-4 1e-3 1e-2 1e-1];
   sig2 = [1 1e1 1e2 5e2 1e3 5e3 1e4];
   % END of dataset dependent parameters

   data = load('ORL_32x32.mat');  
   YY = data.fea'; % columns of X represent vectorized data of squared images
   % i1 = 32; i2 = 32; N = 400; nc = 40; % 400 face images of 40 persons (each image 32x32 pixels)
    
   labels=data.gnd;   % to be used for oracle based validation
   clear data  

elseif strcmp(dataName,'COIL20')
   % Dataset dependant parameters
   dimSubspace=9;   % subspaces dimension
   numIn = 50;  % number of in-sample data
   numOut = 22; % number of out-of-sample data
   nc = 20; % number of groups
   % kernel LSC algorithm
   lambda = [1e-5 1e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1];  
   sig2 = [1 5 1e1 5e1 1e2 5e2 1e3 5e3 1e4 5e4 1e5];
   % END of dataset dependent parameters
  
   load COIL20.mat
   YY=transpose(fea); % columns of X represent vectorized data of squared images
   % i1=32; i2=32; N=1440; nc=20; % 1440 images of 20 objects (72 images per object) (each image is 32x32 pixels)
   clear fea;
    
   labels=gnd;   % to be used for oracle based validation
   clear gnd  

elseif strcmp(dataName,'COIL100')
   % Dataset dependant parameters
   dimSubspace=9;   % subspaces dimension
   numIn = 50;  % number of in-sample data
   numOut = 22; % number of out-of-sample data   
   nc = 100; % number of groups
   % kernel LSC algorithm
   lambda = [1e-5 1e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1];  
   sig2 = [1 5 1e1 5e1 1e2 5e2 1e3 5e3 1e4 5e4 1e5];
   % END of dataset dependent parameters
     
   load COIL100.mat
   YY=double(fea.'); % columns of X represent vectorized data of squared images
   % i1=32; i2=32; N=7200; nc=100; % 7200 images of 100 objects (72 images per object) (each image is 32x32 pixels)
   clear fea;

   labels=gnd;    % to be used for oracle based validation
   % nc = 100; % one hundred objects images
   clear gnd
end

%% Iterate
for it=1:numit
    fprintf('Iter %d\n',it);
    
    % Each category is separately split, to ensure proportional representation
    nIn = 1; nOut = 1;
    for c=1:nc % Through all categories
        ind = (labels == c); % Indices of the chosen category
        Xc = YY(:,ind);       % Samples ...
        numSamples = size(Xc, 2); % Number of samples ...
        ind = randperm(numSamples); % Random permutation of the indices
        X_in(:, nIn:nIn+numIn-1 ) = Xc(:, ind(1:numIn)); % Data
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

    % Tuning over sig2 for a fixed lambda
    lambda_tune = lambda(ceil(length(lambda)/2));    
    X_in = normc(X_in);
    DD = pdist2(X_in',X_in');

  %  for jj=1:length(lambda)
    for i=1:length(sig2)
        KXX=exp(-(DD.*DD)/2/sig2(i));     % Gaussian kernel
        Z = pinv(KXX+lambda_tune*eye(size(KXX,2)))*KXX;
        A(i,:) = SpectralClusteringL(abs(Z)+abs(Z'),nc);
        A(i,:) = bestMap(labels_in,A(i,:));
        CE(i)  = computeCE(A(i,:),labels_in);
        NMI_o(i) = compute_nmi(A(i,:),labels_in);
        F1(i) = compute_f(A(i,:),labels_in);
    end


    [cemin imin] = min(CE);
    sig2_accmax(it) = sig2(imin);

    [nmimax imax] = max(NMI_o);
    sig2_nmimax(it)= sig2(imax);

    [f1max imax] = max(F1);
    sig2_f1max(it)= sig2(imax);
    
    ACC = zeros(length(sig2),length(sig2));
    NMI = zeros(length(sig2),length(sig2));
    Fscore = zeros(length(sig2),length(sig2));

    for i=1:length(sig2)-1
        for j=i+1:length(sig2)
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
    end

    sig2_i = sig2(imax(jmax));
    labels_i = A(imax(jmax),:);
    sig2_j = sig2(jmax);
    labels_j = A(jmax,:);

    sig2_k1 = (2*sig2_i + sig2_j)/3;
    sig2_k2 = (sig2_i + 2*sig2_j)/3;

    rel_err = (sig2_j-sig2_i)/sig2_j;

    while rel_err > rel_err_thr
        KXX=exp(-(DD.*DD)/2/sig2_k1);     % Gaussian kernel
        Z = pinv(KXX+lambda_tune*eye(size(KXX,2)))*KXX;
        labels_k1 = SpectralClusteringL(abs(Z)+abs(Z'),nc);

        KXX=exp(-(DD.*DD)/2/sig2_k2);     % Gaussian kernel
        Z = pinv(KXX+lambda_tune*eye(size(KXX,2)))*KXX;
        labels_k2 = SpectralClusteringL(abs(Z)+abs(Z'),nc);

        labels_i = bestMap(labels_in,labels_i);
        labels_j = bestMap(labels_in,labels_j);
        labels_k1 = bestMap(labels_in,labels_k1);
        labels_k2 = bestMap(labels_in,labels_k2);

        if hyper_flag == 1
            metric_ik1 = 1 - computeCE(labels_i,labels_k1);
            metric_k1k2 = 1 - computeCE(labels_k1,labels_k2);
            metric_k2j = 1 - computeCE(labels_k2,labels_j);
        elseif hyper_flag == 2
            metric_ik1 = compute_nmi(labels_i,labels_k1);
            metric_k1k2 = compute_nmi(labels_k1,labels_k2);
            metric_k2j = compute_nmi(labels_k2,labels_j);
        end

        if (metric_ik1 >= metric_k1k2) && (metric_ik1 >= metric_k2j)
            sig2_j = sig2_k1;
            labels_j = labels_k1;
            sig2_k1 = (2*sig2_i + sig2_j)/3;
            sig2_k2 = (sig2_i + 2*sig2_j)/3;
            rel_err=(sig2_k1-sig2_i)/sig2_k1;
        elseif metric_k1k2 >= metric_k2j
            sig2_i = sig2_k1;
            sig2_j = sig2_k2;
            labels_i = labels_k1;
            labels_j = labels_k2;
            sig2_k1 = (2*sig2_i + sig2_j)/3;
            sig2_k2 = (sig2_i + 2*sig2_j)/3;
            rel_err=(sig2_k2-sig2_k1)/sig2_k2;
        else
            sig2_i = sig2_k2;
            labels_i = labels_k2;
            sig2_k1 = (2*sig2_i + sig2_j)/3;
            sig2_k2 = (sig2_i + 2*sig2_j)/3;
            rel_err=(sig2_j-sig2_k2)/sig2_j;
        end
    end

    if hyper_flag == 1
        metric_ik1 = 1 - computeCE(labels_i,labels_k1);
        metric_ik2 = 1 - computeCE(labels_k2,labels_j);
    elseif hyper_flag == 2
        metric_ik1 = compute_nmi(labels_i,labels_k1);
        metric_ik2 = compute_nmi(labels_k2,labels_j);
    end

    if metric_ik1 >= metric_ik2
        sig2_k = sig2_k1;
    else
        sig2_k = sig2_k2;
    end

    sig2_est(it)=sig2_k;

    clear CE NMI_o F1
    % tuning over lambda for a fixed sig2
    for i=1:length(lambda)
        if hyper_flag == 1
            sig2_star = sig2_accmax(it);
        elseif hyper_flag == 2
            sig2_star = sig2_nmimax(it);
        end
        KXX=exp(-(DD.*DD)/2/sig2_star);     % Gaussian kernel
        Z = pinv(KXX+lambda(i)*eye(size(KXX,2)))*KXX;
        A(i,:) = SpectralClusteringL(abs(Z)+abs(Z'),nc);
        A(i,:) = bestMap(labels_in,A(i,:));
        CE(i)  = computeCE(A(i,:),labels_in);
        NMI_o(i) = compute_nmi(A(i,:),labels_in);
        F1(i) = compute_f(A(i,:),labels_in);
    end

    [cemin imin] = min(CE);
    lambda_accmax(it) = lambda(imin);

    [nmimax imax] = max(NMI_o);
    lambda_nmimax(it)= lambda(imax);

    [f1max imax] = max(F1);
    lambda_f1max(it)= lambda(imax);

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
    end

    lambda_i = lambda(imax(jmax));
    labels_i = A(imax(jmax),:);
    lambda_j = lambda(jmax);
    labels_j = A(jmax,:);

    lambda_k1 = (2*lambda_i + lambda_j)/3;
    lambda_k2 = (lambda_i + 2*lambda_j)/3;

    rel_err = (lambda_j-lambda_i)/lambda_j;

    while rel_err > rel_err_thr
        KXX=exp(-(DD.*DD)/2/sig2_k);     % Gaussian kernel
        Z = pinv(KXX+lambda_k1*eye(size(KXX,2)))*KXX;
        labels_k1 = SpectralClusteringL(abs(Z)+abs(Z'),nc);

        Z = pinv(KXX+lambda_k2*eye(size(KXX,2)))*KXX;
        labels_k2 = SpectralClusteringL(abs(Z)+abs(Z'),nc);

        if hyper_flag == 1
            metric_ik1 = 1 - computeCE(labels_i,labels_k1);
            metric_k1k2 = 1 - computeCE(labels_k1,labels_k2);
            metric_k2j = 1 - computeCE(labels_k2,labels_j);
        elseif hyper_flag == 2
            metric_ik1 = compute_nmi(labels_i,labels_k1);
            metric_k1k2 = compute_nmi(labels_k1,labels_k2);
            metric_k2j = compute_nmi(labels_k2,labels_j);
        end

        labels_i = bestMap(labels_in,labels_i);
        labels_j = bestMap(labels_in,labels_j);
        labels_k1 = bestMap(labels_in,labels_k1);
        labels_k2 = bestMap(labels_in,labels_k2);

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

    if hyper_flag == 1
       KXX=exp(-(DD.*DD)/2/sig2_accmax(it));     % Gaussian kernel
       Z = pinv(KXX+lambda_accmax(it)*eye(size(KXX,2)))*KXX;
    elseif hyper_flag == 2
        KXX=exp(-(DD.*DD)/2/sig2_nmimax(it));     % Gaussian kernel
        Z = pinv(KXX+lambda_nmimax(it)*eye(size(KXX,2)))*KXX;        
    end
    
    labels_star=SpectralClusteringL(abs(Z)+abs(Z'),nc);
    labels_star = bestMap(labels_in,labels_star);
        
    ACC_star(it) = 1 - computeCE(labels_star,labels_in);
    NMI_star(it) = compute_nmi(labels_star,labels_in);
    F1_star(it) = compute_f(labels_star',labels_in);

    N_in = size(X_in,2);
    I_ONES = (eye(N_in)-ones(N_in,N_in)/N_in);
    K_uncentered=KXX;
    K = I_ONES*KXX*I_ONES;  % centering
    [U,LAM]=eig(K); R=N_in-1;
    tmp=diag(LAM); tmp=tmp(1:R); LAM=diag(tmp);
    U=U(:,1:R);
    Y=real(sqrt(LAM)*U');

    % Clustering out-of-sample data
    [B_star, begB_star, enddB_star, mu_star]  = bases_estimation(Y,labels_star,dimSubspace); % bases estimation

    KXX=exp(-(DD.*DD)/2/sig2_k);     % Gaussian kernel
    Z = pinv(KXX+lambda_k*eye(size(KXX,2)))*KXX;
    labels_est = SpectralClusteringL(abs(Z)+abs(Z'),nc);

    CE_est  = computeCE(labels_est,labels_in);
    ACC_est(it) = 1 - CE_est;
    NMI_est(it) = compute_nmi(labels_est,labels_in);
    F1_est(it) = compute_f(labels_est,labels_in');
    labels_est = bestMap(labels_in, labels_est);

    % OUT-OF-SAMPLE DATA
    % estimate bases labels estimated on in-sample data
    N_in = size(X_in,2);
    I_ONES = (eye(N_in)-ones(N_in,N_in)/N_in);
    K_uncentered=KXX;
    K = I_ONES*KXX*I_ONES;  % centering
    [U,LAM]=eig(K);
    tmp=diag(LAM); tmp=tmp(1:R); LAM=diag(tmp);
    U=U(:,1:R);
    Y=real(sqrt(LAM)*U');

    % Clustering out-of-sample data
    [B_y, begB_y, enddB_y, mu_Y]  = bases_estimation(Y,labels_est,dimSubspace); % bases estimation

    A0 = labels_out;
    N_out = size(X_out,2);
    Xtest=normc(X_out);
    DD = pdist2(X_in',Xtest');
    KX_out = exp(-(DD.*DD)/2/sig2_k);
    K_ones = K_uncentered*ones(N_in,1)/N_in;
    KXout = I_ONES*(KX_out-K_ones);
    dl=real(1./sqrt(diag(LAM)));
    Y_out = real(diag(dl)*U'*KXout);

    for l=1:nc
        Y_outm = Y_out - mu_Y(:,l);    % make data zero mean for distance calculation
        BB=B_y(:,begB_y(l):enddB_y(l));
        Yproj = (BB*BB')*Y_outm;
        Dproj = Y_outm - Yproj;
        D(l,:) = sqrt(sum(Dproj.^2,1));
    end
    [~, A_x] = min(D); % A_x is a label
    clear D

    % Performance on out-of-sample data
    A_x = bestMap(A0,A_x);
    ACC_out(it)  = 1 - computeCE(A_x,A0);
    NMI_out(it) = compute_nmi(A0,A_x);
    Fscore_out(it) = compute_f(A0',A_x);
    clear A_x

    for l=1:nc
        Y_outm = Y_out - mu_star(:,l);    % make data zero mean for distance calculation
        BB=B_star(:,begB_star(l):enddB_star(l));
        Yproj = (BB*BB')*Y_outm;
        Dproj = Y_outm - Yproj;
        D(l,:) = sqrt(sum(Dproj.^2,1));
    end
    [~, A_x] = min(D); % A_x is a label
    clear D

    % Performance on out-of-sample data
    A_x = bestMap(A0,A_x);
    ACC_out_star(it)  = 1 - computeCE(A_x,A0);
    NMI_out_star(it) = compute_nmi(A0,A_x);
    Fscore_out_star(it) = compute_f(A0',A_x);
    clear A_x

    save LFSG_kernel_LSR_ORL_metric_ACC.mat ACC_star ACC_est ACC_out NMI_star NMI_est NMI_out...
    F1_star F1_est Fscore_out ACC_out_star NMI_out_star Fscore_out_star ...
    sig2_est sig2_accmax sig2_nmimax sig2_f1max ...
    lambda_est lambda_accmax lambda_nmimax lambda_f1max 
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

display('***************** sig2 **********************')
display('MAXACC ORACLE:')
mean(sig2_accmax)
std(sig2_accmax)
display('MAXNMI ORACLE:')
mean(sig2_nmimax)
std(sig2_nmimax)
display('MAXF1 ORACLE:')
mean(sig2_f1max)
std(sig2_f1max)
display('sig2_EST:')
mean(sig2_est)
std(sig2_est)

% ranksum two sided Wilcoxon test of statistical significance
p_sig2_acc = ranksum(sig2_accmax,sig2_est)
p_sig2_nmi = ranksum(sig2_nmimax,sig2_est)
p_sig2_F1 = ranksum(sig2_f1max,sig2_est)

display('***************** lambda **********************')
display('MAXACC ORACLE:')
mean(lambda_accmax)
std(lambda_accmax)
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
p_lambda_acc = ranksum(lambda_accmax,lambda_est)
p_lambda_nmi = ranksum(lambda_nmimax,lambda_est)
p_lambda_F1 = ranksum(lambda_f1max,lambda_est)

save LFSG_kernel_LSR_ORL_metric_ACC.mat ACC_star ACC_est ACC_out NMI_star NMI_est NMI_out...
    F1_star F1_est Fscore_out ACC_out_star NMI_out_star Fscore_out_star ...
    sig2_est sig2_accmax sig2_nmimax sig2_f1max p_sig2_F1 p_sig2_nmi p_sig2_acc...
    lambda_est lambda_accmax lambda_nmimax lambda_f1max p_lambda_F1 p_lambda_nmi p_lambda_acc