%% LFSG_GraphFiltering_LSR
% I. Kopriva 2024-09

%%
clear
close all

% Set path to all subfolders
addpath(genpath('.'));

%% parameters
hyper_flag = 1; % 1 - max ACC criterion; 2 - nax NMI criterion; 3 - max F1 criterion

% Number of evaluations
numit =25;
rel_err_thr = 0.001;
epsilon = 1e-4; % stopping criterion for graph filtering method

%% Load the data from the chosen dataset

% Please uncomment the dataset that you want to use and comment the other ones
% dataName = 'YaleBCrop025';
% dataName = 'MNIST';
 dataName = 'USPS';
% dataName = 'ORL';
% dataName = 'COIL20'; 
% dataName = 'COIL100'; % Color images

% Performance metrics are saved in *.mat file named according to:
% save LFSG_GraohFiltering_LSR_dataset_metric_ACC_or_NMI (depends on the value of hyper_flag)

%% prepare data 
if strcmp(dataName,'YaleBCrop025')
   % Dataset dependent parameters
   dimSubspace=9;   % subspaces dimension
   numIn = 64;  % number of in-sample data
   ni = 64; % number of images per group
   nc = 38;  % number of groups
   % Graph-filtering LSR algorithm
   lambda = [1e-5 1e-4 1e-3 1e-2 1e-1 0.5];
   k = [1:10];
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
   numIn = 100;  % number of in-sample data
   nc = 10;   % number of groups
   % Graph-filtering LSR algorithm
   lambda = [1e-5 1e-4 1e-3 1e-2 1e-1 0.5];
   k = [1:10];
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
   numIn =  100;  % number of in-sample data
   nc = 10;    % number of groups
   % Graph-filtering LSR algorithm
   lambda = [1e-5 1e-4 1e-3 1e-2 1e-1 0.5];
   k = [1:10];
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
   numIn = 10;  % number of in-sample data
   nc = 40; % number of groups
   % Graph-filtering LSR algorithm
   lambda = [1e-5 1e-4 1e-3 1e-2 1e-1];
   k = [1:10];
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
   % Graph-filtering LSR algorithm
   lambda = [1e-5 1e-4 1e-3 1e-2 1e-1 0.5];
   k = [1:10];
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
   % Graph-filtering LSR algorithm
   lambda = [1e-5 1e-4 1e-3 1e-2 1e-1 0.5];
   k = [1:10];
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
    nIn = 1; 
    for c=1:nc % Through all categories
        ind = (labels == c); % Indices of the chosen category
        Xc = YY(:,ind);       % Samples ...
        numSamples = size(Xc, 2); % Number of samples ...
        ind = randperm(numSamples); % Random permutation of the indices
        X_in(:, nIn:nIn+numIn-1 ) = Xc(:, ind(1:numIn)); % Data
        labels_in(  nIn:nIn + numIn-1) = c; % Labels
        nIn  = nIn  + numIn; % Next indices
    end
    X_in(:,nIn:end) = []; % Cut out the surplus of the allocated space
    labels_in(nIn:end) = [];

    X_in = normc(X_in);
    N_in=size(X_in,2);

    clear CE NMI_o F1
    % Tuning over k for a fixed lambda
    lambda_tune = lambda(ceil(length(lambda)/2));    
    for i=1:length(k)
        error = 1.1*epsilon;
        W = zeros(N_in,N_in); %eye(N_in);
        X_bar = X_in;
        while error > epsilon
            XbarTXbar=transpose(X_bar)*X_bar;
            Z = pinv(XbarTXbar + lambda_tune*eye(N_in))*XbarTXbar;
            W_1 = W;
            W = (abs(Z) + transpose(abs(Z)))/2;
            dd = sum(W,2);
            D = diag(1./sqrt(dd));
            L = eye(N_in) - D*W*D;
            X_bar = transpose(power(eye(N_in)-L/2,k(i))*transpose(X_in));
            error = norm(W-W_1,'fro');
        end
        A(i,:) = SpectralClusteringL(W,nc);
        A(i,:) = bestMap(labels_in,A(i,:));
        CE(i)  = computeCE(A(i,:),labels_in);
        NMI_o(i) = compute_nmi(A(i,:),labels_in);
        F1(i) = compute_f(A(i,:),labels_in);
    end

    [cemin imin] = min(CE);
    k_accmax(it) = k(imin);

    [nmimax imax] = max(NMI_o);
    k_nmimax(it)= k(imax);

    [f1max imax] = max(F1);
    k_f1max(it)= k(imax);
    
    ACC = zeros(length(k),length(k));
    NMI = zeros(length(k),length(k));
    Fscore = zeros(length(k),length(k));

    for i=1:length(k)-1
        for j=i+1:length(k)
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

    k_i = k(imax(jmax));
    labels_i = A(imax(jmax),:);
    k_j = k(jmax);
    labels_j = A(jmax,:);

    k_k1 = round((2*k_i + k_j)/3);
    k_k2 = round((k_i * 2*k_j)/3);

    rel_err = (k_j-k_i)/k_j;

    while rel_err > rel_err_thr
        X_bar = X_in;
        error = 1.1*epsilon;
        W = eye(N_in);
        while error > epsilon
            XbarTXbar=transpose(X_bar)*X_bar;
            Z = pinv(XbarTXbar + lambda_tune*eye(N_in))*XbarTXbar;
            W_1 = W;
            W = (abs(Z) + transpose(abs(Z)))/2;
            dd = sum(W,2);
            D = diag(1./sqrt(dd));
            L = eye(N_in) - D*W*D;
            X_bar = transpose(power(eye(N_in)-L/2,k_k1)*transpose(X_in));
            error = norm(W-W_1,'fro');
        end
        labels_k1 = SpectralClusteringL(W,nc);

        X_bar = X_in;
        error = 1.1*epsilon;
        W = eye(N_in);
        while error > epsilon
            XbarTXbar=transpose(X_bar)*X_bar;
            Z = pinv(XbarTXbar + lambda_tune*eye(N_in))*XbarTXbar;
            W_1 = W;
            W = (abs(Z) + transpose(abs(Z)))/2;
            dd = sum(W,2);
            D = diag(1./sqrt(dd));
            L = eye(N_in) - D*W*D;
            X_bar = transpose(power(eye(N_in)-L/2,k_k2)*transpose(X_in));
            error = norm(W-W_1,'fro');
        end
        labels_k2 = SpectralClusteringL(W,nc);

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
            k_j = k_k1;
            labels_j = labels_k1;
            k_k1 = round((2*k_i + k_j)/3);
            k_k2 = round((k_i + 2*k_j)/3);
            rel_err=(k_k1-k_i)/k_k1;
        elseif metric_k1k2 >= metric_k2j
            k_i = k_k1;
            k_j = k_k2;
            labels_i = labels_k1;
            labels_j = labels_k2;
            k_k1 = round((2*k_i + k_j)/3);
            k_k2 = round((k_i + 2*k_j)/3);
            rel_err=(k_k2-k_k1)/k_k2;
        else
            k_i = k_k2;
            labels_i = labels_k2;
            k_k1 = round((2*k_i + k_j)/3);
            k_k2 = round((k_i + 2*k_j)/3);
            rel_err=(k_j-k_k2)/k_j;
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
        k_k = k_k1;
    else
        k_k = k_k2;
    end

    k_est(it)=k_k;

    clear CE NMI_o F1
    % tuning over lambda for a fixed "k"
    for i=1:length(lambda)
        if hyper_flag == 1
            k_star = k_accmax(it);
        elseif hyper_flag == 2
            k_star = k_nmimax(it);
        end

        X_bar = X_in;
        error = 1.1*epsilon;
        W = eye(N_in);
        while error > epsilon
            XbarTXbar=transpose(X_bar)*X_bar;
            Z = pinv(XbarTXbar + lambda(i)*eye(N_in))*XbarTXbar;
            W_1 = W;
            W = (abs(Z) + transpose(abs(Z)))/2;
            dd = sum(W,2);
            D = diag(1./sqrt(dd));
            L = eye(N_in) - D*W*D;
            X_bar = transpose(power(eye(N_in)-L/2,k_star)*transpose(X_in));
            error = norm(W-W_1,'fro');
        end

        A(i,:) = SpectralClusteringL(W,nc);
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
        X_bar = X_in;
        error = 1.1*epsilon;
        W = eye(N_in);
        while error > epsilon
            XbarTXbar=transpose(X_bar)*X_bar;
            Z = pinv(XbarTXbar + lambda_k1*eye(N_in))*XbarTXbar;
            W_1 = W;
            W = (abs(Z) + transpose(abs(Z)))/2;
            dd = sum(W,2);
            D = diag(1./sqrt(dd));
            L = eye(N_in) - D*W*D;
            X_bar = transpose(power(eye(N_in)-L/2,k_k)*transpose(X_in));
            error = norm(W-W_1,'fro');
        end
        labels_k1 = SpectralClusteringL(W,nc);

        X_bar = X_in;
        error = 1.1*epsilon;
        W = eye(N_in);
        while error > epsilon
            XbarTXbar=transpose(X_bar)*X_bar;
            Z = pinv(XbarTXbar + lambda_k2*eye(N_in))*XbarTXbar;
            W_1 = W;
            W = (abs(Z) + transpose(abs(Z)))/2;
            dd = sum(W,2);
            D = diag(1./sqrt(dd));
            L = eye(N_in) - D*W*D;
            X_bar = transpose(power(eye(N_in)-L/2,k_k)*transpose(X_in));
            error = norm(W-W_1,'fro');
        end
        labels_k2 = SpectralClusteringL(W,nc);

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
        X_bar = X_in;
        error = 1.1*epsilon;
        W = eye(N_in);
        while error > epsilon
            XbarTXbar=transpose(X_bar)*X_bar;
            Z = pinv(XbarTXbar + lambda_accmax(it)*eye(N_in))*XbarTXbar;
            W_1 = W;
            W = (abs(Z) + transpose(abs(Z)))/2;
            dd = sum(W,2);
            D = diag(1./sqrt(dd));
            L = eye(N_in) - D*W*D;
            X_bar = transpose(power(eye(N_in)-L/2,k_accmax(it))*transpose(X_in));
            error = norm(W-W_1,'fro');
        end
    elseif hyper_flag == 2 
        X_bar = X_in;
        error = 1.1*epsilon;
        W = eye(N_in);
        while error > epsilon
            XbarTXbar=transpose(X_bar)*X_bar;
            Z = pinv(XbarTXbar + lambda_nmimax(it)*eye(N_in))*XbarTXbar;
            W_1 = W;
            W = (abs(Z) + transpose(abs(Z)))/2;
            dd = sum(W,2);
            D = diag(1./sqrt(dd));
            L = eye(N_in) - D*W*D;
            X_bar = transpose(power(eye(N_in)-L/2,k_nmimax(it))*transpose(X_in));
            error = norm(W-W_1,'fro');
        end
    end
    
    labels_star=SpectralClusteringL(W,nc);
    labels_star = bestMap(labels_in,labels_star);
        
    ACC_star(it) = 1 - computeCE(labels_star,labels_in);
    NMI_star(it) = compute_nmi(labels_star,labels_in);
    F1_star(it) = compute_f(labels_star',labels_in);

    X_bar = X_in;
    error = 1.1*epsilon;
    W = eye(N_in);
    while error > epsilon
        XbarTXbar=transpose(X_bar)*X_bar;
        Z = pinv(XbarTXbar + lambda_k*eye(N_in))*XbarTXbar;
        W_1 = W;
        W = (abs(Z) + transpose(abs(Z)))/2;
        dd = sum(W,2);
        D = diag(1./sqrt(dd));
        L = eye(N_in) - D*W*D;
        X_bar = transpose(power(eye(N_in)-L/2,k_k)*transpose(X_in));
        error = norm(W-W_1,'fro');
    end
    labels_est = SpectralClusteringL(W,nc);
    labels_est = bestMap(labels_in,labels_est);

    CE_est  = computeCE(labels_est,labels_in);
    ACC_est(it) = 1 - CE_est;
    NMI_est(it) = compute_nmi(labels_est,labels_in);
    F1_est(it) = compute_f(labels_est,labels_in');

   save LFSG_GraphFiltering_LSR_USPS_metric_ACC.mat ACC_star ACC_est NMI_star NMI_est F1_star F1_est ...
   k_est k_accmax k_nmimax k_f1max lambda_est lambda_accmax lambda_nmimax lambda_f1max 
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

% ranksum two sided Wilcoxon test of statistical significance
p_acc_in = ranksum(ACC_star,ACC_est)

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

% ranksum two sided Wilcoxon test of statistical significance
p_nmi_in = ranksum(NMI_star,NMI_est)

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

% ranksum two sided Wilcoxon test of statistical significance
p_F1_in = ranksum(F1_star,F1_est)

display('***************** "k" **********************')
display('MAXACC ORACLE:')
mean(k_accmax)
std(k_accmax)
display('MAXNMI ORACLE:')
mean(k_nmimax)
std(k_nmimax)
display('MAXF1 ORACLE:')
mean(k_f1max)
std(k_f1max)
display('k_EST:')
mean(k_est)
std(k_est)

% ranksum two sided Wilcoxon test of statistical significance
p_k_acc = ranksum(k_accmax,k_est)
p_k_nmi = ranksum(k_nmimax,k_est)
p_k_F1 = ranksum(k_f1max,k_est)

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

save LFSG_GraphFiltering_LSR_USPS_metric_ACC.mat ACC_star ACC_est NMI_star NMI_est F1_star F1_est ...
    k_est k_accmax k_nmimax k_f1max p_k_F1 p_k_nmi p_k_acc...
    lambda_est lambda_accmax lambda_nmimax lambda_f1max p_lambda_F1 p_lambda_nmi p_lambda_acc

