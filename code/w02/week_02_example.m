% go to directory 
cd('/Users/boris/Documents/1_github/Fall2020/code/w02')
myFile = 'ts.mat';
mySurf = 'SM.mat';
myYeo  = 'yeo.mat'; 
myMarg = 'G1.mat';
myMyel = 'myelin.mat'

load(mySurf); 
load(myFile);
load(myYeo);
load(myMarg); 
load(myMyel); 


%% --   
% 1) build correlation matrix = a functional connectome
r               = corr(ts);
z               = 0.5 * log( (1+r) ./ (1-r) ); 
z(isinf(z))     = 0; 
z(isnan(z))     = 0;
    
% display = looks unstructured
f=figure, 
    imagesc(z,[0 1]), 

%% --     
% 2) sort by communities 
% display parcellations/modules 
f=figure, surf_viewer(yeo, SM,'')

[sy, sindex] = sort(yeo); 
f=figure, 
    imagesc(z(sindex,sindex),[0 1])
 
    
%% -- 
% 3) SCA     
%    
PCC             = 3215; 
VIS             = 2748; 

f=figure;  
    surf_viewer(z(VIS,:),SM,'visual', [0 .7])
    colormap(parula);
   
f=figure;  
    surf_viewer(z(PCC,:),SM,'PCC', [0 .7])
    colormap(hot);
    
    
%% -- 
% 4) load gradient & myelin 
%
f=figure, surf_viewer(G1, SM,'')
f=figure, surf_viewer(myelin, SM,'',[1.5 2])


