cwd = fileparts(mfilename('fullpath'));
gemini_root = [cwd, filesep, '../../../GEMINI'];
addpath([gemini_root, filesep, 'script_utils'])
addpath([gemini_root, filesep, 'vis'])


%READ IN THE SIMULATION INFORMATION
ID='~/zettergmdata/simulations/input/GDI_periodic_highres_fileinput_large/';
xg=readgrid(ID);


%LOAD THE FRAME OF THE SIMULATION THAT WE WANT TO PERTURB
direc=ID;
filebase='GDI_periodic_highres_fileinput_large';
filename=[filebase,'_ICs.dat'];
[ne,v1,Ti,Te,ns,vs1,Ts,simdate]=loadframe3Dcurvnoelec(direc,filename);
lsp=size(ns,4);


%%GDI EXAMPLE (PERIODIC)
sigx2=20e3;
meanx3=0e3;
sigx3=20e3;
meanx2=-50e3;

scalefact=5;

for isp=1:lsp
  for ix2=1:xg.lx(2)
    amplitude=rand(xg.lx(1),1,xg.lx(3));
    amplitude=0.1*amplitude;
    nsperturb(:,ix2,:,isp)=ns(:,ix2,:,isp)+...                                           %original data
                8d0*ns(:,ix2,:,isp).*exp(-1d0*(xg.x2(2+ix2)-meanx2).^18/2d0/sigx2.^18);    %patch, note offset in the x2 index!!!!
    if (ix2>10 & ix2<xg.lx(2)-10)
      nsperturb(:,ix2,:,isp)=nsperturb(:,ix2,:,isp)+amplitude.*ns(:,ix2,:,isp);
    end                                    %noise
    nsperturb(:,ix2,:,isp)=scalefact*nsperturb(:,ix2,:,isp);
  end
end
nsperturb=max(nsperturb,1e4);


%WRITE OUT THE RESULTS TO A NEW FILE
outdir=ID;
dmy=[simdate(3),simdate(2),simdate(1)];
UTsec=simdate(4)*3600;
writedata(dmy,UTsec,nsperturb,vs1,Ts,outdir,[filebase,'_perturb']);

