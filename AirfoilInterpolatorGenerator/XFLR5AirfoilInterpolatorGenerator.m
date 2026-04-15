close all; clear all; clc;

%% Inputs
cfdtable = "CFD TABLE";
analytfileloca = "ANAYTIC FILE FOLDER";
smoothingparam = "SMOOTHING PARAM";
debug = true;


%% Get the post stall data from the airfoil from the CFD data:
analytfiles = dir(analytfileloca);
analytfiles = analytfiles(~[analytfiles.isdir]); % eliminate non file files from the list

% Get the data from the CFD files: 
CLCDtable = readtable(cfdtable);
mach = CLCDtable.Mach(:);
reynolds = CLCDtable.Reynolds(:);
alpha = CLCDtable.Alpha(:);
dragforce = CLCDtable.DragForce(:);
liftforce = CLCDtable.LiftForce(:);
velocity = CLCDtable.InletVelocity(:);

% Calculate the post stall lift and drag coefficients
post_stall_cl = liftforce./(0.5*1.225*velocity.^2*0.2);
post_stall_cd = dragforce./(0.5*1.225*velocity.^2*0.2);


% Create an interpolant (a function) that we can evaluate for post
% stall data:
PSCL_interpolant = scatteredInterpolant(alpha,reynolds,mach,post_stall_cl);
PSCD_interpolant = scatteredInterpolant(alpha,reynolds,mach,post_stall_cd);


% Clear up some memory
clear mach reynolds alpha dragforce liftforce velocity post_stall_cl post_stall_cd cfdfiles


% Pre allocate space for all of the extrapolated data
fitalphas = -90:0.5:90;
fitdatacls = zeros(length(fitalphas), length(analytfiles));
fitdatacds = zeros(length(fitalphas), length(analytfiles));
fitdataRE = zeros(length(analytfiles), 1);
fitdataMA = zeros(length(analytfiles), 1);

if debug
    figure('name', "Airfoil data and curve fits")
end

for ff = 1:length(analytfiles)
    % Get the data from the airfoil (analytical)
    foildata = importdata(strcat(analytfileloca, analytfiles(ff).name));
    
    % extract the data from the file
    foilalphas = foildata.data(:, 1);
    foilCls = foildata.data(:, 2);
    foilCds = foildata.data(:, 3);
    
    % determine the zero lift angle of attack
    % [~, Cl_a0] = min(abs(foilCls));
    % Cl_a0 = foilalphas(Cl_a0);
    
    % if debug
    % subplot(2, 1, 1)
    % hold off
    % plot(foilalphas, foilCls, '*', foilalphas, foilCds, '*')
    % xlabel("alpha")
    % legend("Cl", "Cd")
    % subplot(2, 1, 2)
    % hold off
    % plot(foilalphas, foilCls./foilCds, '*')
    % xlabel("alpha")
    % legend("Cl / Cd")
    % 
    % % extend the analytical data from foilalphas to fitalphas
    % % First, extend the cl data from the stall point to the end of the
    % % range fitalphas
    % end

    fitCls = zeros(size(fitalphas));
    fitCds = zeros(size(fitalphas));


    % Find the index corresponding to the last analytical data point in
    % the fitted domain:
    [~, fit_clst] = min(abs(fitalphas - foilalphas(end)));

    % Find the index corresponding to the first analytical data point in
    % the fitted domain:
    [~, fit_clft] = min(abs(fitalphas - foilalphas(1)));

    % populate the fitted data with the analytical data we have first
    for index = fit_clft:1:fit_clst
        fitCls(index) = interp1(foilalphas, foilCls, fitalphas(index));
        fitCds(index) = interp1(foilalphas, foilCds, fitalphas(index));
    end

    % Get the current mach and reynolds number of the analytical data
    mIndex = strfind(analytfiles(ff).name, "_M");
    mIndex = mIndex(end); % Get the last _m in the name (will always be mach number)
    rIndex = strfind(analytfiles(ff).name, "_Re");
    rIndex = rIndex(end); % Get the last _Re in the name (will always be mach number)

    fitdataMA(ff) = str2double(analytfiles(ff).name(mIndex+2:mIndex+5));
    fitdataRE(ff) = str2double(analytfiles(ff).name(rIndex+3:rIndex+7))*1000000;


    % Populate the rest of the data with data evaluated from
    % interpolating the CFD data
    for index = fit_clst+1:1:length(fitalphas)
        fitCls(index) = PSCL_interpolant(fitalphas(index), fitdataRE(ff), fitdataMA(ff));
        fitCds(index) = PSCD_interpolant(fitalphas(index), fitdataRE(ff), fitdataMA(ff));
    end

    % Blend the data points at the end to make everything smooth
    for index = fit_clst-5:1:fit_clst
        fitCls(index) = fitCls(index)*(fit_clst - index)/5 + PSCL_interpolant(fitalphas(index), fitdataRE(ff), fitdataMA(ff))*(1-(fit_clst - index)/5);
        fitCds(index) = fitCds(index)*(fit_clst - index)/5 + PSCD_interpolant(fitalphas(index), fitdataRE(ff), fitdataMA(ff))*(1-(fit_clst - index)/5);
    end

    % Fill in the data on the LHS of the zero lift AOA
    for index = fit_clft-1:-1:1
        % The CL in the negative AoA range should be the negative of
        % the positive aoa range CL
        fitCls(index) = -PSCL_interpolant(-fitalphas(index), fitdataRE(ff), fitdataMA(ff));

        % The CD in the negative AoA range should be the same in
        % magnitude as its positive AoA range partner
        fitCds(index) = PSCD_interpolant(-fitalphas(index), fitdataRE(ff), fitdataMA(ff));
    end

    % To blend the CL data on the LHS we offset it to match the
    % analytical cl data
    fitCls(1:fit_clft-1) = fitCls(1:fit_clft-1) + (fitCls(fit_clft) - fitCls(fit_clft-1));


    % Impose limits on the values to prevent bad fittings
    % Limit on CL as alpha approaches plus or minus 90 is zero
    fitCls(1) = 0;
    fitCls(end) = 0;
    % Limit on CL as alpha approaches plus or minus 90 is one
    fitCds(1) = 1;
    fitCds(end) = 1;
    % CL at alpha = 45 degrees should be approx 1
    [~, alf45] = min(abs(fitalphas - 45));
    fitCls(alf45) = 1.15;


    % if debug
    % % plot the data for debugging if desired
    % subplot(2, 1, 1)
    % hold off
    % plot(fitalphas, fitCls, '*', fitalphas, fitCds, '*')
    % xlabel("alpha")
    % legend("Cl", "Cd")
    % xline(0, 'k')
    % yline(0, 'k')
    % end


    % Curve fit our data and smooth it
    foilClsfit = fit(fitalphas.', fitCls.', 'smoothingspline', 'SmoothingParam', smoothingparam, 'Exclude', isnan(fitCls));
    foilCls = feval(foilClsfit, fitalphas);
    
    foilCdsfit = fit(fitalphas.', fitCds.', 'smoothingspline', 'SmoothingParam', smoothingparam, 'Exclude', isnan(fitCds));
    foilCds = feval(foilCdsfit, fitalphas);

    if debug
    % plot the data for debugging if desired
    hold on
    plot(fitalphas, foilCls)
    xlabel("alpha")
    legend("Cl", "Cd")
    xline(0, 'k')
    yline(0, 'k')
    ylim([-3, 3])
    xlim([-5, 15])
    drawnow
    end

    % Store the data at this reynolds/mach
    fitdatacls(:, ff) = foilCls;
    fitdatacds(:, ff) = foilCds;
end


% Need to create 5 column vectors, one for each of CL, CD, alpha,
% reynolds and mach, where each row corresponds to a unique point in
% the provided data above

% Fitalphas get repeated for every set of points
fitalphas = repmat(fitalphas, length(analytfiles), 1);

% Re and Ma also get repeated for every point, but, in the other
% direction

% Need to now interpolate in 3D
fitdataRE = repmat(fitdataRE, 1, length(foilCls));
fitdataMA = repmat(fitdataMA, 1, length(foilCls));

% Somehow i have all these matrices transposed, so sneaky transpose to
% correct ordering
fitalphas = fitalphas.';
fitdataRE = fitdataRE.';
fitdataMA = fitdataMA.';

% Reshape all of these matrices into vectors
veclength = numel(fitdatacls);
fitdatacls = reshape(fitdatacls, veclength, 1);
fitdatacds = reshape(fitdatacds, veclength, 1);
fitalphas = reshape(fitalphas, veclength, 1);
fitdataRE = reshape(fitdataRE, veclength, 1);
fitdataMA = reshape(fitdataMA, veclength, 1);

% Use the interpolant to create a dense grid of data

% interpolate our data into the grid

% create our starting grid:
[newalf, newrey, newmac] = meshgrid(unique(fitalphas), unique(fitdataRE), unique(fitdataMA));

% interpolate our data into the grid
gridded_cl = griddata(fitalphas, fitdataRE, fitdataMA, fitdatacls, newalf, newrey, newmac, "nearest");
gridded_cd = griddata(fitalphas, fitdataRE, fitdataMA, fitdatacds, newalf, newrey, newmac, "nearest");

% create a scattered interpolant of the available data (for some slower
% calculation)
cl_scat = scatteredInterpolant(fitalphas, fitdataRE, fitdataMA, fitdatacls, "natural");
cd_scat = scatteredInterpolant(fitalphas, fitdataRE, fitdataMA, fitdatacds, "natural");

% Fill in the nan values where they exist
nanvalues = isnan(gridded_cl) & isnan(gridded_cd);
gridded_cl(nanvalues) = cl_scat(newalf(nanvalues), newrey(nanvalues), newmac(nanvalues));
gridded_cd(nanvalues) = cd_scat(newalf(nanvalues), newrey(nanvalues), newmac(nanvalues));

% contourslice(newalf, newrey, newmac, gridded_cl, [], [], [0.05, 0.25, 0.45], [-1:0.05:2])
% xlabel("alpha"); ylabel("reynolds"); zlabel("mach");
% view(3);
% xlim([-5, 20]);

% Convert from meshgrid to ndgrid format (fml)
P = [2 1 3];
newalf = permute(newalf, P);
newrey = permute(newrey, P);
newmac = permute(newmac, P);
gridded_cl = permute(gridded_cl, P);
gridded_cd = permute(gridded_cd, P);

% Use the gridded data to create an efficient interpolant
cl = griddedInterpolant(deg2rad(newalf),newrey,newmac,gridded_cl, 'makima');
cd = griddedInterpolant(deg2rad(newalf),newrey,newmac,gridded_cd, 'makima');

