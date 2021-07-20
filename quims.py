import glob
import os as os
import astropy as ap
import numpy as np
import scipy as sp
import astropy.io.fits as aio
from astropy.wcs import WCS
from astropy import units as u
import matplotlib.pyplot as pl
import scipy.odr as odr
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.transforms import Bbox
from matplotlib.patches import Rectangle

# primary function for fitting
def mainFitting(fileAddress,fileOut=None,xCrop = None,yCrop = None,maskSig=5,sizeFlagLow=0,sizeFlagUp=5):
    '''
    
    Extracts a mask of peaks in the input image and uses Orthogonal-Distance-Regression to assign the mask
    a best-fit circle, effectively tracing the caustic curve for many sources with obvious evidence of lensing.
    
    Parameters
    ----------
    fileAddress : string
        The .fits image of a lensed source with evidence of lensing.
    
    fileOut : string
        The name of a folder which output images are saved to. By default, images are saved in the current directory.
    
    xCrop : dict
        A dict of integers specifying lower and upper limits of cropping along the x-axis.
    
    yCrop : dict
        A dict of integers specifying lower and upper limits of cropping along the y-axis.
        
    maskSig : float
        Standard-deviations used to binary-mask. By default, this is 5.
    
    sizeFlagLow : float
        Minimum radius of best-fit circle, if smaller the source is flagged for manual review.
    
    sizeFlagUp : float
        Maximum radius of best-fit circle, if larger the source is flagged for manual review.
    
    Returns
    -------
    odrOut : Output
        SciPy.odr Output containing fit-parameters. If flagged for manual-review, this is set to 'NONE'.
    
    pixelSize : float
        Resolution of fits-images in arcseconds-per-pixel.
    
    xCropLow : float
        Lower-bound for cropping along x-axis. Used to convert from cropped to raw image coordinates.
        If flaggged for manual-review, this is set to 'NONE'.
    
    yCropLow : float
        Lower-bound for cropping along y-axis. Used to convert from cropped to raw image coordinates.
        If flagged for manual-review, this is set to 'NONE'.
    
    sourceName : string
        Name of the source, by default this is assigned from the center-coordinate of the image. Given a
        successful fit, the sourceName is updated to the center-point of the best-fit circle.
    
    Notes
    -----
        This was tested on continuum images of lensed sources extracted from the ALMA Science 
        Archive. In princible, it can be applied to continuum images of lensed sources taken by other 
        instruments and saved in standard .fits format, but this has not been fully explored. Images saved
        include input .fits with crop-region highlighted, cropped image with best-fit circle overlayed, 
        and binary-mask with best-fit circle overlayed.
        
    '''
    
    if fileOut == None:
        fileOut = ''
    else:
        fileOut = fileOut + '/'
    
    # --pulling data-- #
    # opening file and getting data from .fits
    cuteRingHDU = aio.open(fileAddress,ignore_missing_end=True);
    data = cuteRingHDU[0].data[0,0,:,:]
    
    # image center coordinates
    centCoord = [int(len(data[0,:])/2),int(len(data[:,0])/2)]
    wcs = WCS(cuteRingHDU[0].header);
    c = wcs.pixel_to_world(centCoord[0],centCoord[1],0,0)[0];
    c.to_string('hmsdms');
    
    # default name to image center, corrected later given successful fit
    sourceName = 'J{0}{1}'.format(c.ra.to_string(unit=u.hourangle, sep='', precision=1, pad=True), c.dec.to_string(sep='', precision=0, alwayssign=True, pad=True))
    pixelSize = np.abs(cuteRingHDU[0].header['CDELT1'])*3600
    # -- #
    
    # --cropping and mask-- #
    if xCrop == None:
        xCropUp = 3*int(len(data[0,:])/4); xCropLow = int(len(data[0,:])/4)
    else:
        xCropUp = xCrop[1]; xCropLow = xCrop[0]
    
    if yCrop == None:
        yCropUp = 3*int(len(data[:,0])/4); yCropLow = int(len(data[:,0])/4)
    else:
        yCropUp = yCrop[1]; yCropLow = yCrop[0]
    dataCrop = data[xCropLow:xCropUp,yCropLow:yCropUp]
    
    # mask
    binMask = dataCrop > maskSig*np.nanstd(data.flatten())
    
    cuteRingHDU.close()
    
    # flag if no-peaks
    if np.sum(binMask) == 0:
        return None,pixelSize,None,None,sourceName
    # -- #
    
    # --Input Points-- #
    points = np.zeros((2,np.sum(binMask)))
    dex = 0
    for i in range(len(binMask[0,:])):
        for j in range(len(binMask[:,0])):
            if binMask[i,j]:
                points[0,dex] = i
                points[1,dex] = j
                dex+=1
    # -- #
    
    # --ODR-- #
    def circ(B,x):
            return (x[0] - B[0])**2 + (x[1] - B[1])**2 - B[2]**2
    odrCirc = odr.Model(circ,implicit=True);
    odrData = odr.Data(np.row_stack([points[0,:], points[1,:]]),y=1);
    B0 = [np.mean(points[0,:]),np.mean(points[1,:]),10];
    myODR = odr.ODR(odrData,odrCirc,beta0=B0);
    odrOut = myODR.run();
    out = odrOut.beta
    # -- #
    
    # --post-fit flagging-- #
    
    # threshold for flagging large/small sources, default >5'' flagged
    if not (sizeFlagLow < out[2]*pixelSize < sizeFlagUp):
        return None,pixelSize,None,None,sourceName
    
    # circle center outside cropped-region flagged
    if (not (0 < out[1] < len(dataCrop[0,:]))) or (not (0 < out[0] < len(dataCrop[:,0]))):
        return None,pixelSize,None,None,sourceName
    # -- #
    
    # -- tweak source-name to best-fit circle center -- #
    centCoord[0] = int(out[1]+xCropLow)
    centCoord[1] = int(out[0]+yCropLow)
    c = wcs.pixel_to_world(centCoord[0],centCoord[1],0,0)[0];
    c.to_string('hmsdms');
    sourceName = 'J{0}{1}'.format(c.ra.to_string(unit=u.hourangle, sep='', precision=1, pad=True), c.dec.to_string(sep='', precision=0, alwayssign=True, pad=True))
    # -- #
    
    # -- plotting image & circle -- #
    fig, ax = pl.subplots()
    pl.imshow(data,origin='upper')
    c = pl.Circle((out[1]+xCropLow, out[0]+yCropLow), out[2], linewidth=1, ls='--',fill=False,alpha=0.5,color='black')
    ax.add_patch(c)
    radius = int(out[2])
    center = [int(out[1]+xCropLow),int(out[0]+yCropLow)]
    pl.xlim((center[0] - 3*radius,center[0] + 3*radius))
    pl.ylim((center[1] - 3*radius,center[1] + 3*radius))
    pl.text(0.05,0.92,sourceName,c='white',transform=ax.transAxes,fontsize=12)
    dataCoordBar = 1/pixelSize
    asb = AnchoredSizeBar(ax.transData,dataCoordBar,"$ 1'' $",loc='lower right',frameon=False,color='white')
    ax.add_artist(asb)
    xTickStart = center[0]-2*radius
    yTickStart = center[1]-2*radius
    pl.xticks(np.arange(xTickStart,xTickStart + 6*radius,int(dataCoordBar)),labels='');
    pl.yticks(np.arange(yTickStart,yTickStart + 6*radius,int(dataCoordBar)),labels='');
    pl.savefig(fileOut + sourceName + '_imOverlay.tiff',dpi=420,bbox_inches='tight')
    pl.close()
    # -- #
    
    # --plotting binaryMask & circle-- #
    fig, ax = pl.subplots()
    pl.imshow(data > maskSig*np.nanstd(data.flatten()),origin='upper')
    c = pl.Circle((out[1]+xCropLow, out[0]+yCropLow), out[2], linewidth=1, ls='--',fill=False,alpha=0.5,color='black')
    ax.add_patch(c)
    radius = int(out[2])
    center = [int(out[1])+xCropLow,int(out[0])+yCropLow]
    pl.xlim((center[0] - 3*radius,center[0] + 3*radius))
    pl.ylim((center[1] - 3*radius,center[1] + 3*radius))
    pl.text(0.05,0.92,sourceName,c='white',transform=ax.transAxes,fontsize=12)
    dataCoordBar = 1/pixelSize
    asb = AnchoredSizeBar(ax.transData,dataCoordBar,"$ 1'' $",loc='lower right',frameon=False,color='white')
    ax.add_artist(asb)
    xTickStart = center[0]-2*radius
    yTickStart = center[1]-2*radius
    pl.xticks(np.arange(xTickStart,xTickStart + 6*radius,int(dataCoordBar)),labels='');
    pl.yticks(np.arange(yTickStart,yTickStart + 6*radius,int(dataCoordBar)),labels='');
    pl.savefig(fileOut + sourceName + '_5sigOverlay.tiff',dpi=420,bbox_inches='tight')
    pl.close()
    # -- #
    
    # --plotting input image and cropping-box-- #
    fig, ax = pl.subplots()
    pl.imshow(data)
    rect = Rectangle((xCropLow,yCropLow),xCropUp-xCropLow,yCropUp-yCropLow,fill=False,ls='--',color='red')
    ax.add_patch(rect)
    pl.savefig(fileOut + sourceName + '_image.tiff',dpi=420,bbox_inches='tight')
    pl.close()
    # -- #
    
    # return odrOut, resolution, and source-name
    return odrOut,pixelSize,xCropLow,yCropLow,sourceName


# I/O
def fitFolder(inputFolder='images',outputFolder='images_product',):
    '''
    
    A convenience function which runs mainFunction() on images in inputFolder and outputs the resulting plots into
    inputFolder. An array including filenames, sourcenames, and fit-parameters is returned. Additionally, 
    a comma-delimited text file of these parameters, outputArr.txt, is created and saved.
    
    Parameters
    ----------
    inputFolder : string
        Folder containing continuum images of lensed sources for fitting. By default, this is a folder
        called 'images' in the current directory.
    
    outputFolder : string
        Folder where images produced by mainFitting() are saved. If the specified folder does not exist,
        it is created. Defaults to 'images_product'.
          
    Returns
    -------
    outputArr : ndarray
        An array of objects containing the filename, sourcename, best-fit center (x,y) in pixel-coordinates, 
        best-fit radius in arcseconds, errors following each parameter (i.e. [...,x,sigmax,y,sigmay,r,sigmar,...]), 
        reduced chi-square, and a flag for manual review. If fitting fails, source is flagged as 'MANUAL' and
        fit-parameters are set to 'NONE'.
        
    Notes
    -----
        Make sure inputFolder only contains continuum images of lensed sources.
        
    '''
    
    # throw error if there is no input directory
    if not os.path.exists(inputFolder):
        raise FileNotFoundError('The input directory was not found')
    
    filesToFit = glob.glob(os.path.join(inputFolder, '*.fits'))
    
    # throw error if there are no images in the directory
    if len(filesToFit) == 0:
        raise FileNotFoundError('There are no .fits files in the specified directory')

    # make output folder for images
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    
    outputArr = np.zeros((len(filesToFit),10),dtype='O')

    for i in range(len(filesToFit)):
        # print(i) for debugging
        outputArr[i,0] = filesToFit[i]
        fitResult,pixelSize,xCrop,yCrop,sourceName = mainFitting(filesToFit[i],outputFolder)
        outputArr[i,1] = sourceName
        
        # flagging sources
        if fitResult == None:
            outputArr[i,2:] = np.empty(len(outputArr[i,2:]),dtype='O')
            outputArr[i,-1] = 'MANUAL'
            continue
        
        # rounding
        betaOut = np.around(fitResult.beta,decimals = 2)
        betaSTD = np.around(fitResult.sd_beta,decimals = 2)
        # center coords & errors (uncropped pixel-coordinates)
        outputArr[i,2] = np.around(betaOut[1]+xCrop,decimals=2)
        outputArr[i,3] = betaSTD[1]
        
        outputArr[i,4] = np.around(betaOut[0]+yCrop,decimals=2)
        outputArr[i,5] = betaSTD[0]
        
        # image separation & error in radians
        outputArr[i,6] = round(betaOut[2]*pixelSize,3)
        outputArr[i,7] = round(betaSTD[2]*pixelSize,3)
        
        # reduced chi-square (for reference, not to be relied on heavily)
        outputArr[i,8] = round(fitResult.res_var,2)
        
        # flag
        outputArr[i,9] = 'OKAY'
        
        # save output array as .csv
    
    np.savetxt('outputArr.txt',outputArr,fmt='%s',delimiter=',')
        
    return outputArr
