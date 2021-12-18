from netCDF4 import Dataset
import glob
import numpy as np
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
from scipy.optimize import minimize as scipyminimize
from scipy.optimize import Bounds
from pyhdf.SD import SD, SDC
from scipy import ndimage as nd
import h5py
from scipy.interpolate import griddata,Rbf
from scipy.interpolate import interp1d
from lmfit import Model



def interlin2d(x,y,z,fsize):
    """
    Linear 2D interpolation of a plane from arbitrary gridded points.

    :param x: 2D array of x coordinates
    :param y: 2D array of y coordinates
    :param z: 2D array of z coordinates
    :param fsize: Tuple of x and y dimensions of plane to be interpolated.
    :return: 2D array with interpolated plane.

    This function works by interpolating lines along the grid point in both dimensions,
    then interpolating the plane area in both the x and y directions, and taking the
    average of the two. Result looks like a series of approximately curvilinear quadrilaterals.

    Note, the structure of the x,y,z coordinate arrays are such that the index of the coordinates
    indicates the relative physical position of the point with respect to the plane to be interpoalted.

    Plane is allowed to be a subset of the range of grid coordinates provided.
    Extrapolation is accounted for, however sharp creases will start to appear
    in the extrapolated region as the grid of coordinates becomes increasingly irregular.

    Scipy's interpolation function is used for the grid lines as it allows for proper linear extrapolation.
    However Numpy's interpolation function is used for the plane itself as it is robust against gridlines
    that overlap (divide by zero distance).


    Example:
    #set up number of grid lines and size of field to interpolate
    nlines=[3,3]
    fsize=(100,100,100)

    #initialize the coordinate arrays
    x=np.empty((nlines[0],nlines[1]))
    y=np.empty((nlines[0],nlines[1]))
    z=np.random.uniform(0.25*fsize[2],0.75*fsize[2],(nlines[0],nlines[1]))

    #set random ordered locations for the interior points
    spacings=(fsize[0]/(nlines[0]-2),fsize[1]/(nlines[1]-2))
    for k in range(0, nlines[0]):
        for l in range(0, nlines[1]):
            x[k, l] = round(random.uniform(0, 1) * (spacings[0] - 1) + spacings[0] * (k - 1) + 1)
            y[k, l] = round(random.uniform(0, 1) * (spacings[1] - 1) + spacings[1] * (l - 1) + 1)

    #fix the edge points to the edge
    x[0, :] = 0
    x[-1, :] = fsize[1]-1
    y[:, 0] = 0
    y[:, -1] = fsize[0]-1

    field = interlin2d(x,y,z,fsize)
    """


    #number of lines in grid in x and y directions
    nsegx=x.shape[0]
    nsegy=x.shape[1]

    #lines along the grid points to be interpolated, x and y directions
    #0 indicates own axis, 1 is height (z axis)
    intlinesx=np.empty((2,nsegy,fsize[0]))
    intlinesy=np.empty((2,nsegx,fsize[1]))

    #account for the first and last points being fixed to the edges
    intlinesx[0,0,:]=0
    intlinesx[0,-1,:]=fsize[1]-1
    intlinesy[0,0,:]=0
    intlinesy[0,-1,:]=fsize[0]-1

    #temp fields for interpolation in x and y directions
    tempx=np.empty((fsize[0],fsize[1]))
    tempy=np.empty((fsize[0],fsize[1]))

    #interpolate grid lines in the x direction
    for k in range(nsegy):
        interp = interp1d(x[:,k], y[:,k], kind='linear', copy=False, fill_value='extrapolate')
        intlinesx[0,k,:] = np.round(interp(range(fsize[0])))
        interp = interp1d(x[:, k], z[:, k], kind='linear', copy=False, fill_value='extrapolate')
        intlinesx[1, k, :] = interp(range(fsize[0]))
    intlinesx[0,:,:].sort(0)

    # interpolate grid lines in the y direction
    for k in range(nsegx):
        interp = interp1d(y[k, :], x[k, :], kind='linear', copy=False, fill_value='extrapolate')
        intlinesy[0, k, :] = np.round(interp(range(fsize[1])))
        interp = interp1d(y[k, :], z[k, :], kind='linear', copy=False, fill_value='extrapolate')
        intlinesy[1, k, :] = interp(range(fsize[1]))
    intlinesy[0,:,:].sort(0)

    #interpolate plane in x direction
    for k in range(fsize[1]):
        tempx[k, :] = np.interp(range(fsize[1]),intlinesx[0,:,k], intlinesx[1,:,k])

    #interpolate plane in y direction
    for k in range(fsize[1]):
        tempy[:, k] = np.interp(range(fsize[0]), intlinesy[0, :, k], intlinesy[1, :, k])

    return (tempx+tempy)/2



# define the forward model
def IrrF(wl,shift,stretch,p0,p1,p2,p3,wl0,refwl,refdata):

    #wl: original wavelengths
    #refdata: data to be fitted at wl (w2_distorted)
    w2_distorted = wl+shift+stretch*(wl-wl0)
    scale_p = np.array([p3,p2,p1,p0])
    s2_fit = np.interp(w2_distorted,refwl,refdata)*np.polyval(scale_p,w2_distorted-wl0)
    return s2_fit

def RadF(wl,shift,stretch,p0,p1,p2,p3,wl0,refwl,refdata,RingCoef,Ring):

    #wl: original wavelengths
    #refdata: data to be fitted at wl (w2_distorted)
    w2_distorted = wl+shift+stretch*(wl-wl0)
    scale_p = np.array([p3,p2,p1,p0])
    s2_fit = np.interp(w2_distorted,refwl,refdata)*np.polyval(scale_p,w2_distorted-wl0)*(1+RingCoef*np.interp(w2_distorted,refwl,Ring))
    return s2_fit

def CalRadPix(lamda3,lamdar3,S3,Irr3,lamda4,lamdar4,S4,Irr4,Rad3,Rad4,Ring3,Ring4,refwlrt,lamda0s,ws3,wg3,ws4,wg4,minwl,sepwl,maxwl,**kwargs):

    dlamda=0.2

    if 'dlamda' in kwargs:
        dlamda=kwargs['dlamda']

    FullCal=True

    if 'FullCal' in kwargs:
        FullCal=kwargs['FullCal']  #if Full Cal == True we have two extra coefficient ([5] [6]) corresponding to the shift and stretch


    CaliDone=False

    if 'CaliDone' in kwargs:
        CaliDone=kwargs['CaliDone']

    if CaliDone:
        CaliData=kwargs['CaliData']

    if 'retcount' in kwargs:
        retcount=kwargs['retcount']

    newinit=False
    if 'inits' in kwargs:
        newinit=True
        intpars=kwargs['inits']

    method='Scipy'
    if 'method' in kwargs:
        method=kwargs['method']


    #irradiance wavelengths after calibration
    wln3 = lamda3 + ws3 + wg3 * (lamda3 - lamda0s[0])
    wln4 = lamda4 + ws4 + wg4 * (lamda4 - lamda0s[1])

    #print(wg3,wg4)

    if CaliDone==False:
        windowinds = ((wln4 >= (sepwl - dlamda)) & (wln4 <= (maxwl + dlamda)) & (np.isnan(Rad4) == False)).nonzero()
        if (np.count_nonzero(~np.isnan(Rad4[windowinds])) < 0.9 * len(Rad4[windowinds])):
            return np.arange(7) - 1
        # print(lamda4[windowinds], Rad4[windowinds], lamdar4, S4, lamda0s[1], Ring4)
        # startonce = time.process_time()
        if newinit:
            RadCal4 = CalRads(wln4[windowinds], Rad4[windowinds], lamdar4, S4, lamda0s[1], Ring4, FullCal=FullCal,
                              inits=intpars[1::2], method=method)
        else:
            RadCal4 = CalRads(wln4[windowinds], Rad4[windowinds], lamdar4, S4, lamda0s[1], Ring4, FullCal=FullCal,
                              method=method)
        # print('Radiance Calibration:', time.process_time() - startonce)

        # print(RadCal3,RadCal4)

        if FullCal:
            wld4 = np.array(wln4 + RadCal4[5] + RadCal4[6] * (wln4 - lamda0s[1]))  # + RadCal3[4]
        else:
            wld4 = np.array(wln4 + RadCal4[5])

        sr2rad4 = S4 * (
                RadCal4[0] + RadCal4[1] * (lamdar4 - lamda0s[1]) + RadCal4[2] * ((lamdar4 - lamda0s[1]) ** 2) +
                RadCal4[3] * ((lamdar4 - lamda0s[1]) ** 3)) * (1 + RadCal4[4] * Ring4)
        sr2rad4itp = interp1d(lamdar4, sr2rad4, fill_value="extrapolate")
        cor4 = pearsonr(sr2rad4itp(wld4[windowinds]), Rad4[windowinds])[0]

        if FullCal:
            radws4 = RadCal4[5]
            radwg4 = RadCal4[6]
        else:
            radws4 = RadCal4[5]
            radwg4 = 0.

        # wld4 = np.array(wln4 + RadCal4.params['shift'].value ) #+ RadCal4[5] * (lamda4 - lamda0s[1])
        # cor4=pearsonr(RadCal4.best_fit, Rad4[windowinds])[0]

        if (cor4 < 0.975) & (FullCal == False):
            if newinit:
                RadCal4 = CalRads(wln4[windowinds], Rad4[windowinds], lamdar4, S4, lamda0s[1], Ring4, FullCal=True,
                                  inits=intpars[1::2], method=method)
            else:
                RadCal4 = CalRads(wln4[windowinds], Rad4[windowinds], lamdar4, S4, lamda0s[1], Ring4, FullCal=True,
                                  method=method)
            wld4 = np.array(wln4 + RadCal4[5] + RadCal4[6] * (wln4 - lamda0s[1]))  # + RadCal3[4]

            sr2rad4 = S4 * (
                    RadCal4[0] + RadCal4[1] * (lamdar4 - lamda0s[1]) + RadCal4[2] * ((lamdar4 - lamda0s[1]) ** 2) +
                    RadCal4[3] * ((lamdar4 - lamda0s[1]) ** 3)) * (1 + RadCal4[4] * Ring4)
            sr2rad4itp = interp1d(lamdar4, sr2rad4, fill_value="extrapolate")
            cor4 = pearsonr(sr2rad4itp(wld4[windowinds]), Rad4[windowinds])[0]
            radws4 = RadCal4[5]
            radwg4 = RadCal4[6]

        # ax.plot(lamdar4, S4, color='red', linewidth=0.3)
        # zax.plot(wld4[windowinds], Rad4[windowinds], color='green', linewidth=0.3)
        # zax.plot(lamdar4, sr2rad4, color='orange', linewidth=0.3,linestyle='--')
        # zax.scatter(sr2rad4itp(wld4[windowinds]), Rad4[windowinds], s=0.3, color='blue')
        # plt.text(0.3, 0.8, '{:10.3f}'.format(cor4), transform=zax.transAxes, color='blue')
        # if FullCal == True:
        #     outpng = 'CaliFull.'+'{:10.0f}'.format(retcount).strip()+'.png'
        # else:
        #     outpng = 'Cali.'+'{:10.0f}'.format(retcount).strip()+'.png'
        # plt.savefig('/Users/chili/RadCal/' + outpng, dpi=600)
        # plt.close()
        # print(RadCal4)
        #
        # exit()

        if (cor4 < 0.975):
            return np.arange(7) - 1

        radrc4 = RadCal4[4]

        windowinds = ((wln3 >= (minwl - dlamda)) & (wln3 <= (sepwl + dlamda)) & (np.isnan(Rad3) == False)).nonzero()

        if (np.count_nonzero(~np.isnan(Rad3[windowinds])) < 0.9 * len(Rad3[windowinds])):

            return np.arange(7) - 1

        if newinit:

            RadCal3 = CalRads(wln3[windowinds], Rad3[windowinds], lamdar3, S3, lamda0s[0], Ring3, FullCal=FullCal,
                              inits=intpars[0::2], method=method, fixRing=radrc4)
        else:

            RadCal3 = CalRads(wln3[windowinds], Rad3[windowinds], lamdar3, S3, lamda0s[0], Ring3, FullCal=FullCal,
                              method=method, fixRing=radrc4)

        # wld3 = np.array(wln3 + RadCal3.params['shift'].value )  #+ RadCal3[5] * (lamda3 - lamda0s[0])
        # cor3 = pearsonr(RadCal3.best_fit, Rad3[windowinds])[0]

        if FullCal:
            wld3 = np.array(wln3 + RadCal3[5] + RadCal3[6] * (wln3 - lamda0s[0]))  # + RadCal3[4]
        else:
            wld3 = np.array(wln3 + RadCal3[5])  # + RadCal3[4]
        sr2rad3 = S3 * (RadCal3[0] + RadCal3[1] * (lamdar3 - lamda0s[0]) + RadCal3[2] * ((lamdar3 - lamda0s[0]) ** 2) +
                        RadCal3[3] * ((lamdar3 - lamda0s[0]) ** 3)) * (1 + RadCal3[4] * Ring3)
        sr2rad3itp = interp1d(lamdar3, sr2rad3, fill_value="extrapolate")

        cor3 = pearsonr(sr2rad3itp(wld3[windowinds]), Rad3[windowinds])[0]



        if FullCal:
            radws3 = RadCal3[5]
            radwg3 = RadCal3[6]
        else:
            radws3 = RadCal3[5]
            radwg3 = 0.

        if (cor3 < 0.975) & (FullCal == False):
            if newinit:
                RadCal3 = CalRads(wln3[windowinds], Rad3[windowinds], lamdar3, S3, lamda0s[0], Ring3, FullCal=True,
                                  inits=intpars[0::2], method=method, fixRing=radrc4)
            else:
                RadCal3 = CalRads(wln3[windowinds], Rad3[windowinds], lamdar3, S3, lamda0s[0], Ring3, FullCal=True,
                                  method=method, fixRing=radrc4)

            # wld3 = np.array(wln3 + RadCal3.params['shift'].value )  #+ RadCal3[5] * (lamda3 - lamda0s[0])
            # cor3 = pearsonr(RadCal3.best_fit, Rad3[windowinds])[0]

            wld3 = np.array(wln3 + RadCal3[5] + RadCal3[6] * (wln3 - lamda0s[0]))  # + RadCal3[4]
            sr2rad3 = S3 * (
                    RadCal3[0] + RadCal3[1] * (lamdar3 - lamda0s[0]) + RadCal3[2] * ((lamdar3 - lamda0s[0]) ** 2) +
                    RadCal3[3] * ((lamdar3 - lamda0s[0]) ** 3)) * (1 + RadCal3[4] * Ring3)
            sr2rad3itp = interp1d(lamdar3, sr2rad3, fill_value="extrapolate")

            cor3 = pearsonr(sr2rad3itp(wld3[windowinds]), Rad3[windowinds])[0]

            radws3 = RadCal3[5]
            radwg3 = RadCal3[6]

        if cor3 < 0.975:
            return np.arange(7) - 1

        radrc3 = RadCal3[4]
        # try:
        #
        #
        #
        # except:
        #     return np.arange(7) - 1
    else:
        radws3=CaliData[0]
        radws4=CaliData[1]
        radwg3 = CaliData[2]
        radwg4 = CaliData[3]
        radrc3 = CaliData[4]
        radrc4 = CaliData[5]
        wld4 = np.array(wln4 + radws4 + radwg4 * (wln4 - lamda0s[1]))
        wld3 = np.array(wln3 + radws3 + radwg3 * (wln3 - lamda0s[0]))
        if (np.isnan(radws3))|(radws3<-0.99):
            return np.arange(7) - 1



    # fig, zax = plt.subplots()
    #
    # # zax.set_xlim(minwl, maxwl)
    # # ax.plot(lamdar3, S3, color='red', linewidth=0.3)
    # # zax = ax.twinx()
    # # zax.plot(wld3[windowinds], Rad3[windowinds], color='green', linewidth=0.3)
    # # zax.plot(lamdar3,sr2rad3,color='orange',linewidth=0.3,linestyle='--')
    # zax.scatter(sr2rad3itp(wld3[windowinds]), Rad3[windowinds], s=0.3, color='red')
    # plt.text(0.1, 0.8, '{:10.3f}'.format(cor3), transform=zax.transAxes, color='red')





    wln = np.append(wln3, wln4)
    Rs = np.append(Irr3, Irr4)
    wld = np.append(wld3, wld4)
    Rad = np.append(Rad3, Rad4)
    Ring3interp = interp1d(lamdar3, Ring3, fill_value="extrapolate")
    Ring4interp = interp1d(lamdar4, Ring4, fill_value="extrapolate")

    #RingCor = np.append(1. / (Ring3interp(wld3) * RadCal3.params['RingCoef'].value + 1.), 1. / (Ring4interp(wld4) * RadCal4.params['RingCoef'].value + 1.))
    RingCor = np.append(1. / (Ring3interp(wld3) * radrc3 + 1.),
                        1. / (Ring4interp(wld4) * radrc4 + 1.))

    Rad = Rad.flatten() * RingCor.flatten()

    Radinterp = interp1d(wld[(np.isnan(Rad) == False)], Rad[(np.isnan(Rad) == False)], fill_value="extrapolate")
    Rsinterp = interp1d(wln[(np.isnan(Rs) == False)], Rs[(np.isnan(Rs) == False)], fill_value="extrapolate")


    stokes = (Radinterp(refwlrt) / Rsinterp(refwlrt)).flatten()

    #return [RadCal3.params['shift'].value+ws3,RadCal4.params['shift'].value+ws4,stokes]

    return [radws3,radws4,radwg3,radwg4, radrc3, radrc4, stokes]

def rebin(arr, new_shape,**kwargs):

    new_shape=np.array(new_shape).astype(int)

    DoAdd=False
    if 'Add' in kwargs:
        if kwargs['Add']==True:
            DoAdd=True

    if new_shape[0]<arr.shape[0]:
        shape = (new_shape[0], arr.shape[0] // new_shape[0],
                 new_shape[1], arr.shape[1] // new_shape[1])
        if DoAdd:
            return np.nansum(np.nansum(arr.reshape(shape), axis=-1), axis=1)
        else:
            return np.nanmean(np.nanmean(arr.reshape(shape), axis=-1), axis=1)
    else:
        shape = (arr.shape[0], new_shape[0] // arr.shape[0],
                 arr.shape[1], new_shape[1] // arr.shape[1])

        stackarr = np.stack([np.stack([arr] * shape[1], axis=1)]*shape[3],axis=3)
        return stackarr.reshape(new_shape)


def BRDF(SZA,VZA,RAA,riso,rvol,rgeo):

    hb=2.
    br=1.
    dpi=np.pi

    tanszapr = br*np.tan(dpi*SZA/180.)
    tanvzapr = br*np.tan(dpi*VZA/180.)
    if br==1.:
        szapr=dpi*SZA/180.
        vzapr=dpi*VZA/180.
    else:
        szapr=np.arctan(tanszapr)
        vzapr=np.arctan(tanvzapr)


    cosRAA=np.cos(dpi*RAA/180.)
    cossza=np.cos(dpi*SZA/180.)
    cosvza=np.cos(dpi*VZA/180.)

    cosseta=cossza*cosvza+np.sin(dpi*SZA/180.)*np.sin(dpi*VZA/180.)*cosRAA
    seta=np.arccos(cosseta)

    Dsq=tanszapr**2+tanvzapr**2-2*tanszapr*tanvzapr*cosRAA

    cosszapr=np.cos(szapr)
    cosvzapr=np.cos(vzapr)
    cosmult=cosszapr*cosvzapr
    secsum=1./cosszapr+1./cosvzapr

    cossetapr = cosmult + np.sin(szapr) * np.sin(vzapr) * cosRAA

    cost=hb*np.sqrt(Dsq+(tanszapr*tanvzapr*np.sin(dpi*RAA/180.))**2)/secsum
    cost[cost>=1]=1.
    cost[cost<=-1]=-1.

    t=np.arccos(cost)
    O=1/dpi*(t-np.sin(t)*cost)*secsum

    Kvol=((0.5*dpi-seta)*cosseta+np.sin(seta))/(cossza+cosvza)-0.25*dpi
    Kgeo=O-secsum+0.5*(1+cossetapr)/cosmult

    BRDF=riso+Kvol*rvol+Kgeo*rgeo

    BRDF[BRDF<1.e-6]=1.e-6

    return BRDF
def ReadDailyOMNO2(ClimNO2dir,year,month):

    Molecm2DU = 2.69e16

    stryear = '{:10.0f}'.format(year).strip()
    strmonth = ('{:10.0f}'.format(month + 100).strip())[1:]
    files = glob.glob(ClimNO2dir + 'OMI-Aura_L3-OMNO2d_' + stryear + 'm' + strmonth + '*.he5')

    # calculate mean total and tropospheric vertical column
    firstfile = True
    for NO2file in files:

        f = h5py.File(NO2file, 'r')
        ds = f["HDFEOS/GRIDS/ColumnAmountNO2/"]  # OMI_Total_Column_Amount_SO2/Data_Fields/
        TOTdata = ds["Data Fields/ColumnAmountNO2CloudScreened"][:]
        TROPdata = ds["Data Fields/ColumnAmountNO2TropCloudScreened"][:]

        if firstfile:
            monthTOT = np.zeros(TOTdata.shape)
            monthTROP = np.zeros(TOTdata.shape)
            monthsample = np.zeros(TOTdata.shape, dtype=int)
            CLat = -89.875+np.arange(720)*0.25
            CLon = -179.875+np.arange(1440)*0.25
            firstfile = False
        else:
            valinds = ((np.isnan(TOTdata) == False) & (np.isnan(TROPdata) == False)).nonzero()
            if np.array(valinds).size > 0:
                monthTOT[valinds] = monthTOT[valinds] + TOTdata[valinds]
                monthTROP[valinds] = monthTROP[valinds] + TROPdata[valinds]
                monthsample[valinds] = monthsample[valinds] + 1
        f.close()

    # molecs/cm2 to DU
    monthTOT = monthTOT / monthsample / Molecm2DU
    monthTROP = monthTROP / monthsample / Molecm2DU
    monthStrat = monthTOT - monthTROP


    return [monthTROP,monthStrat,CLat,CLon]

#stratospheric separation,data1\2\3 corresponding to the initial (after subtracting high sources), interpolated, and smoothed one  [stradata1,stradata2, stradata3]=
def StratIntp(Lats,Lons,totdata,CPdata,CFdata,monthTROP,monthSTRAT,CLat,CLon,**kwargs):

    outtemp=False

    if 'outfile' in kwargs:
        outtemp=True
        outfile=kwargs['outfile']

    nextra=50

    ns,nl=Lats.shape


    #Total retrieved column - climatological trop column
    tropclim=RemapRs(monthTROP, CLat, CLon, Lats, Lons)
    tropclim[tropclim<0.]=0.
    iniStrat=totdata-tropclim

    iniStrat[(tropclim>0.02)|(iniStrat<0.)|(iniStrat>0.2)]=np.nan
    iniStrat[(CPdata<200.)&(CFdata>0.9)]=totdata[(CPdata<200.)&(CFdata>0.9)]

    #Calculate extended lats and lons
    [eLats,eLons]=ExtrapoLatLon(Lats,Lons,nextra=nextra)
    eStrat=RemapRs(monthSTRAT,CLat,CLon,eLats,eLons)
    (eStrat[nextra:ns+nextra,:])[:,nextra:nl+nextra]=iniStrat  #fill the obs-clim estimates

    #interpolate the data


    if outtemp:
        dso = Dataset(outfile, mode='w', format='NETCDF4')
        dso.createDimension('x', nl+2*nextra)
        dso.createDimension('y', ns+2*nextra)
        outdata = dso.createVariable('Lat', np.float32, ('y', 'x'))
        outdata.units = 'degree'
        outdata[:] = eLats
        outdata = dso.createVariable('Lon', np.float32, ('y', 'x'))
        outdata.units = 'degree'
        outdata[:] = eLons
        outdata = dso.createVariable('StratosNO2_Ini', np.float32, ('y', 'x'))
        outdata.units = 'DU'
        outdata[:] = eStrat

        dso.close()

    return iniStrat,iniStrat,iniStrat











def ExtrapoLatLon(Lats,Lons,nextra):

    ns,nl=Lats.shape

    outns=ns+2*nextra
    outnl=nl+2*nextra

    outLats=np.zeros([outns,outnl])
    outLons=np.zeros([outns,outnl])

    (outLats[nextra:ns+nextra,:])[:,nextra:nl+nextra]=Lats
    (outLons[nextra:ns + nextra, :])[:, nextra: nl + nextra] = Lons

    #for Lats, it changes more rapidly on the first dimension, so we extrapolate the 1st dimension first

    for ix in np.arange(nl):
        outLats[0:nextra,ix+nextra]=np.flip(np.arange(nextra))/3*(outLats[nextra,ix+nextra]-outLats[nextra+3,ix+nextra])+outLats[nextra,ix+nextra]
        outLats[(ns+nextra):,ix+nextra]=np.arange(nextra)/3*(outLats[nextra+ns-1,ix+nextra]-outLats[nextra+ns-4,ix+nextra])+outLats[nextra+ns-1,ix+nextra]
    for iy in np.arange(outns):
        outLats[iy,0:nextra]=np.flip(np.arange(nextra))/3*(outLats[iy,nextra]-outLats[iy,nextra+3])+outLats[iy,nextra]
        outLats[iy,(nl + nextra):] = np.arange(nextra)/3 * (outLats[iy,nextra + nl - 4] - outLats[iy,nextra + nl - 4]) + outLats[iy,nextra + nl - 1]

    #for lons, extrapolate the 2nd dimension first
    for iy in np.arange(ns):
        outLons[iy+nextra,0:nextra]=np.flip(np.arange(nextra))/3*(outLons[iy+nextra,nextra]-outLons[iy+nextra,nextra+3])+outLons[iy+nextra,nextra]
        outLons[iy+nextra,(nl + nextra):] = np.arange(nextra)/3 * (outLons[iy+nextra,nextra + nl - 1] - outLons[iy+nextra,nextra + nl - 4]) + outLons[iy+nextra,nextra + nl - 1]
    for ix in np.arange(outnl):
        outLons[0:nextra,ix]=np.flip(np.arange(nextra))/3*(outLons[nextra,ix]-outLons[nextra+3,ix])+outLons[nextra,ix]
        outLons[(ns+nextra):,ix]=np.arange(nextra)/3*(outLons[nextra+ns-1,ix]-outLons[nextra+ns-4,ix])+outLons[nextra+ns-1,ix]

    return [outLats,outLons]





def InterpRow(infile, Vars, retrows):


    ds=Dataset(infile,'r')

    firstrow=ds['rows'][0]

    Intpdata=[]
    firstVar=True
    for Var in Vars:

        multrowdata=ds[Var][:]
        if firstVar:
            nr,ny,nx=multrowdata.shape
            firstVar==False

        Vardata=np.zeros([ny,nx])

        for ix in np.arange(nx,dtype=int):
            thisrow=ix+firstrow
            rowdist=retrows-thisrow

            if np.min(np.absolute(rowdist))==0:
                Vardata[:,ix]=multrowdata[retrows==thisrow,:,ix]
            else:

                minrow=np.argwhere(rowdist==np.max(rowdist[rowdist<0])).flatten()
                maxrow=np.argwhere(rowdist==np.min(rowdist[rowdist>0])).flatten()
                Vardata[:,ix]=(thisrow-retrows[minrow])/(retrows[maxrow]-retrows[minrow])*(multrowdata[maxrow,:,ix]-multrowdata[minrow,:,ix])+multrowdata[minrow,:,ix]


        Intpdata.append(Vardata)

    ds.close()


    return Intpdata

#read 0.05 deg regridded MCD43 parameters
def ReadMCD43(dir,year,doy):

    stryear='{:10.0f}'.format(year).strip()
    strdoy=('{:10.0f}'.format(doy+1000).strip())[1:]

    MCDfile=glob.glob(dir+stryear+'/'+strdoy+'/*.hdf')

    if len(MCDfile)<1:
        return np.zeros(5,dtype=int)

    MCDfile=MCDfile[0]

    try:
        ds = SD(MCDfile, SDC.READ)
    except:
        return np.zeros(5, dtype=int)

    # datasets_dic = ds.datasets()
    #
    # for idx, sds in enumerate(datasets_dic.keys()):
    #     print (idx, sds)

    Lat = 90.-np.arange(3600)*0.05
    Lon = -180.+np.arange(7200)*0.05


    Par1 = ds.select('BRDF_Albedo_Parameter1_Band3').get()*0.001
    Par2 = ds.select('BRDF_Albedo_Parameter2_Band3').get()*0.001
    Par3 = ds.select('BRDF_Albedo_Parameter3_Band3').get()*0.001

    QA = ds.select('BRDF_Quality').get()

    Par1[QA>200]=np.nan
    Par2[QA>200]=np.nan
    Par3[QA>200]=np.nan


    ds.end()

    return [np.flip(Lat),Lon,np.flip(Par1,axis=0),np.flip(Par2,axis=0),np.flip(Par3,axis=0)]


def RemapRs(indata,inlat,inlon,outlats,outlons):

    dlat=np.absolute(inlat[1]-inlat[0])
    dlon = np.absolute(inlon[1] - inlon[0])

    latinds=np.array(np.round((outlats-inlat[0])/dlat)).flatten().astype(int)
    loninds=np.array(np.round((outlons-inlon[0])/dlon)).flatten().astype(int)

    ny, nx = outlats.shape


    if len(indata.shape)==3:
        nwl=(indata.shape)[0]
        outdata = indata[:, latinds, loninds].reshape([nwl, ny, nx])
    else:
        outdata = indata[latinds, loninds].reshape([ny, nx])

    return outdata

def fill(data, invalid=None,**kwargs):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid')
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'.
                 data value are replaced where invalid is True
                 If None (default), use: invalid  = np.isnan(data)

    Output:
        Return a filled array.
    """
    if invalid is None: invalid = np.isnan(data)
    if 'interp' in kwargs:
        if kwargs['interp']==True:
            x, y = np.indices(data.shape)
            interp = data.copy()
            interp[np.isnan(interp)] = griddata((x[invalid==False], y[invalid==False]),data[invalid==False],\
                                                (x[invalid==True], y[invalid==True]),method='linear')  # points to interpolate
            return interp
    elif 'rbf' in kwargs:
        if kwargs['rbf']==True:
            x, y = np.indices(data.shape)
            interp = data.copy()
            xyrbf=Rbf(x[invalid==False], y[invalid==False],data[invalid==False], epsilon=2, function='gaussian')
            interp[np.isnan(interp)] = xyrbf(x[invalid==True], y[invalid==True]) # points to interpolate
            return interp
    else:
        ind = nd.distance_transform_edt(invalid,
                                        return_distances=False,
                                        return_indices=True)
        return data[tuple(ind)]

def ReadStatic(staticfile):

    ds = Dataset(staticfile, 'r')
    SR3 = ds['band_3/irradiance_flux_cf'][:].squeeze()
    Ring3 = ds['band_3/radiance_ring_flux_cf'][:].squeeze()/SR3
    lamdar3=ds['band_3/wavelength'][:].squeeze()
    SR4 = ds['band_4/irradiance_flux_cf'][:].squeeze()
    Ring4 = ds['band_4/radiance_ring_flux_cf'][:].squeeze() / SR4
    lamdar4 = ds['band_4/wavelength'][:].squeeze()
    ds.close()
    return [lamdar3,SR3,Ring3,lamdar4,SR4,Ring4]

def ReadRegAOD(AODdir,year,doy,Lats,Lons):

    ny,nx=Lats.shape

    stryear='{:10.0f}'.format(year).strip()
    strdoy = ('{:10.0f}'.format(doy + 1000).strip())[1:]

    AODfiles = glob.glob(AODdir + stryear+'/'+strdoy + '/*hdf')

    #nrecs=np.zeros([ny,nx],dtype=int)
    outAOD=np.zeros([ny,nx])

    #Lats1D=Lats.flatten()
    #Lons1D=Lons.flatten()
    startind=0
    for iy in np.arange(ny):
        for ix in np.arange(nx):
            plat=Lats[iy,ix]
            plon=Lons[iy,ix]
            [startind,pAOD]=ExtracMOD04(AODdir,doy,plat,plon,AODfiles=AODfiles,startind=startind)


            outAOD[iy,ix]=pAOD
            #nrecs[iy,ix]=nrecs[iy,ix]+1



    # for AODfile in AODfiles:
    #
    #     try:
    #         ds = SD(AODfile, SDC.READ)
    #
    #     except:
    #         continue
    #
    #
    #     Lat = ds.select('Latitude').get().flatten()
    #     Lon = ds.select('Longitude').get().flatten()
    #
    #
    #
    #     nsnap=Lat.size
    #     print(nsnap)
    #     #dist=np.zeros([nref,nsnap])
    #
    #     dist=(np.stack([Lats1D]*nsnap,axis=1)-np.stack([Lat]*nref,axis=0))**2+(np.stack([Lons1D]*nsnap,axis=1)-np.stack([Lon]*nref,axis=0))**2
    #     print(np.min(dist),np.max(dist))
    #     if np.min(dist)>8.e-4:
    #         ds.end()
    #         continue
    #
    #     mindist=np.min(dist,axis=1)
    #     mininds=np.argmin(dist,axis=1)
    #
    #
    #     AODvar = 'Image_Optical_Depth_Land_And_Ocean'
    #     AODQAvar = AODvar + '_QA_Flag'
    #
    #     AODobj = ds.select(AODvar).flatten()
    #     AOD = (AODobj.get() * AODobj.attributes()['scale_factor']).flatten()
    #     ds.end()
    #
    #     AODmin=AOD[mininds]
    #     Latc=Lat[mininds]
    #     Lonc=Lon[mininds]
    #
    #     AODmin[AODmin<=0]=0.
    #
    #     tmininds=(mindist<=8.e-4).nonzero()
    #     print(Lats1D[tmininds],Lons1D[tmininds],Latc[tmininds],Lonc[tmininds])
    #     tAOD[tmininds]=tAOD[tmininds]+AODmin[tmininds]
    #     nrecs[tmininds]=nrecs[tmininds]+1

    #outAOD=tAOD/nrecs
    #outAOD[(nrecs==0)|(tAOD<=0.)]=np.nan
    return outAOD#.reshape([ny,nx])

def ReadHRGAOD(AODdir,year,doy):

    strdoy = '{:10.0f}'.format(year).strip()+('{:10.0f}'.format(doy + 1000).strip())[1:]

    AODfile = glob.glob(AODdir + '*'+strdoy+'*nc')[0]

    try:
        ds = Dataset(AODfile, 'r')

    except:
        return np.zeros(3) - 1


    Lat=ds['Latitude'][:]
    Lon=ds['Longitude'][:]
    AOD=ds['AOD_550_GF'][:]
    AOD[AOD<0.]=np.nan
    ds.close()

    return (AOD,Lat,Lon)

def ReadMOD08(AODdir, doy):
    strdoy = ('{:10.0f}'.format(doy + 1000).strip())[1:]

    AODfile = glob.glob(AODdir + strdoy + '/*hdf')[0]

    try:
        ds = SD(AODfile, SDC.READ)

    except:
        return np.zeros(3)-1

    Lat = 90.-np.arange(180)  #mod08/Data_Fields/
    Lon = -180+np.arange(360)

    AODvar = 'Aerosol_Optical_Depth_Land_Mean'

    AODobj = ds.select(AODvar)
    AOD = (AODobj.get() * AODobj.attributes()['scale_factor'])[1,:,:]

    AODvar = 'Deep_Blue_Aerosol_Optical_Depth_550_Land_Mean'
    AODobj = ds.select(AODvar)
    DBAOD = AODobj.get() * AODobj.attributes()['scale_factor']
    ds.end()

    AOD[AOD<0.]=DBAOD[AOD<0.]
    AOD[AOD<0.]=np.nan

    return (np.flip(AOD.squeeze(),axis=0),np.flip(Lat),Lon)

def ExtracMOD04(AODdir, doy, plat, plon,**kwargs):

    strdoy=('{:10.0f}'.format(doy+1000).strip())[1:]


    if 'AODfiles' in kwargs:
        AODfiles=glob.glob(AODdir+strdoy+'/*hdf')
    else:
        AODfiles=kwargs['AODfiles']

    if 'startind' in kwargs:
        startind=kwargs['startind']
    else:
        startind=0

    nfiles=len(AODfiles)

    Foundflag=False

    for ifile in np.arange(nfiles)+startind:

        if Foundflag==True:
            ifile=ifile-1
            if ifile>=nfiles:
                ifile=ifile-nfiles
            break

        if ifile<nfiles:
            AODfile=AODfiles[ifile]
        else:
            AODfile=AODfiles[ifile-nfiles]


        try:
            ds = SD(AODfile, SDC.READ)

        except:
            continue


        Lat = ds.select('Latitude').get()
        Lon = ds.select('Longitude').get()

        dist=(Lat-plat)**2+(Lon-plon)**2

        if np.min(dist)>1e-3:
            continue
        else:
            # default AOD varaible
            AODvar = 'Image_Optical_Depth_Land_And_Ocean'
            AODQAvar = AODvar + '_QA_Flag'

            AODobj = ds.select(AODvar)
            AOD = (AODobj.get() * AODobj.attributes()['scale_factor'])[dist==np.min(dist)]

            Foundflag=True
        ds.end()


    if Foundflag==False:
        return [startind,np.nan]

    if AOD<=0.:
        AOD=np.nan


    return [ifile,AOD]

def ReadO3(O3file):

    ds = Dataset(O3file, 'r')
    O3Col=ds['PRODUCT/ozone_total_vertical_column'][:].squeeze()
    QA=ds['PRODUCT/qa_value'][:].squeeze()
    #O3Col[QA<0.5]=np.nan
    ds.close()
    return [O3Col,QA]

def Converthgt (pres):

    p0=1013.25
    t0=15.
    Hgt=pres-pres
    Hgt=((p0/pres)**(1./5.257)-1.)*(t0+273.15)/0.0065
    #pres=p0/((Hgt*0.0065/(t0+273.15)+1.)**5.257)

    return Hgt


def ReadSO2(SO2file):

    ds = Dataset(SO2file, 'r')
    SO2Col=ds['PRODUCT/sulfurdioxide_total_vertical_column'][:].squeeze()
    QA=ds['PRODUCT/qa_value'][:].squeeze()
    SO2Col[QA<0.5]=np.nan
    ds.close()
    return SO2Col

def ReadNO2(NO2file):

    #,SNO2QA,CldFr,CldAbd,Cldpr

    ds = Dataset(NO2file, 'r')
    NO2Col = ds['PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/nitrogendioxide_summed_total_column'][:].squeeze()
    NO2gCol = ds['PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/nitrogendioxide_ghost_column'][:].squeeze()
    NO2TCol = ds['PRODUCT/nitrogendioxide_tropospheric_column'][:].squeeze()
    #LER = ds ['PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_albedo_nitrogendioxide_window'][:].squeeze()
    #SurfHs = ds['PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_altitude'][:].squeeze()
    QA = ds['PRODUCT/qa_value'][:].squeeze()
    CF = ds['PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/cloud_radiance_fraction_nitrogendioxide_window'][:].squeeze()
    Cabd = ds['PRODUCT/SUPPORT_DATA/INPUT_DATA/cloud_albedo_crb'][:].squeeze()
    Cpr = ds['PRODUCT/SUPPORT_DATA/INPUT_DATA/cloud_pressure_crb'][:].squeeze()/100.  #hpa
    # NO2TCol[QA < 0.5] = np.nan
    # NO2Col[QA < 0.5] = np.nan
    # LER[QA < 0.5] = np.nan
    #SurfHs[QA < 0.5] = np.nan
    ds.close()

    return [NO2Col,NO2gCol,NO2TCol,QA,CF,Cabd,Cpr]



def ReadSAO(SAOsolarfile):
    NAv=6.02214e23
    lamdar=[]
    datar=[]
    scaling=1.e4/NAv #from photons cm-2 s-1 nm-1 to mols m-2 s-1 nm-1

    fid = open(SAOsolarfile, 'r')
    lines=fid.readlines()
    for line in lines[5:]:
        strline = line.split()
        lamdar.append(np.float(strline[0])) #nm
        datar.append(np.float(strline[1]) * scaling) #mols m-2 s-1 nm-1
    fid.close()
    return [np.array(lamdar),np.array(datar)]

def ReadNO2Xsec(NO2Xsecfile):

    lamdar = []
    datar = []


    fid = open(NO2Xsecfile, 'r')
    lines = fid.readlines()
    for line in lines[1:]:
        strline = line.split()
        lamdar.append(np.float(strline[0]))  # nm
        datar.append(np.float(strline[2]))  # mols m-2 s-1 nm-1
    fid.close()
    return [np.array(lamdar), np.array(datar)]


def ConvolSRF(lamda,data,srflamda,srfdlamda,srf):

    #lamda: wavelength of unconvoluted data
    #data: unconvoluted data
    #srflamda: central wavelength
    #srfdlamda: deltalamda for each srf grid
    #srf: srf functions as in the shape of [ndlamda,nlamda]
    #1. Calculate the wavelength surface:

    nlamda,ndlamda=srf.shape
    lamda2d=np.stack([srflamda]*ndlamda,axis=1)+np.stack([srfdlamda]*nlamda,axis=0)

    #2. Interpolate data from lamda to lamda2d
    interpdata=np.zeros([nlamda,ndlamda])
    interpdata=np.interp(lamda2d,lamda,data,left=np.nan,right=np.nan)


    Convdata=np.sum(interpdata*srf,axis=1)/np.sum(srf,axis=1)



    return [srflamda,Convdata]

#residual of irradiance
def IrrModel(reflamda,refdata,lamda,lamda0,shift,stretch,p0,p1,p2,p3):

    # Irr(lamdacor) = Irrref(lamdacor)*(p0+p1*lamdacor+p1*lamdacor^2)

    datainterp = interp1d(reflamda,refdata, fill_value="extrapolate")
    interpwls=np.array(lamda+shift+stretch*(lamda-lamda0))
    calidata = datainterp(interpwls)*(p1*(interpwls-lamda0)+p2*((interpwls-lamda0)**2)+p3*((interpwls-lamda0)**3)+p0)#+x[5]*((interpwls-lamda0)**3)+x[0])



    #print(np.sum(((calidata-data))**2))
    return calidata

#residual of irradiance
def IrrRes(x,*args):

    # Irr(lamdacor) = Irrref(lamdacor)*(p0+p1*lamdacor+p1*lamdacor^2)
    reflamda=args[0]
    refdata=args[1]
    lamda=args[2]
    data=args[3]
    lamda0=args[4]
    datainterp = interp1d(reflamda,refdata, fill_value="extrapolate")
    interpwls=np.array(lamda+x[3]+x[4]*(lamda-lamda0))
    calidata = datainterp(interpwls)*(x[1]*(interpwls-lamda0)+x[2]*((interpwls-lamda0)**2)+x[5]*((interpwls-lamda0)**3)+x[0])#+x[5]*((interpwls-lamda0)**3)+x[0])

    #print(np.sum(((calidata-data))**2))
    return np.sum(((calidata-data))**2)/np.sum(data**2)

#residual of radiance
def RadRes(x,*args):

    # Rad(lamdacor) = Irrref(lamdacor)*(p0+p1*lamdacor+p1*lamdacor^2)*(1+Cring*Iring)

    reflamda=args[0]
    refdata=args[1]
    lamda=args[2]
    data=args[3]
    lamda0=args[4]
    refRing=args[5]
    FullCal = args[6]

    datainterp = interp1d(reflamda,refdata, fill_value="extrapolate")
    Ringinterp = interp1d(reflamda, refRing, fill_value="extrapolate")
      #+x[5]*(lamda-lamda0)  #+x[4]
    if FullCal==True:
        interpwls = np.array(lamda+x[5]+x[6]*(lamda-lamda0))
    else:
        interpwls = np.array(lamda+x[5])
    calidata = datainterp(interpwls)

    poly=x[1]*(interpwls-lamda0)+x[2]*((interpwls-lamda0)**2)+x[3]*((interpwls-lamda0)**3)+x[0]
    CorRing = Ringinterp(interpwls)*x[4]+1  #x[5]

    return np.sum(((poly*CorRing*calidata - data)) ** 2)/np.sum(data**2)

def RadModel(reflamda,refdata,lamda,lamda0,refRing,p0,p1,p2,p3,RingCoef,shift,stretch):

    # Rad(lamdacor) = Irrref(lamdacor)*(p0+p1*lamdacor+p1*lamdacor^2)*(1+Cring*Iring)

    datainterp = interp1d(reflamda,refdata, fill_value="extrapolate")
    Ringinterp = interp1d(reflamda, refRing, fill_value="extrapolate")
      #+x[5]*(lamda-lamda0)  #+x[4]

    interpwls = np.array(lamda+shift+stretch*(lamda-lamda0))
    calidata = datainterp(interpwls)

    poly=p1*(interpwls-lamda0)+p2*((interpwls-lamda0)**2)+p3*((interpwls-lamda0)**3)+p0
    CorRing = Ringinterp(interpwls)*RingCoef+1  #x[5]

    return poly*CorRing*calidata


#irradiance calibration
def CaliIrr(lamda,data,reflamda,refdata,lamda0,**kwargs):

    #P0 = 1.
    #ws0 = 0.
    #wg0 = 0.
    #Irr(lamdacor) = Irrref(lamdacor)*(p0+p1*lamdacor+p1*lamdacor^2)

    dlamda = 0.2

    if 'dlamda' in kwargs:
        dlamda = kwargs['dlamda']

    Optmethod = 'Scipy'

    if 'method' in kwargs:
        Optmethod = kwargs['method']

    x0 = np.zeros(6)  #3 polynomial and 2 shift/squeeze parameters
    x0[0] = np.mean(data)/np.mean(refdata)
    x0[1] = 0.0
    x0[2] = 0.0
    x0[3] = 0.0
    x0[4] = 0.0  # , min = x[0], max = x[-1])
    x0[5] = 0.0



    lbds = np.array([-1*np.inf,-1*np.inf,-1*np.inf, -1*dlamda, -1*np.inf, -1*np.inf])
    hbds = np.array([np.inf, np.inf,np.inf,dlamda, np.inf,np.inf])  # np.max([maxxW - np.min(x), np.max(x) - maxxW])
    bounds = Bounds(lbds, hbds)

    if Optmethod=='Scipy':
        res = scipyminimize(IrrRes, x0, args=(reflamda,refdata,lamda,data,lamda0),bounds=bounds,method='Powell') #\ # method='Powell', \
        return res.x

    else:
        parnames = ['p0', 'p1', 'p2', 'shift', 'stretch','p3']
        instrument_model = Model(IrrModel, \
                                 independent_vars=['reflamda', 'refdata', 'lamda', 'lamda0'], \
                                 param_names=parnames)

        instrument_model.set_param_hint('stretch', value=x0[4], min=lbds[4], max=hbds[4])
        instrument_model.set_param_hint('shift', value=x0[3], min=lbds[3], max=hbds[3])

        instrument_model.set_param_hint('p3', value=x0[5], min=lbds[5], max=hbds[5])
        instrument_model.set_param_hint('p2', value=x0[2], min=lbds[2], max=hbds[2])
        instrument_model.set_param_hint('p1', value=x0[1], min=lbds[1], max=hbds[1])
        instrument_model.set_param_hint('p0', value=x0[0], min=lbds[0], max=hbds[0])

        params = instrument_model.make_params(verbose=False)
        result = instrument_model.fit(data, params=params, reflamda=reflamda, refdata=refdata, lamda=lamda, lamda0=lamda0,method='powell')  #,method='trust_constr'

        outpars = []
        for par in parnames:
            outpars.append(result.params[par].value)


        return np.array(outpars)


    #wl, shift, stretch, p0, p1, p2, p3, wl0, refwl, refdata
    # instrument_model = Model(IrrF,independent_vars=['wl', 'wl0', 'refwl', 'refdata'],param_names = ['shift', 'stretch', 'p0', 'p1', 'p2','p3'])
    # # give initial values to parameters to be fitted. they can also be fixed
    # instrument_model.set_param_hint('shift', value = 0.,min=-1*dlamda,max=dlamda)
    # instrument_model.set_param_hint('stretch', value = 0.)
    # instrument_model.set_param_hint('p3', value = 0.)
    # instrument_model.set_param_hint('p2', value = 0.)
    # instrument_model.set_param_hint('p1', value = 0.)
    # instrument_model.set_param_hint('p0', value = np.mean(data)/np.mean(refdata))
    # params = instrument_model.make_params(verbose=False)
    # res = instrument_model.fit(data, params=params, wl=lamda, wl0=lamda0, refwl=reflamda, refdata=refdata,method='powell')
    # return res
    # #return res.params
    # #exit()               #options={'verbose': 0}, \
    #                 #bounds=bounds)  # constraints=[lconstr, nlconstr],
    # # datainterp = interp1d(reflamda, refdata, fill_value="extrapolate")
    # # interpwls = np.array(lamda + res.x[1] + res.x[2] * (lamda - lamda0))
    #
    # #print(res.x[3:5])
    #
    #
    # # if res.x[1]>0.03:
    # #
    # return [res.params['p0'].value,res.params['p1'].value,res.params['p2'].value,res.params['shift'].value,res.params['stretch'].value,res.params['p3'].value]
    #return res.x

def CalRads(lamda,rad,lamdar,sr,lamda0,radRing,**kwargs):  #,NO2Cross,Ring
    # rad: radiance
    # lamdar: wavelength in ISRF
    # sr: solar reference irradiance
    # minwl,maxwl,lamda0, spectral window and center
    # NO2Cross: (Convoluted) cross section of NO2 for fitting
    # Ring: (Convoluted) Ring spectrum

    dlamda=0.1

    if 'dlamda' in kwargs:
        dlamda=kwargs['dlamda']

    FullCal = True

    if 'inits' in kwargs:
        initpars=kwargs['inits']

    if 'FullCal' in kwargs:
        FullCal = kwargs[
            'FullCal']  # if Full Cal == True we have two extra coefficient ([5] [6]) corresponding to the shift and stretch
    #
    if FullCal:
        x0 = np.zeros(7)  #4 polynomial coefficients
        method='Powell'
    else:
        x0 = np.zeros(6)
        method='Nelder-Mead'

    Optmethod='Scipy'

    if 'method' in kwargs:
        Optmethod = kwargs['method']

    fixRing=False
    if 'fixRing' in kwargs:
        fixRing=True
        fixrc=kwargs['fixRing']
    #method = 'Powell'

    x0[0] = np.mean(rad) / np.mean(sr)
    x0[1] = 0.0
    x0[2] = 0.0  # , min = x[0], max = x[-1])
    x0[3] = 0.0  #
    #x0[4] = 0.001   #shift
    if fixRing:
        x0[4]=fixrc
    else:
        if 'inits' in kwargs:
            x0[4]=initpars[2]
        else:
            x0[4] = 0.0  # Ring effect  #we turn off the stretch parameter in radiance (adopt the stretch from irradiance calibration)
    
    if 'inits' in kwargs:
        x0[5]=initpars[0]
    else:
        x0[5] = 0.0
        
    if FullCal:
        if 'inits' in kwargs:
            x0[6] = initpars[1]
        else:
            x0[6] = 0.0

    lbds = np.array([-1 * np.inf, -1 * np.inf, -1 * np.inf,-1*np.inf, -0.1, -1 * dlamda]) #, -1 * dlamda
    hbds = np.array([np.inf, np.inf, np.inf,np.inf, 0.1, dlamda])  # np.max([maxxW - np.min(x), np.max(x) - maxxW])

    if FullCal:
        lbds=np.append(lbds,np.array([-np.inf]))
        hbds=np.append(hbds,np.array([np.inf]))

    bounds = Bounds(lbds, hbds) #

    if Optmethod=='Scipy':
        res = scipyminimize(RadRes, x0, args=(lamdar, sr, lamda, rad, lamda0, radRing,FullCal), bounds=bounds,method=method)#,options={'xtol':1.e-20,'maxiter': 10000 })  #,options={'fatol': 1.e-6,'maxiter': 5000 } ,options={'maxiter': 10000})#, method='Powell', \

        return res.x

    else:
        # build lmfit model object and claim what are permanently fixed variables
        # and what can be optimized
        if FullCal:
            parnames=['p0', 'p1', 'p2', 'p3', 'RingCoef', 'shift','stretch']
            instrument_model = Model(RadModel, \
                                     independent_vars=['reflamda', 'refdata', 'lamda', 'lamda0', 'refRing'], \
                                     param_names=parnames)
        else:
            parnames=['p0', 'p1', 'p2', 'p3', 'RingCoef', 'shift']
            instrument_model = Model(RadModel, \
                                     independent_vars=['reflamda', 'refdata', 'lamda', 'lamda0', 'refRing','stretch'], \
                                     param_names=parnames)

        instrument_model.set_param_hint('shift', value=x0[5],min=lbds[5],max=hbds[5])
        if fixRing:
            instrument_model.set_param_hint('RingCoef', value=x0[4], min=lbds[4], max=hbds[4],vary=False)
        else:
            instrument_model.set_param_hint('RingCoef', value=x0[4],min=lbds[4],max=hbds[4])
        instrument_model.set_param_hint('p3', value=x0[3],min=lbds[3],max=hbds[3])
        instrument_model.set_param_hint('p2', value=x0[2],min=lbds[2],max=hbds[2])
        instrument_model.set_param_hint('p1', value=x0[1],min=lbds[1],max=hbds[1])
        instrument_model.set_param_hint('p0', value=x0[0],min=lbds[0],max=hbds[0])
        if FullCal:
            instrument_model.set_param_hint('stretch', value=x0[6],min=lbds[6],max=hbds[6],vary=False)

        params = instrument_model.make_params(verbose=False)
        if FullCal:
            result = instrument_model.fit(rad, params=params, reflamda=lamdar,refdata=sr,lamda=lamda,lamda0=lamda0,refRing=radRing,method='powell')
        else:
            result = instrument_model.fit(rad, params=params, reflamda=lamdar, refdata=sr, lamda=lamda, lamda0=lamda0,
                                          refRing=radRing,stretch=0.)
        outpars=[]
        for par in parnames:
            outpars.append(result.params[par].value)

        return np.array(outpars)
    #res = scipyminimize(RadRes, x0, args=(lamdar, sr, lamda, rad, lamda0,radRing),bounds=bounds, method='Nelder-Mead')#,options={'maxiter': 10000})#, method='Powell', \



    #                     #options={'verbose': 0}, \
    #                     #)  # constraints=[lconstr, nlconstr],
    #
    # print(res.x[4:6])
    # datainterp = interp1d(lamdar, sr, fill_value="extrapolate")
    # Ringinterp = interp1d(lamdar, radRing, fill_value="extrapolate")
    # interpwls = np.array(lamda + res.x[4] + res.x[5] * (lamda - lamda0))
    # calidata = datainterp(interpwls)
    #
    # poly = res.x[1] * (interpwls - lamda0) + res.x[2] * ((interpwls - lamda0) ** 2) + res.x[0]
    # CorRing = Ringinterp(interpwls) * res.x[3] + 1
    # print(rad,poly*CorRing*calidata)

    # if res.x[1] > 0.03:
    #     datainterp = interp1d(reflamda, refdata, fill_value="extrapolate")
    #     interpwls = np.array(lamda + res.x[1] + res.x[2] * (lamda - lamda0))
    #     print(pearsonr(datainterp(np.array(lamda)), data)[0] ** 2, pearsonr(datainterp(interpwls), data)[0] ** 2)
    # instrument_model = Model(RadF, independent_vars=['wl', 'wl0', 'refwl', 'refdata','Ring','stretch'],
    #                          param_names=['shift', 'p0', 'p1', 'p2', 'p3','RingCoef'])
    # # give initial values to parameters to be fitted. they can also be fixed
    # instrument_model.set_param_hint('shift', value=0., min=-1 * dlamda, max=dlamda)
    # instrument_model.set_param_hint('p3', value=0.)
    # instrument_model.set_param_hint('p2', value=0.)
    # instrument_model.set_param_hint('p1', value=0.)
    # instrument_model.set_param_hint('p0', value=np.mean(rad) / np.mean(sr))
    # instrument_model.set_param_hint('RingCoef', value=0.)
    # params = instrument_model.make_params(verbose=False)
    # res = instrument_model.fit(rad, params=params, wl=lamda, wl0=lamda0, refwl=lamdar, refdata=sr,Ring=radRing,stretch=0.,method='powell',verbose=False)
    # # return res.params
    # # exit()               #options={'verbose': 0}, \
    # # bounds=bounds)  # constraints=[lconstr, nlconstr],
    # # datainterp = interp1d(reflamda, refdata, fill_value="extrapolate")
    # # interpwls = np.array(lamda + res.x[1] + res.x[2] * (lamda - lamda0))
    #
    # # print(res.x[3:5])
    #
    # # if res.x[1]>0.03:
    # #
    #
    # return res



def ReadRads(file,**kwargs):


    ReadGeo=True

    if 'DataOnly' in kwargs:
        if kwargs['DataOnly']==True:
            ReadGeo=False

    OneSpec=False  #if only experiment on one pixel
    if 'OneSpec' in kwargs:
        OneSpec=kwargs['OneSpec']
    if 'iscan' in kwargs:
        irow=kwargs['irow']
        iscan=kwargs['iscan']
        minscan=np.min(iscan)
        maxscan=np.max(iscan)
        minrow = np.min(irow)
        maxrow = np.max(irow)
        nscan=len(iscan)
        nrow=len(irow)





    strband=(file.split('L1B_RA_BD')[-1])[0]
    ds = Dataset(file, 'r')

    if 'Range' in kwargs:
        Range = kwargs['Range']
        latmin=Range[0]
        lonmin=Range[1]
        latmax=Range[2]
        lonmax=Range[3]
        lat = ds['BAND' + strband + '_RADIANCE/STANDARD_MODE/GEODATA/latitude'][:].squeeze()
        lon = ds['BAND' + strband + '_RADIANCE/STANDARD_MODE/GEODATA/longitude'][:].squeeze()
        minmax=np.array(((lat>=latmin)&(lat<=latmax)&(lon>=lonmin)&(lon<=lonmax)).nonzero())
        if (minmax.size<2):
            ds.close()
            if ReadGeo:
                return np.arange(10)-1
            else:
                return np.arange(5)-1
        minscan=np.min(minmax[0,:])
        maxscan=np.max(minmax[0,:])
        minrow=np.min(minmax[1,:])
        maxrow=np.max(minmax[1,:])
        nrow=maxrow-minrow+1
        nscan=maxscan-minscan+1
        iscan=np.arange(nscan)+minscan
        irow=np.arange(nrow)+minrow

    if (('iscan' in kwargs) | ('Range' in kwargs)):
        dradiance = ds['BAND' + strband + '_RADIANCE/STANDARD_MODE/OBSERVATIONS/radiance']
        tn,scann,rown,nspec=dradiance.shape
        radiance=dradiance[0,minscan:maxscan+1,minrow:maxrow+1,:]
        QA=ds['BAND'+strband+'_RADIANCE/STANDARD_MODE/OBSERVATIONS/quality_level'][0,minscan:maxscan+1,minrow:maxrow+1,:]
        radiance[QA<99.99]=np.nan
        radiance=radiance.reshape([nscan, nrow,nspec])
    else:
        radiance = ds['BAND' + strband + '_RADIANCE/STANDARD_MODE/OBSERVATIONS/radiance'][:]
        QA = ds['BAND' + strband + '_RADIANCE/STANDARD_MODE/OBSERVATIONS/quality_level'][:]
        radiance[QA < 99.99] = np.nan
    if ReadGeo:
        if ~('Range' in kwargs):
            lat = ds['BAND' + strband + '_RADIANCE/STANDARD_MODE/GEODATA/latitude']
            lon = ds['BAND' + strband + '_RADIANCE/STANDARD_MODE/GEODATA/longitude']


        SZA=ds['BAND'+strband+'_RADIANCE/STANDARD_MODE/GEODATA/solar_zenith_angle']
        VZA=ds['BAND'+strband+'_RADIANCE/STANDARD_MODE/GEODATA/viewing_zenith_angle']
        RAA=ds['BAND'+strband+'_RADIANCE/STANDARD_MODE/GEODATA/solar_azimuth_angle'][:]-ds['BAND'+strband+'_RADIANCE/STANDARD_MODE/GEODATA/viewing_azimuth_angle'][:]
        RAA[RAA>=360.]=RAA[RAA>=360.]-360.
        RAA[RAA < 0.] = RAA[RAA < 0.] + 360.
        RAA=180.-np.absolute(RAA-180.)  #to be consistent with the "RAA" in the LUT, namely the absolute difference without considering the "opposite" of solar beam

    if ReadGeo:
        if (('iscan' in kwargs) | ('Range' in kwargs)):
            #print(SZA[:].squeeze().shape,SZA[:].squeeze()[minscan:maxscan+1,minrow:maxrow+1].shape,(SZA[:].squeeze())[minscan:maxscan+1,minrow:maxrow+1].shape)
            if'Range' in kwargs:
                outdata = [radiance, SZA[:].squeeze()[minscan:maxscan + 1, minrow:maxrow + 1].reshape([nscan, nrow]), \
                           VZA[:].squeeze()[minscan:maxscan + 1, minrow:maxrow + 1].reshape([nscan, nrow]), \
                           RAA[:].squeeze()[minscan:maxscan + 1, minrow:maxrow + 1].reshape([nscan, nrow]), \
                           lat[:].squeeze()[minscan:maxscan + 1, minrow:maxrow + 1].reshape([nscan, nrow]), \
                           lon[:].squeeze()[minscan:maxscan + 1, minrow:maxrow + 1].reshape([nscan, nrow]),minscan,nscan,minrow,nrow]

            else:
                outdata = [radiance, SZA[:].squeeze()[minscan:maxscan+1,minrow:maxrow+1].reshape([nscan, nrow]), \
                           VZA[:].squeeze()[minscan:maxscan+1,minrow:maxrow+1].reshape([nscan, nrow]), \
                           RAA[:].squeeze()[minscan:maxscan+1,minrow:maxrow+1].reshape([nscan, nrow]), \
                           lat[:].squeeze()[minscan:maxscan+1,minrow:maxrow+1].reshape([nscan, nrow]), \
                           lon[:].squeeze()[minscan:maxscan+1,minrow:maxrow+1].reshape([nscan, nrow])]
        else:
            if 'Range' in kwargs:
                outdata = [radiance, SZA[:].squeeze(), VZA[:].squeeze(), RAA[:].squeeze(), lat[:].squeeze(),lon[:].squeeze(),minscan,nscan,minrow,nrow]
            else:
                outdata = [radiance, SZA[:].squeeze(),VZA[:].squeeze(), RAA[:].squeeze(),lat[:].squeeze(),lon[:].squeeze()]

    else:
        if 'Range' in kwargs:
            outdata = [radiance.squeeze(),minscan,nscan,minrow,nrow]
        else:
            outdata = radiance.squeeze()


    ds.close()

    return outdata

def ConvISRF(lamdar,Er,SRFfile):
    ds = Dataset(SRFfile, 'r')
    srfs4 = ds['band_4/isrf']
    srflamdas4 = ds['band_4/wavelength'][:].squeeze()
    srfdlamda4 = ds['band_4/delta_wavelength'][:]
    srfs3 = ds['band_3/isrf']
    srflamdas3 = ds['band_3/wavelength'][:].squeeze()
    srfdlamda3 = ds['band_3/delta_wavelength'][:]


    nrow3, nspec3 = srflamdas3.shape
    nrow4, nspec4 = srflamdas4.shape
    # convoluted SAO data for each row
    lamdar3 = np.zeros([nrow3, nspec3])
    datar3 = np.zeros([nrow3, nspec3])
    lamdar4 = np.zeros([nrow4, nspec4])
    datar4 = np.zeros([nrow4, nspec4])

    # These are what the irradiance should be for the TROPOMI channels
    for row in np.arange(nrow3):
        Convlamda, Convdata = ConvolSRF(lamdar, Er, srflamdas3[row, :].squeeze(), srfdlamda3,
                                        srfs3[row, :, :].squeeze())
        lamdar3[row, :] = Convlamda
        datar3[row, :] = Convdata

    for row in np.arange(nrow4):
        Convlamda, Convdata = ConvolSRF(lamdar, Er, srflamdas4[row, :].squeeze(), srfdlamda4,
                                        srfs4[row, :, :].squeeze())
        lamdar4[row, :] = Convlamda
        datar4[row, :] = Convdata

    outdata=[lamdar3,datar3,lamdar4,datar4]

    ds.close()

    return outdata

def ReadCalIrr(lamdar3,datar3,lamdar4,datar4,Irrfile,minwl,sepwl,maxwl,lamda0s,**kwargs):


    dlamda = 0.2

    if 'dlamda' in kwargs:
        dlamda = kwargs['dlamda']

    Optmethod = 'Scipy'

    if 'method' in kwargs:
        Optmethod = kwargs['method']



    DoCal=True
    if 'DoCal' in kwargs:
        DoCal=kwargs['DoCal']
    #read irradiance (photons "mol.m-2.nm-1.s-1")
    ds = Dataset(Irrfile, 'r')
    data4 = ds['BAND4_IRRADIANCE/STANDARD_MODE/OBSERVATIONS/irradiance'][:].squeeze()
    QA4 = ds['BAND4_IRRADIANCE/STANDARD_MODE/OBSERVATIONS/quality_level'][:].squeeze()
    lamdas4 = ds['BAND4_IRRADIANCE/STANDARD_MODE/INSTRUMENT/calibrated_wavelength'][:].squeeze()
    data4[QA4<0.99]=np.nan


    data3 = ds['BAND3_IRRADIANCE/STANDARD_MODE/OBSERVATIONS/irradiance'][:].squeeze()
    QA3 = ds['BAND3_IRRADIANCE/STANDARD_MODE/OBSERVATIONS/quality_level'][:].squeeze()
    lamdas3 = ds['BAND3_IRRADIANCE/STANDARD_MODE/INSTRUMENT/calibrated_wavelength'][:].squeeze()
    data3[QA3 < 0.99] = np.nan



    # plt.plot(lamdar3[0,15:],datar3[0,15:],color='black',linewidth=0.5)
    # plt.plot(lamdar4[0, 0:497], datar4[0,0:497], color='black',linewidth=0.5)
    # plt.plot(lamdas3[0, :], data3[0,:], color='red',linewidth=1)
    # plt.plot(lamdas4[0, :], data4[0, :], color='red',linewidth=1)
    #plt.plot(lamdar[(lamdar>=300)&(lamdar<=500)],Er[(lamdar>=300)&(lamdar<=500)],color='blue',linewidth=0.05,linestyle='--')
    nrow3, nspec3 = data3.shape
    nrow4, nspec4 = data4.shape

    if 'irow' in kwargs:
        irow=kwargs['irow']
        rowrange=np.arange(1)+irow
    else:
        rowrange=np.arange(nrow3)

    ws3=np.zeros(nrow3)
    wg3 = np.zeros(nrow3)
    ws4 = np.zeros(nrow4)
    wg4 = np.zeros(nrow4)

    if DoCal:
        for row in rowrange:

            lamda = lamdas3[row, :]
            data = data3[row, :]
            reflamda = lamdar3
            refdata = datar3[row, :]
            windowinds = ((lamda >= (minwl-dlamda)) & (lamda <= (sepwl+dlamda))& (np.isnan(data) == False)).nonzero()   #(lamda >= (minwl-dlamda)) & (lamda <= (maxwl+dlamda)) &

            Fitresult = CaliIrr(lamda[windowinds], data[windowinds], reflamda, refdata, lamda0s[0],method=Optmethod)

            #ws3[row] = Fitresult.params['shift'].value
            #wg3[row] = Fitresult.params['stretch'].value

            ws3[row] = Fitresult[3]
            wg3[row] = Fitresult[4]



        for row in rowrange:

            lamda = lamdas4[row, :]
            data = data4[row, :]
            reflamda = lamdar4
            refdata = datar4[row, :]
            windowinds = ((lamda >= (sepwl-dlamda)) & (lamda <= (maxwl+dlamda))&(np.isnan(data) == False)).nonzero()  #
            Fitresult = CaliIrr(lamda[windowinds], data[windowinds], reflamda, refdata, lamda0s[1],method=Optmethod)
            # ws4[row] = Fitresult.params['shift'].value
            # wg4[row] = Fitresult.params['stretch'].value
            ws4[row] = Fitresult[3]
            wg4[row] = Fitresult[4]


        







    # plt.plot(lamdas3[0, :]+ws3[0]+wg3[0]*(lamdas3[0,:]-lamda0s[0]), data3[0, :], color='blue', linewidth=0.9,linestyle='--')
    # plt.plot(lamdas4[0, :]+ws4[0]+wg4[0]*(lamdas3[0,:]-lamda0s[0]), data4[0, :], color='blue', linewidth=0.9,linestyle='--')
    #
    # interpwls = np.array(lamdas3[0,:] + ws3[0] + wg3[0] * (lamdas3[0,:] - lamda0s[0]))
    # datainterp = interp1d(lamdar3[0,:],datar3[0,:], fill_value="extrapolate")
    # r1=pearsonr(datainterp(np.array(lamdas3[0, :])), data3[0, :])
    # r2=pearsonr(datainterp(interpwls), data3[0, :])
    # plt.text(300,8.e-6,'Band3: shift='+'{:10.2e}'.format(ws3[0]).strip()+' nm, stretch='+'{:10.2e}'.format(wg3[0]).strip()+'\n'+'r2='+'{:10.4f}'.format(r1[0]**2).strip()+' ('+'{:10.4f}'.format(r2[0]**2).strip()+')')#+', r2='+'{:10.4f}'.format(0]^2).strip()+' ('+'{:10.4f}'.format(pearsonr(datainterp(interpwls),data3[0,:])[0]^2).strip()+')'
    # interpwls = np.array(lamdas4[0, :] + ws4[0] + wg4[0] * (lamdas4[0, :] - lamda0s[1]))
    # datainterp = interp1d(lamdar4[0,:],datar4[0,:], fill_value="extrapolate")
    # r1 = pearsonr(datainterp(np.array(lamdas4[0, :])), data4[0, :])
    # r2 = pearsonr(datainterp(interpwls), data4[0, :])
    # plt.text(400, 3.e-6, 'Band4: shift=' + '{:10.2e}'.format(ws4[0]).strip() + ' nm, stretch=' + '{:10.2e}'.format(wg4[0]).strip() +'\n'+'r2=' + '{:10.4f}'.format(r1[0]**2).strip() + ' (' + '{:10.4f}'.format(r2[0]**2).strip() + ')')
    #
    # plt.savefig('/Users/chili/irr.png', dpi=600)

    # plt.plot(np.arange(nrow3), ws3, color='red')
    # plt.plot(np.arange(nrow4), ws4, color='blue')
    # plt.savefig('/Users/chili/ws.png', dpi=600)
    # plt.close()
    #
    # plt.plot(np.arange(nrow3),wg3,color='red')
    # plt.plot(np.arange(nrow4),wg4,color='blue')
    # plt.savefig('/Users/chili/wg.png', dpi=600)
    # plt.close()


    outdata = [lamdas3[:].squeeze(), ws3, wg3,data3,lamdas4[:].squeeze(), ws4, wg4,data4]

    ds.close()


    return outdata

