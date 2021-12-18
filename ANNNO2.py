from TROPOMI import ReadRads, ReadCalIrr,ReadNO2, ReadStatic, RemapRs, CalRadPix, ReadMCD43, BRDF, Converthgt, ReadO3
from MLPros import PCAapply
import glob
import numpy as np
from datetime import datetime
import pickle as pk
from netCDF4 import Dataset
import time
import os
import argparse


#machine learning model directories, one for all simulations, one for low concentration, one for high concentration
DataDir='/Users/chili/PythonCodes/ANNNO2/1Day/'
MLdir=DataDir+'ANNs/'
subdirs=['High','Low']
retpars=['TOTCol','Alt']
nretpar=len(retpars)
PCNo=[15,8,10]
nsubdir=len(subdirs)
npc=max(PCNo)


NAv=6.02214e23

#day=2

parser=argparse.ArgumentParser()
parser.add_argument('--year',type=int)
parser.add_argument('--month',type=int)
parser.add_argument('--day',type=int)
parser.add_argument('--Region')

args=parser.parse_args()
year=args.year
month=args.month
day=args.day
Region=args.Region

if Region=='NNA':
    latmin=39.
    lonmin=-125.
    latmax=49.
    lonmax=-70.


if Region=='SNA':
    latmin=29.
    lonmin=-125.
    latmax=39.
    lonmax=-70.

if Region=='CN':
    latmin=20.
    lonmin=100
    latmax=45.
    lonmax=140.

if Region=='SEU':
    latmin=38.
    lonmin=-5.
    latmax=47.
    lonmax=45.

if Region=='NEU':
    latmin=47.
    lonmin=-5.
    latmax=55.
    lonmax=45.

outdir=MLdir+Region+'/'

doy=(datetime(year,month,day) - datetime(year, 1, 1)).days + 1

FullCal=False
addTERRA=True


Concscaling=0.01
#Lowthres=0.7 #DU to separate low and high concentration conditions
# cwl=445. #nm
# wlhw=55. #nm, a 90 nm wide specral range
# minwl=cwl-wlhw
# maxwl=cwl+
minwl=390.
maxwl=495.
sepwl=401.
lamda0s=[(minwl+sepwl)/2,(sepwl+maxwl)/2]

instrument_rows=[0,50,100,150,200,250,300,350,400,449]
nretrow=len(instrument_rows)
#lamda0s=[362.5,451.5]

refwls=320.5+0.2*np.arange(891)
wlinds=np.array(((refwls >= minwl) & (refwls <= maxwl)).nonzero()).squeeze()
refwlrt=refwls[wlinds]

#machine learning model inputs:
varinputs=['SZA','VZA','RAA','O3','HSurf','RsPC1','RsPC2','RsPC3']  #,'AOD'
varinwb=np.array([0,1,2,4,12,13,14,15]) #index in the LUTmean and LUTstd data array
nvarinwb=len(varinwb)
Lowvarinds=[0,1,2,5,6,7]  #for column retrieval (low)
Highvarinds=[0,1,2,4,5,6,7]
Stratvarinds=[0,1,2,5,6,7]
scalings=np.array([1.,1.,1.,0.1, 0.01,10.,100.,100.])
nvtinputs=16

#pcinputs=np.arange(11)+1 #from PC2 to PC12
#outputinds=np.array([9,10])
#pcinputs=[0,1,2,3,4]  #[1,2,3,4,6,9,10,11]



#conversion from Mol/m2 to DU
Molm2DU=NAv/2.69e20



DoRingCor=True
#input data dirs

CaliDone=False
CaliDir=DataDir+'ANNs/Narrow/'+Region+'/'

#Irradiance
Irrdir=DataDir+'IRR/'
dlamda=0.2

#radiance and geometry
B3Raddir=DataDir+'L1B_B3/'
B4Raddir=DataDir+'L1B_B4/'

#spectral response function file
#SRFfile='/Users/chili/Downloads/isrf_release/isrf/binned_uvn_spectral_unsampled/S5P_OPER_AUX_SF_UVN_00000101T000000_99991231T235959_20180320T084215.nc'
#SAOsolarfile=DataDir+'/sao2010.solref.converted'
#NO2Xsecfile=DataDir+'/NO2_UV00.dat'
StaticFile=DataDir+'/S5P_OPER_REF_SOLAR__00000000T000000_99999999T999999_20180115T164926.nc'

Rspcfile=DataDir+'SurfPC/PC3.pkl'
Rspca = pk.load(open(Rspcfile, 'rb'))
RsPCsNorm=[0.54880311,0.42164358]

# #altitudes (m)
GMTEDfile=DataDir+'Glob5kmalt.nc'
ds=Dataset(GMTEDfile,'r')
altlat=ds['lat'][:]
altlon=ds['lon'][:]
globalt=ds['altitude'][:]
ds.close()


#WMfile=DataDir+'Glob5kmwatermask.nc'
#ds=Dataset(WMfile,'r')
#waterlat=ds['lat'][:]
#waterlon=ds['lon'][:]
#watermask=ds['water_mask'][:]
#ds.close()

# GMTEDfile=DataDir+'elevation_5KMmn_GMTEDmn.tif'
# im = Image.open(GMTEDfile)
# globalt = np.flip(np.asarray(im),axis=0)
# altres = np.array(im.tag[33550])
# altlu=np.array(im.tag[33922])
# im.close()
#
# dso = Dataset('/Users/chili/Glob5kmalt.nc', mode='w', format='NETCDF4')
# dso.createDimension('lon', (globalt.shape)[1])
# dso.createDimension('lat', (globalt.shape)[0])
#
# outdata = dso.createVariable('lat', np.float32, ('lat'))
# outdata.units = 'degree'
# outdata[:]=np.flip(altlu[4]-np.arange((globalt.shape)[0])*altres[0])
#
# outdata = dso.createVariable('lon', np.float32, ('lon'))
# outdata.units = 'degree'
# outdata[:]=altlu[3]+np.arange((globalt.shape)[1])*altres[1]
#
# outdata = dso.createVariable('altitude', np.float32, ('lat','lon'))
# outdata.units = 'm'
# outdata[:]=globalt
#
# dso.close()
#
# exit()

#land/water
# lcfile=DataDir+'consensus_full_class_12.tif'
# im = Image.open(lcfile)
# globalt = np.flip(np.asarray(im),axis=0)
# altres = np.array(im.tag[33550])
# altlu=np.array(im.tag[33922])
# im.close()
# print(altres,altlu,globalt.shape)
#
# dso = Dataset('/Users/chili/Glob1kmwatermask.nc', mode='w', format='NETCDF4')
# dso.createDimension('lon', (globalt.shape)[1]/5)
# dso.createDimension('lat', (globalt.shape)[0]/5)
#
# outdata = dso.createVariable('lat', np.float32, ('lat'))
# outdata.units = 'degree'
# outdata[:]=np.flip(altlu[4]-altres[0]*2-np.arange((globalt.shape)[0]/5)*altres[0]*5)
# print(np.flip(altlu[4]-altres[0]*2-np.arange((globalt.shape)[0]/5)*altres[0]*5))
# outdata = dso.createVariable('lon', np.float32, ('lon'))
# outdata.units = 'degree'
# outdata[:]=altlu[3]+altres[1]*2+np.arange((globalt.shape)[1]/5)*altres[1]*5
#
# outdata = dso.createVariable('water_mask', np.float32, ('lat','lon'))
# outdata.units = 'unitless'
# outdata[:]=rebin(globalt,[(globalt.shape)[0]/5,(globalt.shape)[1]/5])
#
# dso.close()
# exit()


LERwls=[345., 354., 367., 372., 376., 388., 406., 416., 418., 425., 442., 452., 488., 499.]

# LERinds=[3,4,5,6,7,9,10,11,12,13,15,16,20,22]
# OMILERfile=DataDir+'/OMI-Aura_L3-OMLER_2005m01-2009m12_v003-2010m0503t063707.he5'
# f = h5py.File(OMILERfile, 'r')
# ds = f["HDFEOS/GRIDS/EarthSurfaceReflectanceClimatology/"]  # OMI_Total_Column_Amount_SO2/Data_Fields/
# LERdata=ds['Data Fields/MonthlySurfaceReflectance']
# LERflag=ds['Data Fields/MonthlySurfaceReflectanceFlag']
# LERlat=ds['Data Fields/Latitude'][:]
# dLERlat=np.absolute(LERlat[1]-LERlat[0])
# LERlon=ds['Data Fields/Longitude'][:]
# dLERlon=np.absolute(LERlon[1]-LERlon[0])
#
# LERmonth=0.001*LERdata[month-1,LERinds,:,:].squeeze()
# flagmonth=LERflag[month-1,:,:].squeeze()
# flagmonth[(flagmonth != 185) & (flagmonth != 195) & (flagmonth != 230) & (flagmonth != 240) & (flagmonth != 250)]=0
# LERmonth[LERmonth<0]=np.nan
# LERmonth[np.stack([flagmonth]*len(LERinds),axis=0)<100]=np.nan
# f.close()


LERinds=[0,1,2,3,4,6,7,8,9,10,12,13,17,19]
OMIMERISfile=DataDir+'MERIS/SurfRefClim.nc'
strmonth=('{:10.0f}'.format(month+100).strip())[1:]
ds = Dataset(OMIMERISfile, 'r')
LERlat=ds['latitude'][:]
dLERlat=np.absolute(LERlat[1]-LERlat[0])
LERlon=ds['longitude'][:]
dLERlon=np.absolute(LERlon[1]-LERlon[0])
LERmonth=0.001*ds[strmonth][LERinds,:,:]
LERmonth[LERmonth<0]=np.nan
ds.close()

#read MODIS BRDF file
[MCDLat,MCDLon,Par1,Par2,Par3]=ReadMCD43(DataDir+'MCD43/MCD43C1/',year,doy)

if np.array(MCDLat).size<2:
    exit()


#radiance PCs,[pca0_All,pca50,....,pca449_All,pca0_Low,.....]
radpca=[]
for subdir in subdirs:
    for instrow in np.arange(nretrow):
        radpcfile = MLdir+subdir+'/' + 'pca_' + '{:10.0f}'.format(instrument_rows[instrow]).strip() + '.wb'
        radpca.append(pk.load(open(radpcfile, 'rb')))

for isubdir in np.arange(len(subdirs)):
    subdir=subdirs[isubdir]
    for instrow in np.arange(nretrow):
        LUTmeanstdfile = MLdir+subdir+'/' + 'meanstd_' + '{:10.0f}'.format(instrument_rows[instrow]).strip() + '.npy'

        with open(LUTmeanstdfile, 'rb') as f:
            if instrow == 0:
                submeanLUT = np.stack([np.array(np.load(f))] * 1, axis=1)
                substdLUT = np.stack([np.array(np.load(f))] * 1, axis=1)
                # meanradpc = np.stack([np.array(np.load(f))]*1,axis=1)
                substdradpc = np.stack([np.array(np.load(f))] * 1, axis=1)
            else:
                submeanLUT = np.append(submeanLUT, np.stack([np.array(np.load(f))] * 1, axis=1), axis=1)
                substdLUT = np.append(substdLUT, np.stack([np.array(np.load(f))] * 1, axis=1), axis=1)
                # meanradpc = np.append(meanradpc, np.stack([np.array(np.load(f))]*1,axis=1) , axis=1)
                substdradpc = np.append(substdradpc, np.stack([np.array(np.load(f))] * 1, axis=1), axis=1)

    if isubdir==0:
        meanLUT=np.stack([submeanLUT]*1,axis=2)
        stdLUT=np.stack([substdLUT]*1,axis=2)
        stdradpc=np.zeros([npc,nretrow,nsubdir])
        #stdradpc=np.stack([substdradpc]*1,axis=2)
    else:
        meanLUT = np.append(meanLUT,np.stack([submeanLUT]*1,axis=2),axis=2)
        stdLUT = np.append(stdLUT,np.stack([substdLUT]*1,axis=2),axis=2)
        #stdradpc = np.append(stdradpc,np.stack([substdradpc]*1,axis=2),axis=2)
    pcnumber=(substdradpc.shape)[0]

    stdradpc[0:pcnumber,:,isubdir]=substdradpc



ann=[]  #1-10 Anns for TROPCol (High), then 11-20 TOTCol, 21-30 Alt, 31-40 TOTCol for Low
annind=0
for subdir in subdirs:
    if subdir=='High':
        for retpar in retpars:

            for instrow in np.arange(nretrow):
                ANNfile = MLdir + subdir + '/' + 'ann_' + '{:10.0f}'.format(instrument_rows[instrow]).strip()+'.'+retpar + '.wb'
                ann.append(pk.load(open(ANNfile, 'rb')))
                annind=annind+1

                #print(subdir,retpar,instrow,annind)
    else:
        for instrow in np.arange(nretrow):
            ANNfile = MLdir+subdir+'/' + 'ann_' + '{:10.0f}'.format(instrument_rows[instrow]).strip() +'.'+retpars[0]+ '.wb'
            ann.append(pk.load(open(ANNfile, 'rb')))
            annind=annind+1

            #print(subdir, retpar, instrow,annind)



#AOD, 3 km resolution
#AODdir = DataDir+ 'MYD04/MYD04_3K/'

#AOD 1 degree resolution
#
#AODdir=DataDir+'/MYD08/MYD08_D3/'+stryear+'/'
# [AODDay,AODlat,AODlon]=ReadMOD08(AODdir,doy)
# if addTERRA:
#     TERRAdir=DataDir+'/MYD08/MOD08_D3/'+stryear+'/'
#     [TAODDay,TAODlat,TAODlon]=ReadMOD08(TERRAdir,doy)
#     AODDay[np.isnan(AODDay)]=TAODDay[np.isnan(AODDay)]


#AOD 0.01 degree resolution
# AODdir=DataDir+'MODIS_AOD_L3_HRG/MYD/'
# [AODDay,AODlat,AODlon]=ReadHRGAOD(AODdir,year,doy)
# if addTERRA:
#     TERRAdir=DataDir+'MODIS_AOD_L3_HRG/MOD/'
#     [TAODDay,TAODlat,TAODlon]=ReadHRGAOD(TERRAdir,year,doy)
#     AODDay[np.isnan(AODDay)]=TAODDay[np.isnan(AODDay)]
# if np.array(AODlat).size<2:
#     exit()
#OMI LER at multiple wavelength to extract the 3 PCs
LERfile=''
LERPCfile=''

#O3 dir to read the O3 column and LER at 320 nm for scaling from climatology OMI LER
O3dir=DataDir+'O3/'
NO2dir=DataDir+'NO2/'

#surface altitude
GLOBEdir=''



#
stryear='{:10.0f}'.format(year).strip()
strdate=stryear+('{:10.0f}'.format(month+100).strip())[1:]+('{:10.0f}'.format(day+100).strip())[1:]
B3Radfiles=glob.glob(B3Raddir+'S5P_OFFL_L1B_RA_BD3_'+strdate+'T*.nc')


#irradiance calibratino file
CaliTempfile=DataDir+'SolarCali/CaliIrr.'+strdate+'.wb'
#Do calibration of irradiance for this day, also read the reference (off-flight) calibrated wavelength
#[lamdar,Er]=ReadSAO(SAOsolarfile)
#[lamdar3,SR3,lamdar4,SR4]=ConvISRF(lamdar,Er,SRFfile)

[lamdar3,SR3,Ring3,lamdar4,SR4,Ring4]=ReadStatic(StaticFile)


# [lamdaNO2,XNO2]=ReadNO2Xsec(NO2Xsecfile)
# [lamdar3,NO2Xsec3,lamdar4,NO2Xsec4]=ConvISRF(lamdar,Er,SRFfile)


#one irradiance file in one day
Irrfile=glob.glob(Irrdir+'S5P_OFFL_L1B_IR_UVN_*_*_*_*_*_'+strdate+'T*.nc')[0]

if os.path.exists(CaliTempfile)==False:
    start = time.process_time()
    [lamdas3, irws3, irwg3, Irr3, lamdas4, irws4, irwg4, Irr4] = ReadCalIrr(lamdar3, SR3, lamdar4, SR4, Irrfile, minwl,sepwl,maxwl, lamda0s,method='lmfit')  # ,DoCal=False) #
    # exit()
    print('Irradiance Calibration:', time.process_time() - start)

    with open(CaliTempfile, 'wb') as f:
        np.save(f, irws3)
        np.save(f, irwg3)
        np.save(f, irws4)
        np.save(f, irwg4)

else:
    start = time.process_time()
    [lamdas3, irws3, irwg3, Irr3, lamdas4, irws4, irwg4, Irr4] = ReadCalIrr(lamdar3, SR3, lamdar4, SR4, Irrfile, minwl,sepwl,maxwl, lamda0s,DoCal=False,method='lmfit')





    with open(CaliTempfile, 'rb') as f:
        irws3=np.load(f)
        irwg3=np.load(f)
        irws4=np.load(f)
        irwg4=np.load(f)




for B3Radfile in B3Radfiles:

    B3Base = B3Radfile.split('/')[-1].split('_L1B_RA_BD3_')
    [Rads3,SZAs,VZAs,RAAs,Lats,Lons,minscan,nscan,minrow,nrow]=ReadRads(B3Radfile,Range=[latmin,lonmin,latmax,lonmax])

    if Rads3.size<2:
        continue

    scani = np.arange(nscan).astype(int) + minscan
    rowi = np.arange(nrow).astype(int) + minrow
    maxscan = np.max(scani)
    maxrow = np.max(rowi)

    #AOD = ReadRegAOD(AODdir, year,doy, Lats, Lons)
    #AOD[np.isnan(AOD)]=np.nanmean(AOD)
    #print(AODs.shape)
    #AOD=np.zeros(Lats.shape)+0.29
    Rs = RemapRs(LERmonth,LERlat,LERlon,Lats,Lons)



    riso = RemapRs(Par1, MCDLat, MCDLon, Lats, Lons)
    rvol = RemapRs(Par2, MCDLat, MCDLon, Lats, Lons)
    rgeo = RemapRs(Par3, MCDLat, MCDLon, Lats, Lons)


    #BRDF definition of RAA is 0 for forward scattering...
    MCDRs = BRDF(SZAs,VZAs,180.-RAAs,riso,rvol,rgeo)
    Rscals = MCDRs / np.mean(Rs[11:13, :, :], axis=0)
    Rs = Rs * np.stack([Rscals] * len(LERinds), axis=0)

    #wm = RemapRs(watermask,waterlat,waterlon,Lats,Lons)
    alt = RemapRs(globalt,altlat,altlon,Lats,Lons)

    #AOD = RemapRs(AODDay,AODlat,AODlon,Lats,Lons)
    #AODnofill=AOD.copy()
    #AOD = fill(AOD,None)

    #Dist=(Lats-33.755409)**2+(Lons+84.38850133)**2


    B4Radfile=B4Raddir+B3Base[0]+'_L1B_RA_BD4_'+B3Base[1]
    try:
        Rads4 = ReadRads(B4Radfile,DataOnly=True,iscan=scani,irow=rowi)
    except:
        print('Failed reading RADIANCE:', B4Radfile)
        continue
    nscan, nrow, nspec = Rads3.shape
    nrec = nscan * nrow


    # # read O3
    # O3file=(glob.glob(O3dir+B3Base[0]+'_L2__O3_____'+(B3Base[1])[0:38]+'*'))[0]
    # [TO3,O3QA]=ReadO3(O3file)
    # TO3=TO3[minscan:maxscan + 1, minrow:maxrow + 1]*Molm2DU

    # # # read SO2
    # SO2file = (glob.glob(O3dir + B3Base[0] + '_L2__SO2____' + (B3Base[1])[0:38] + '*'))[0]
    # TSO2 = ReadSO2(SO2file)* Molm2DU

    # read NO2
    NO2file = (glob.glob(NO2dir + B3Base[0] + '_L2__NO2____' + (B3Base[1])[0:38]+'*'))[0]

    #totalcolumn, tropospheric column, quality flag, cloud fraction, cloud albedo , cloud pressure, LER
    [SNO2Col,SNO2gCol,SNO2TCol,SNO2QA,CldFr,CldAbd,Cldpr]=ReadNO2(NO2file)  #mol m-2, unitless and meters

    # Rscals = LER420[minscan:maxscan+1,minrow:maxrow+1].reshape([nscan,nrow])/np.mean(Rs[6:12,:,:],axis=0)
    #
    # Rs=Rs*np.stack([Rscals]*len(LERinds),axis=0)

    # print((SNO2Col[:,291]*Molm2DU>0.7).nonzero())
    # exit()
    # #

    #convert cloud albedo to spectrum
    #convert cloud pressure to altitude
    CldAbd[(CldAbd<0.)|(np.isnan(CldAbd))]=999.
    RsCld=np.stack([CldAbd[minscan:maxscan+1,minrow:maxrow+1]]*len(LERinds),axis=0)
    HCld=Converthgt(Cldpr[minscan:maxscan+1,minrow:maxrow+1])


    outfile = outdir + B3Base[0] + '_LCNO2_' + B3Base[1]

    if CaliDone:
        Califile=CaliDir+B3Base[0] + '_LCNO2_' + B3Base[1]
        try:
            dsc = Dataset(Califile, 'r')
            Caliws3 = dsc['band3_shift'][:]
            Caliws4 = dsc['band4_shift'][:]
            Caliwg3 = dsc['band3_stretch'][:]
            Caliwg4 = dsc['band4_stretch'][:]
            Calirc3 = dsc['band3_RingCoeff'][:]
            Calirc4 = dsc['band4_RingCoeff'][:]
            dsc.close()
        except:
            continue

    dso = Dataset(outfile, mode='w', format='NETCDF4')
    dso.createDimension('x', nrow)
    dso.createDimension('y', nscan)

    dso.createDimension('used_row', nretrow)

    outdata = dso.createVariable('latitude', np.float32, ('y', 'x'))
    outdata.units = 'degree'
    outdata[:]=Lats

    outdata = dso.createVariable('rows', np.float32, ('x'))
    outdata.units = 'unitless'
    outdata[:]=rowi

    outdata = dso.createVariable('scans', np.float32, ('y'))
    outdata.units = 'unitless'
    outdata[:] = scani


    outdata = dso.createVariable('longitude', np.float32, ('y', 'x'))
    outdata.units = 'degree'
    outdata[:] = Lons

    outdata = dso.createVariable('NO2_std_TOTCol', np.float32, ('y', 'x'))
    outdata.units = 'DU'
    outdata[:] = SNO2Col[minscan:maxscan+1,minrow:maxrow+1] * Molm2DU

    outdata = dso.createVariable('NO2_std_GhostCol', np.float32, ('y', 'x'))
    outdata.units = 'DU'
    outdata[:] = SNO2gCol[minscan:maxscan + 1, minrow:maxrow + 1] * Molm2DU


    outdata = dso.createVariable('NO2_std_TROPCol', np.float32, ('y', 'x'))
    outdata.units = 'DU'
    outdata[:] = SNO2TCol[minscan:maxscan + 1, minrow:maxrow + 1] * Molm2DU

    outdata = dso.createVariable('NO2_std_QA', np.float32, ('y', 'x'))
    outdata.units = 'unitless'
    outdata[:] = SNO2QA[minscan:maxscan + 1, minrow:maxrow + 1]

    # outdata = dso.createVariable('O3TCol', np.float32, ('y', 'x'))
    # outdata.units = 'DU'
    # outdata[:] = TO3
    #
    # outdata = dso.createVariable('O3_QA', np.float32, ('y', 'x'))
    # outdata.units = 'unitless'
    # outdata[:] = O3QA[minscan:maxscan + 1, minrow:maxrow + 1]

    outdata = dso.createVariable('Cloud_Fraction', np.float32, ('y', 'x'))
    outdata.units = 'unitless'
    outdata[:] = CldFr[minscan:maxscan + 1, minrow:maxrow + 1]

    outdata = dso.createVariable('Cloud_Albedo', np.float32, ('y', 'x'))
    outdata.units = 'unitless'
    outdata[:] = CldAbd[minscan:maxscan + 1, minrow:maxrow + 1]

    outdata = dso.createVariable('Cloud_Pressure', np.float32, ('y', 'x'))
    outdata.units = 'hpa'
    outdata[:] = Cldpr[minscan:maxscan + 1, minrow:maxrow + 1]

    outdata = dso.createVariable('Cloud_altitude', np.float32, ('y', 'x'))
    outdata.units = 'm'
    outdata[:] = HCld


    outdata = dso.createVariable('SurfAltitude', np.float32, ('y', 'x'))
    outdata.units = 'm'
    outdata[:] = alt

    outdata = dso.createVariable('SurfAlbedo', np.float32, ('y', 'x'))
    outdata.units = 'unitless'
    outdata[:] = np.mean(Rs[5:13,:,:],axis=0)

    outdata = dso.createVariable('ScalarAlbedo', np.float32, ('y', 'x'))
    outdata.units = 'unitless'
    outdata[:] = Rscals

    outdata = dso.createVariable('SolarZenithAngle', np.float32, ('y', 'x'))
    outdata.units = 'degree'
    outdata[:] = SZAs

    outdata = dso.createVariable('SatZenithAngle', np.float32, ('y', 'x'))
    outdata.units = 'degree'
    outdata[:] = VZAs

    outdata = dso.createVariable('RelAzimuthAngle', np.float32, ('y', 'x'))
    outdata.units = 'degree'
    outdata[:] = RAAs

    # outdata = dso.createVariable('O3', np.float32, ('y', 'x'))
    # outdata.units = 'DU'
    # outdata[:] = TO3[minscan:maxscan+1,minrow:maxrow+1]

    # outdata = dso.createVariable('SO2', np.float32, ('y', 'x'))
    # outdata.units = 'DU'
    # outdata[:] = TSO2[minscan:maxscan + 1, minrow:maxrow + 1]

    # outdata = dso.createVariable('AOD', np.float32, ('y', 'x'))
    # outdata.units = 'unitless'
    # outdata[:] = AOD
    #
    # outdata = dso.createVariable('AOD_nofill', np.float32, ('y', 'x'))
    # outdata.units = 'unitless'
    # outdata[:] = AODnofill

    outNO2C = np.zeros([nscan, nrow]) - 9999
    outNO2H = np.zeros([nscan, nrow]) - 9999
    outAOD = np.zeros([nscan, nrow]) - 9999

    outws3 = np.zeros([nscan, nrow]) - 9999
    outringc3 = np.zeros([nscan, nrow]) - 9999
    outwg3 = np.zeros([nscan, nrow]) - 9999

    outws4 = np.zeros([nscan, nrow]) - 9999
    outringc4 = np.zeros([nscan, nrow]) - 9999
    outwg4 = np.zeros([nscan, nrow]) - 9999



    #Calibration of L1B Radiance and calculate I/L for the wavelengthes
    stokes=np.zeros([nscan,nrow,len(refwlrt)])
    stokesflag=np.zeros([nscan,nrow],dtype=int)+1
    caliradcount=0
    #retcount=0
    start = time.process_time()
    for ipx in np.arange(nscan):

        for ipy in np.arange(nrow):
            if (np.count_nonzero(np.isnan(Rs[:,ipx,ipy]))>0):  #(np.isnan(AOD[ipx,ipy])) | np.isnan(TO3[ipx+scani[0],ipy+rowi[0]]) |

                stokesflag[ipx,ipy]=0
                continue


            if caliradcount==0:
                [radws3,radws4,radwg3,radwg4,RingC3,RingC4,outstokes]=CalRadPix(lamdas3[ipy+rowi[0],:],lamdar3,SR3[ipy+rowi[0],:],Irr3[ipy+rowi[0],:],\
                                                                  lamdas4[ipy+rowi[0],:],lamdar4,SR4[ipy+rowi[0],:],Irr4[ipy+rowi[0],:],\
                                                                  Rads3[ipx,ipy,:],Rads4[ipx,ipy,:],Ring3[ipy+rowi[0],:],Ring4[ipy+rowi[0],:],\
                                                                  refwlrt,lamda0s,irws3[ipy+rowi[0]],irwg3[ipy+rowi[0]],irws4[ipy+rowi[0]],\
                                                                  irwg4[ipy+rowi[0]],minwl,sepwl,maxwl,dlamda=dlamda,FullCal=FullCal,CaliDone=CaliDone,method='lmfit')#,CaliData=[Caliws3[ipx, ipy],Caliws4[ipx, ipy],Caliwg3[ipx, ipy],Caliwg4[ipx, ipy],Calirc3[ipx, ipy],Calirc4[ipx, ipy]])
                # CaliData=[Caliws3[ipx,ipy],Caliws4[ipx,ipy],Caliwg3[ipx,ipy],Caliwg4[ipx,ipy],Calirc3[ipx,ipy],Calirc4[ipx,ipy]],retcount=retcount)
            else:
                [radws3, radws4, radwg3, radwg4, RingC3, RingC4, outstokes] = CalRadPix(lamdas3[ipy + rowi[0], :],
                                                                                        lamdar3, SR3[ipy + rowi[0], :],
                                                                                        Irr3[ipy + rowi[0], :], \
                                                                                        lamdas4[ipy + rowi[0], :],
                                                                                        lamdar4, SR4[ipy + rowi[0], :],
                                                                                        Irr4[ipy + rowi[0], :], \
                                                                                        Rads3[ipx, ipy, :],
                                                                                        Rads4[ipx, ipy, :],
                                                                                        Ring3[ipy + rowi[0], :],
                                                                                        Ring4[ipy + rowi[0], :], \
                                                                                        refwlrt, lamda0s,
                                                                                        irws3[ipy + rowi[0]],
                                                                                        irwg3[ipy + rowi[0]],
                                                                                        irws4[ipy + rowi[0]], \
                                                                                        irwg4[ipy + rowi[0]], minwl,sepwl,maxwl, dlamda=dlamda,FullCal=FullCal,CaliDone=CaliDone,inits=newinits,method='lmfit')#,CaliData=[Caliws3[ipx, ipy],Caliws4[ipx, ipy],Caliwg3[ipx, ipy],Caliwg4[ipx, ipy],Calirc3[ipx, ipy],Calirc4[ipx, ipy]])
                # CaliData=[Caliws3[ipx,ipy],Caliws4[ipx,ipy],Caliwg3[ipx,ipy],Caliwg4[ipx,ipy],Calirc3[ipx,ipy],Calirc4[ipx,ipy]],retcount=retcount)


            if (np.array(outstokes).size < 2) | (np.min(outstokes) <= 0.):
                stokesflag[ipx, ipy] = 0
                continue

            caliradcount=1
            newinits=[radws3, radws4, radwg3,radwg4,RingC3, RingC4]
            #retcount=retcount+1

            outws3[ipx,ipy] = radws3
            outws4[ipx, ipy] = radws4
            outringc3[ipx,ipy]=RingC3
            outringc4[ipx, ipy] = RingC4
            outwg3[ipx, ipy] = radwg3
            outwg4[ipx, ipy] = radwg4
            stokes[ipx,ipy,:]=outstokes

    print('Radiance Calibration:', time.process_time() - start)

    #input matrix indata[nrec,nvar]
    valinds=np.array((stokesflag.flatten()==1).nonzero()).flatten()


    invalinds=np.array((stokesflag.flatten()==0).nonzero()).flatten()
    nvalid=valinds.size


    if nvalid<1:
        continue



    # PCA apply to the Rs data and apply the scaling due to TROPOMI LER
    Rs=Rs.reshape([len(LERinds),nrec]).transpose()
    RsCld = RsCld.reshape([len(LERinds),nrec]).transpose()

    RsPCs = PCAapply(Rspca, (Rs[valinds,:] - RsPCsNorm[0]) / RsPCsNorm[1],Standardize=False)
    cldRsPCs = PCAapply(Rspca, (RsCld[valinds,:] - RsPCsNorm[0]) / RsPCsNorm[1], Standardize=False)

    # print('surfR:',Rs[valinds[0],:],RsPCs[0,:])
    # print('input data: ', SZAs.flatten()[valinds[0]],VZAs.flatten()[valinds[0]],RAAs.flatten()[valinds[0]],AOD.flatten()[valinds[0]],TO3[minscan:maxscan+1,minrow:maxrow+1].flatten()[valinds[0]],SurfHs[minscan:maxscan+1,minrow:maxrow+1].flatten()[valinds[0]])
    # print('means: ', meanLUT[varinwb,0:2])
    # print('stoke: ',stokes.reshape([nrec,len(refwlrt)])[valinds[0],:])

    #Do the retrieval

    #1. prepare input matrixd
    # iret for different concentration groups (ALL, LOW, HIGH...)

    start = time.process_time()
    for iret in np.arange(nsubdir):

        subdir=subdirs[iret]

        for iscene in np.arange(2):  #clear and cloudy scene


            if iscene==0:
                inputRs=RsPCs
                inputalt=alt
                sceneflag='clear'
            else:
                inputRs=cldRsPCs
                inputalt=HCld
                sceneflag='cloudy'

            if subdir == 'High':
                inretpars = retpars
                outputindsary = np.array([8, 10])

            else:
                inretpars = retpars[0:1]
                if subdir=='Trop':
                    outputindsary=np.array([9])
                else:
                    outputindsary = np.array([8])

            outNO2C = np.zeros([nretrow, nscan, nrow])

            if subdir == 'High':
                outNO2H = np.zeros([nretrow, nscan, nrow])

            for iretpar in np.arange(len(inretpars)):
                retpar=inretpars[iretpar]
                outputinds=[outputindsary[iretpar]]
                if subdir=='High':
                    if retpar=='Alt':
                        pcinputs=np.array([4,7,11,12,13,14])
                    else:
                        pcinputs=np.array([4,12])
                elif subdir=='Strat':
                    pcinputs = np.array([5,6,7,8,9])
                # elif subdir=='Trop':
                #     pcinputs = np.array([4])
                else:
                    pcinputs=np.array([5,6,7])
                #print(subdir,retpar,pcinputs)
                nvar = len(varinwb) + len(pcinputs)
                indata = np.zeros([nrec, nvar])
                indata[:, 0] = (SZAs.flatten() * scalings[0] - meanLUT[varinwb[0], 0, iret]) / stdLUT[varinwb[0], 0, iret]
                indata[:, 1] = (VZAs.flatten() * scalings[1] - meanLUT[varinwb[1], 0, iret]) / stdLUT[varinwb[1], 0, iret]
                indata[:, 2] = (RAAs.flatten() * scalings[2] - meanLUT[varinwb[2], 0, iret]) / stdLUT[varinwb[2], 0, iret]
                #indata[:, 3] = (AOD.flatten() * scalings[3] - meanLUT[varinwb[3], 0, iret]) / stdLUT[varinwb[3], 0, iret]
                #indata[:, 3] = (TO3.flatten() * scalings[3] - meanLUT[varinwb[3], 0, iret]) / stdLUT[varinwb[3], 0, iret]
                # indata[:, 4] = (TO3[minscan:maxscan + 1, minrow:maxrow + 1].flatten() * scalings[4] - meanLUT[varinwb[4],0]) / \
                #                stdLUT[varinwb[4],0]

                indata[:, 4] = (inputalt.flatten() * scalings[4] - meanLUT[varinwb[4], 0, iret]) / stdLUT[varinwb[4], 0, iret]

                for irs in np.arange(3):
                    indata[valinds, irs + 5] = (inputRs[:, irs] * scalings[irs + 5] - meanLUT[varinwb[irs + 5], 0, iret])/stdLUT[varinwb[irs + 5], 0, iret]


                # ANN regression for each row
                for instrow in np.arange(nretrow):

                    if subdir=='High':
                        rowann=ann[instrow + nretrow * iretpar]  #1-30
                    else:
                        rowann=ann[instrow + (iret-1)*nretrow+nretrow * nretpar]  #31-50


                    #print(subdir,retpars[iretpar],instrow + nretrow * iretpar,instrow + (iret-1)*nretrow+nretrow * nretpar)

                    Radrt = (np.log10(stokes.reshape([nrec, len(refwlrt)])) - meanLUT[nvtinputs, instrow, iret]) / stdLUT[
                        nvtinputs, instrow, iret]  # Do the log of the radiance/irradiance
                    Radpcs = PCAapply(radpca[instrow + nretrow * iret], Radrt[valinds, :], Standardize=False)

                    # varinputs = ['SZA', 'VZA', 'RAA', 'AOD', 'O3', 'HSurf', 'RsPC1', 'RsPC2', 'RsPC3']

                    pcadata = np.zeros([nrec, len(pcinputs)])
                    # radpcmean=np.stack([meanradpc[pcinputs,instrow].flatten()]*nvalid,axis=0)
                    radpcstd = np.stack([stdradpc[pcinputs, instrow, iret].flatten()] * nvalid, axis=0)
                    pcadata[valinds, :] = Radpcs[:, pcinputs] / radpcstd

                    indata[:, nvarinwb:] = pcadata

                    # (indata[valinds,:])[:,9:]=Radpcs[:,pcinputs]
                    # print(indata[valinds[0:2],:])

                    annout = np.zeros([nrec, len(outputinds)])
                    if (subdir=='High'):
                        addhighinds = np.append(Highvarinds, np.arange(nvar - nvarinwb) + nvarinwb)
                        anninputs = (indata[valinds, :])[:, addhighinds]
                    elif (subdir=='Low'):  #for both strat and TOTCol
                        addlowinds=np.append(Lowvarinds,np.arange(nvar-nvarinwb)+nvarinwb)
                        anninputs=(indata[valinds,:])[:,addlowinds]
                    else:
                        addstratinds = np.append(Stratvarinds, np.arange(nvar - nvarinwb) + nvarinwb)
                        anninputs = (indata[valinds, :])[:, addstratinds]
                    #print(subdir,anninputs.shape)
                    #print(subdir,anninputs.shape)
                    # if len(outputinds)==1:
                    #    anninputs=anninputs.flatten()
                    # print(instrow,iret)
                    rowout=rowann.predict(anninputs)
                    #print(rowout.shape)
                    if len(outputinds) == 1:
                        rowout = np.stack([rowout] * 1, axis=1)
                    rowout=rowout.reshape([nvalid, len(outputinds)])
                    annout[valinds, :] = rowout* np.stack([stdLUT[outputinds, 0, iret]] * nvalid, axis=0) \
                                         + np.stack([meanLUT[outputinds, 0, iret]] * nvalid, axis=0)

                    if retpar!='Alt':
                        annout = annout * Concscaling

                    annout[invalinds, :] = np.nan

                    if retpar=='TOTCol':
                        outNO2C[instrow, :, :] = annout.reshape([nscan, nrow])

                    if retpar=='Alt':
                        outNO2H[instrow, :, :] = annout.reshape([nscan, nrow])

            outdata = dso.createVariable('NO2_TOTCol'+'_'+sceneflag + '_' + subdir, np.float32, ('used_row', 'y', 'x'))
            outdata.units = 'DU'
            outdata[:] = outNO2C

            if subdir=='High':
                outdata = dso.createVariable('NO2_TROPHeight' +'_'+sceneflag+ '_' + subdir, np.float32, ('used_row', 'y', 'x'))
                outdata.units = 'km'
                outdata[:] = outNO2H
    print('ANN application:', time.process_time() - start)

    # #For each pixel
    # for irow in rowrange:
    #
    #     # if FoundOne:
    #     #     break
    #
    #     wls3=lamdas3[irow,:]
    #     Rs3=Irr3[irow,:]
    #     wls4=lamdas4[irow,:]
    #     Rs4 = Irr4[irow, :]
    #     # NO2Cross3 = NO2Xsec3[irow,:]
    #     # NO3Cross4 = NO2Xsec4[irow,:]
    #
    #
    #
    #     #wavelength for irradiance after calibration
    #     wln3=wls3+irws3[irow]+irwg3[irow]*(wls3-lamda0s[0])
    #     wln4=wls4+irws4[irow]+irwg4[irow]*(wls4-lamda0s[1])
    #
    #     for iscan in scanrange:
    #
    #         # if FoundOne:
    #         #     break
    #
    #         plat=Lats[iscan,irow]
    #         plon=Lons[iscan,irow]
    #
    #
    #         #Find the LER flag and values
    #         LERyind=np.array((((LERlat-plat)<=0.5*dLERlat)&((LERlat-plat)>-0.5*dLERlat)).nonzero()).flatten()
    #         LERxind =np.array((((LERlon-plon)<=0.5*dLERlon)&((LERlon-plon)>-0.5*dLERlon)).nonzero()).flatten()
    #
    #         if np.array(LERyind).size<1:
    #             print(plat,plon,LERyind,LERxind)
    #             continue
    #
    #         if np.array(LERxind).size<1:
    #             print(plat,plon,LERyind,LERxind)
    #             continue
    #
    #         if np.array(LERyind).size>1:
    #             print(plat,plon,LERyind,LERxind)
    #             continue
    #
    #         if np.array(LERxind).size>1:
    #             print(plat,plon,LERyind,LERxind)
    #             continue
    #
    #         if flagmonth[LERyind,LERxind].size>1:
    #             print(plat,plon,LERyind,LERxind,flagmonth.shape)
    #             continue
    #
    #         if flagmonth[LERyind,LERxind]<100:
    #             continue
    #
    #         o3col=TO3[iscan,irow]*Molm2DU
    #         RsNO2=LER420[iscan,irow]
    #         HSurf=SurfHs[iscan,irow]
    #
    #         if ((np.isnan(o3col)) | (np.isnan(RsNO2))):
    #             continue
    #
    #         AOD = ExtracMOD04(AODdir, doy, plat, plon)
    #
    #         #AOD = 0.1
    #
    #         if np.isnan(AOD):
    #             AOD = 0.2
    #             #continue
    #         if OneSpec==False:
    #             Rad3 = Rads3[iscan, irow, :]
    #             Rad4 = Rads4[iscan, irow, :]
    #         else:
    #             Rad3 = Rads3
    #             Rad4 = Rads4
    #         # radiance calibration
    #         windowinds = ((wls3 >= (minwl - dlamda)) & (wls3 <= (maxwl + dlamda)) & (np.isnan(Rad3) == False)).nonzero()
    #         if (np.count_nonzero(~np.isnan(Rad3[windowinds])) < 0.9 * len(Rad3[windowinds]) ):
    #             continue
    #
    #         RadCal3 = CalRads(wls3[windowinds], Rad3[windowinds], lamdar3, SR3[irow,:], lamda0s[0], Ring3[irow,:])
    #
    #         fig,ax=plt.subplots()
    #
    #         ax.set_xlim(minwl,maxwl)
    #         ax.plot(lamdar3,SR3[irow,:],color='red',linewidth=0.3)
    #         zax=ax.twinx()
    #         zax.plot(wls3[windowinds],Rad3[windowinds],color='blue',linewidth=0.3)
    #         wld3 = np.array(wls3 + RadCal3[4] + RadCal3[5] * (wls3 - lamda0s[0]))
    #         zax.plot(wld3[windowinds],Rad3[windowinds],color='green',linewidth=0.3)
    #         sr2rad=SR3[irow,:]*(RadCal3[0]+RadCal3[1]*(lamdar3-lamda0s[0])+RadCal3[2]*((lamdar3-lamda0s[0])**2))*(1+RadCal3[3]*Ring3[irow,:])
    #         zax.plot(lamdar3[(lamdar3>=np.min(wld3[windowinds]))&(lamdar3<=np.max(wld3[windowinds]))],sr2rad[(lamdar3>=np.min(wld3[windowinds]))&(lamdar3<=np.max(wld3[windowinds]))],color='orange',linewidth=0.3)
    #
    #
    #
    #
    #         windowinds = ((wls4 >= (minwl - dlamda)) & (wls4 <= (maxwl + dlamda)) & (np.isnan(Rad4) == False)).nonzero()
    #         if (np.count_nonzero(~np.isnan(Rad4[windowinds])) < 0.9 * len(Rad4[windowinds]) ):
    #             continue
    #
    #         RadCal4 = CalRads(wls4[windowinds], Rad4[windowinds], lamdar4, SR4[irow,:], lamda0s[1], Ring4[irow,:])
    #
    #
    #
    #
    #         ax.plot(lamdar4, SR4[irow, :], color='red',linewidth=0.3)
    #
    #         zax.plot(wls4[windowinds], Rad4[windowinds], color='blue',linewidth=0.3)
    #         wld4 = np.array(wls4 + RadCal4[4] + RadCal4[5] * (wls4 - lamda0s[0]))
    #         zax.plot(wld4[windowinds], Rad4[windowinds], color='green',linewidth=0.3)
    #         sr2rad = SR4[irow, :] * (RadCal4[0] + RadCal4[1] * (lamdar4 - lamda0s[1]) + RadCal4[2] * ((lamdar4 - lamda0s[1]) ** 2))*(1+RadCal4[3]*Ring4[irow,:])
    #         zax.plot(lamdar4[(lamdar4>=np.min(wld4[windowinds]))&(lamdar4<=np.max(wld4[windowinds]))], sr2rad[(lamdar4>=np.min(wld4[windowinds]))&(lamdar4<=np.max(wld4[windowinds]))], color='orange',linewidth=0.3)
    #         plt.savefig('/Users/chili/Cali.png', dpi=600)
    #         plt.close()
    #
    #
    #         if OneSpec==False:
    #             outws3[iscan, irow] = RadCal3[4]
    #             outws4[iscan, irow] = RadCal4[4]
    #             outwg3[iscan, irow] = RadCal3[5]
    #             outwg4[iscan, irow] = RadCal4[5]
    #
    #         psza = SZAs[iscan, irow]
    #         pvza = VZAs[iscan, irow]
    #         praa = RAAs[iscan, irow]
    #
    #         LERRs = LERmonth[:, LERyind, LERxind].flatten()
    #         if np.array(LERRs).size<1:
    #             continue
    #
    #         LERRs = LERRs * RsNO2 / np.mean(LERRs[6:12])
    #
    #         if np.count_nonzero(np.isnan(LERRs))>0:
    #             continue
    #             # Calculate input value as 3 Rs PC scores
    #
    #         RsPCs =PCAapply(Rspca, np.stack([(LERRs - RsPCsNorm[0]) / RsPCsNorm[1]]*1,axis=0), Standardize=False).flatten()
    #         if np.count_nonzero(np.isnan(RsPCs))>0:
    #             continue
    #
    #         # # Do radiance calibration for this pixel
    #         # validins=(wls>).nonzero()
    #         # [ws3,wg3]=CalRads(wln3,Rad3, wls3, Rs3, minwl, maxwl, lamda0s[0])      #, NO2Cross3, Ring3
    #         # [ws4, wg4] = CalRads(wln4,Rad4, wls4, Rs4, minwl, maxwl, lamda0s[1])  #, NO2Cross4, Ring4
    #
    #         # apply same wavelength calibration for iiradiance for now...
    #         # Calculate I/E as the VLIDORT output and PCA score for the 2-12th pcs...
    #         wld3=np.array(wls3+RadCal3[4]+RadCal3[5]*(wls3-lamda0s[0]))
    #         wld4=np.array(wls4+RadCal4[4]+RadCal4[5]*(wls4-lamda0s[1]))
    #         wln=np.append(wln3,wln4)
    #         Rs=np.append(Rs3,Rs4)
    #         wld=np.append(wld3,wld4)
    #
    #
    #
    #
    #         Rad = np.append(Rad3, Rad4)
    #
    #
    #         if DoRingCor:
    #             Ring3interp = interp1d(lamdar3, Ring3[irow, :], fill_value="extrapolate")
    #             Ring4interp = interp1d(lamdar4, Ring4[irow, :], fill_value="extrapolate")
    #             RingCor=np.append(1./(Ring3interp(wld3)*RadCal3[3]+1.),1./(Ring4interp(wld4)*RadCal4[3]+1.))
    #             Rad=Rad.flatten()*RingCor.flatten()
    #
    #
    #         #datainterp = interp1d(wl[(np.isnan(Rad)==False)&((wl<=393.)|((wl>=394)&(wl<=396.5)) | (wl>=397.5))], Rad[(np.isnan(Rad)==False)&((wl<=393.)|((wl>=394)&(wl<=396.5)) | (wl>=397.5))], fill_value="extrapolate")
    #         Radinterp = interp1d(wld[(np.isnan(Rad) == False)],Rad[(np.isnan(Rad) == False)],fill_value="extrapolate")
    #         Rsinterp = interp1d(wln[(np.isnan(Rs) == False)], Rs[(np.isnan(Rs) == False)], fill_value="extrapolate")
    #
    #         #print(refwlrt[(refwlrt>392)&(refwlrt<398)],datainterp(refwlrt[(refwlrt>392)&(refwlrt<398)]))
    #         #continue
    #         stokes=(Radinterp(refwlrt)/Rsinterp(refwlrt)).flatten()
    #
    #         fig,ax=plt.subplots()
    #         ax.plot(refwlrt,stokes)
    #         ax.set_xlim(minwl,maxwl)
    #         zax=ax.twinx()
    #         if DoRingCor:
    #             zax.plot(wld,RingCor,color='red')
    #         #zax.plot(lamdar4, 1+RadCal4[3]*Ring4[irow, :], color='red')
    #         plt.savefig('/Users/chili/OneSpec.png',dpi=600)
    #         plt.close()
    #
    #         if np.count_nonzero((stokes==np.nan) | (stokes==np.inf))>0:
    #             continue
    #
    #         Radrt = np.log10(np.stack([stokes]*1,axis=0))  #Do the log of the radiance/irradiance
    #
    #         Radrt=(Radrt-meanLUT[nvtinputs])/stdLUT[nvtinputs]
    #         if np.count_nonzero((np.isnan(Radrt))|(np.isinf(Radrt)))>0:
    #             continue
    #         Radpcs=PCAapply(radpca,Radrt,Standardize=False).flatten()
    #         if np.count_nonzero(np.isnan(Radpcs))>0:
    #             continue
    #
    #         VTinputs=np.append([psza, pvza, praa, AOD, o3col, HSurf], RsPCs[0:3].flatten()).flatten() * scalings
    #
    #         # print(flagmonth[LERyind,LERxind],plat,plon,LERlat[LERyind],LERlon[LERxind])
    #         # print(Radrt,Radpcs)
    #         # print(VTinputs)
    #         # exit()
    #
    #
    #         # print(VTinputs/scalings)
    #         # print(Radpcs[1:])
    #         # print(meanLUT[0:nvtinputs+1])
    #         # print(stdLUT[0:nvtinputs+1])
    #         print(psza, pvza, praa, AOD, o3col, HSurf, RsPCs[0:3])
    #         print(VTinputs)
    #         print(meanLUT[varinwb])
    #         VTinputs=(VTinputs-meanLUT[varinwb])/stdLUT[varinwb]
    #         ANNinputs=np.stack([np.append(VTinputs.flatten(),Radpcs[pcinputs].flatten())]*1,axis=0)
    #         # print(ANNinputs)
    #         ANNoutputs=ann.predict(ANNinputs)
    #         # print(ANNoutputs)
    #         ANNoutputs=ANNoutputs.flatten()*stdLUT[outputinds]+meanLUT[outputinds]
    #         # print(ANNoutputs)
    #         if OneSpec==False:
    #             outNO2C[iscan,irow]=ANNoutputs[0]*0.1
    #             outNO2H[iscan,irow]=ANNoutputs[1]
    #             outAOD[iscan,irow] = AOD
    #
    #         print(plat,plon,AOD,ANNoutputs,SNO2Col[iscan,irow]*Molm2DU)
    #
    #
    #
    #
    #         FoundOne=True





    outdata = dso.createVariable('band3_shift', np.float32, ('y', 'x'))
    outdata.units = 'nm'
    outdata[:] = outws3

    outdata = dso.createVariable('band4_shift', np.float32, ('y', 'x'))
    outdata.units = 'nm'
    outdata[:] = outws4

    outdata = dso.createVariable('band3_RingCoeff', np.float32, ('y', 'x'))
    outdata.units = 'unitless'
    outdata[:] = outringc3

    outdata = dso.createVariable('band4_RingCoeff', np.float32, ('y', 'x'))
    outdata.units = 'unitless'
    outdata[:] = outringc4

    outdata = dso.createVariable('band3_stretch', np.float32, ('y', 'x'))
    outdata.units = 'unitless'
    outdata[:] = outwg3

    outdata = dso.createVariable('band4_stretch', np.float32, ('y', 'x'))
    outdata.units = 'unitless'
    outdata[:] = outwg4

    outdata = dso.createVariable('irr_band3_shift', np.float32, ('x'))
    outdata.units = 'nm'
    outdata[:] = irws3[rowi]

    outdata = dso.createVariable('irr_band4_shift', np.float32, ('x'))
    outdata.units = 'nm'
    outdata[:] = irws4[rowi]

    outdata = dso.createVariable('irr_band3_stretch', np.float32, ('x'))
    outdata.units = 'unitless'
    outdata[:] = irwg3[rowi]

    outdata = dso.createVariable('irr_band4_stretch', np.float32, ('x'))
    outdata.units = 'unitless'
    outdata[:] = irwg4[rowi]

    dso.close()


