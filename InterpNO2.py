import numpy as np
from netCDF4 import Dataset

import glob
import argparse
import os
from TROPOMI import InterpRow


Lowthres=0.4
retrows=np.array([0,50,100,150,200,250,300,350,400,449]).astype(int)
Molm2DU=6.02214e23/2.69e20

parser=argparse.ArgumentParser()
parser.add_argument('--Region')
parser.add_argument('--buffer',type=float)
buffer=parser.parse_args().buffer

medbf=Lowthres-buffer/2.
Region=parser.parse_args().Region

DataDir='/Users/chili/PythonCodes/ANNNO2/1Day/'
#ClimNO2dir='/Volumes/users/chili/TROPOMI/OMNO2D/'
dir=DataDir+'ANNs/'+Region+'/'
outdir=DataDir+'ANNs/'+Region+'/Interp/'
TROPDir=DataDir+'NO2/'

if not os.path.exists(outdir):
    os.makedirs(outdir)

lowbuffer=buffer
files=glob.glob(dir+'*.nc')
Vars=['NO2_TOTCol_clear_Low','NO2_TOTCol_clear_High','NO2_TROPHeight_clear_High','NO2_TOTCol_cloudy_Low','NO2_TOTCol_cloudy_High']  #variable to do the interpolation

Units=['DU','DU','DU','km','DU','DU','DU']

CFVar='Cloud_Fraction'
AODVar='AOD_nofill'
CPVar='Cloud_Pressure'
#ToTColVar='NO2_std_TOTCol'
#TO3Var='O3TCol'
stdVars=['latitude','longitude','NO2_std_TROPCol','NO2_std_QA','SurfAlbedo','Cloud_Pressure','Cloud_Albedo','SolarZenithAngle','SatZenithAngle','SurfAltitude']  #,'Cloud_Fraction'


#[ClimTROPNO2,ClimSTRATNO2,ClimLat,ClimLon]=ReadDailyOMNO2(ClimNO2dir,year,month)
startind=True
for infile in files:

    outfile = outdir+(infile.split('/'))[-1]

    Basename = infile.split('/')[-1]
    Timeflag = Basename.split('_')[3]

    try:
        intpdata = InterpRow(infile, Vars, retrows)
    except:
        continue





    nscan, nrow = intpdata[0].shape

    dso = Dataset(outfile, mode='w', format='NETCDF4')
    dso.createDimension('x', nrow)
    dso.createDimension('y', nscan)

    ds = Dataset(infile, 'r')

    # NO2LC = ds['NO2_LC'][:]
    band3RingC = ds['band3_RingCoeff'][:]
    band4RingC = ds['band4_RingCoeff'][:]
    NO2gCol = ds['NO2_std_GhostCol'][:]
    minscan = np.int(np.min(ds['scans']))
    minrow = np.int(np.min(ds['rows']))
    for ioutvar in np.arange(len(stdVars)):
        NO2std = ds[stdVars[ioutvar]][:]
        if stdVars[ioutvar] == CFVar:
            CFdata = NO2std

            # NO2std[np.isnan(NO2std)]=0.
        outdata = dso.createVariable(stdVars[ioutvar], np.float32, ('y', 'x'))
        outdata.units = ds[stdVars[ioutvar]].units
        outdata[:] = NO2std

        # if stdVars[ioutvar] == TO3Var:
        #     TO3 = NO2std

        if stdVars[ioutvar] == CPVar:
            CPdata = NO2std

        # if stdVars[ioutvar] == ToTColVar:
        #     NO2ToTCol = NO2std

        if stdVars[ioutvar] == 'latitude':
            Lats = NO2std
        if stdVars[ioutvar] == 'longitude':
            Lons = NO2std

    ds.close()


    # read radiance and geometry fraction from NO2 file
    stdfile = glob.glob(TROPDir + 'S5P_OFFL_L2__NO2____' + Timeflag + '*nc')[0]
    ds = Dataset(stdfile, 'r')

    CFg = ds['PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/cloud_fraction_crb_nitrogendioxide_window'][0, :, :]

    CFg = (CFg[minscan:(minscan + nscan), :])[:, minrow:(minrow + nrow)]

    #update 2021-10-09: Change the total column to
    NO2ToTCol=ds['PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/nitrogendioxide_summed_total_column'][0, :, :]
    NO2ToTCol = (NO2ToTCol[minscan:minscan + nscan, :])[:, minrow:minrow + nrow]*Molm2DU

    CFdata = ds['PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/cloud_radiance_fraction_nitrogendioxide_window'][0, :, :]
    CFdata = (CFdata[minscan:minscan + nscan, :])[:, minrow:minrow + nrow]

    ds.close()

    CFdata[(CFdata < 0.) | (np.isnan(CFdata))] = 0.
    CFdata[CFdata > 9999.] = 0.
    CFdata[CFdata > 1] = 1.


    outdata = dso.createVariable('Cloud_Fraction_Geo', np.float32, ('y', 'x'))
    outdata.units = 'unitless'
    outdata[:] = CFg

    outdata = dso.createVariable('Cloud_Fraction_Rad', np.float32, ('y', 'x'))
    outdata.units = 'unitless'
    outdata[:] = CFdata

    outdata = dso.createVariable('NO2_std_TOTCol', np.float32, ('y', 'x'))
    outdata.units = 'DU'
    outdata[:] = NO2ToTCol

    outdata = dso.createVariable('NO2_std_VisCol', np.float32, ('y', 'x'))
    outdata.units = 'DU'
    outdata[:] = NO2ToTCol - CFg * NO2gCol

    for ioutvar in np.arange(len(Vars)):
        # if ioutvar<minhighind:
        #     (intpdata[ioutvar])[intpdata[ioutvar]>=Lowthres]=np.nan   #For Low regime, only accept if NO2<0.7 DU
        # if ioutvar==minhighind:
        #     NANind=((np.isnan(intpdata[ioutvar]))|(intpdata[ioutvar]<Lowthres)|(intpdata[ioutvar-1]<Lowthres)|(np.isnan(AOD))).nonzero() #|(intpdata[ioutvar+1]<Lowthres)|(intpdata[ioutvar-1]<Lowthres)  |(intpdata[0]<(Lowthres))  #|(intpdata[ioutvar+1]<intpdata[ioutvar])
        # if ioutvar>=minhighind:
        #     (intpdata[ioutvar])[NANind]=np.nan
        # if Vars[ioutvar]=='NO2_TROPHeight_clear_High':
        #     (intpdata[ioutvar])[np.isnan(AOD)]=np.nan
        #(intpdata[ioutvar])[(band3RingC>0.1)|(band4RingC>0.1)]=np.nan
        #(intpdata[ioutvar])[(np.isnan(TO3))|(TO3>9999)] = np.nan
        if Vars[ioutvar]=='NO2_TROPHeight_clear_High':
            lowdata = dso['NO2_TOTCol_clear_Low'][:]
            highdata = dso['NO2_TOTCol_clear_High'][:]
            (intpdata[ioutvar])[(lowdata<(Lowthres-lowbuffer))|(highdata<Lowthres)|(CFg>=0.2)]=np.nan

        if Vars[ioutvar]=='NO2_TROPHeight_cloudy_High':
            lowdata = dso['NO2_TOTCol_cloudy_Low'][:]
            highdata = dso['NO2_TOTCol_cloudy_High'][:]
            (intpdata[ioutvar])[(lowdata<(Lowthres-lowbuffer))|(highdata<Lowthres)|(CFg>=0.2)]=np.nan

        # (intpdata[ioutvar])[:,0:50]=np.nan
        # (intpdata[ioutvar])[:, 400:] = np.nan

        outdata = dso.createVariable(Vars[ioutvar], np.float32, ('y', 'x'))
        outdata.units = Units[ioutvar]
        outdata[:] =intpdata[ioutvar]



    #Merge the column retrievals under low and high conditions
    for pfxVar in ['NO2_TOTCol_clear','NO2_TOTCol_cloudy']:

        lowdata=dso[pfxVar+'_Low'][:]
        highdata = dso[pfxVar + '_High'][:]

        #lowdata[lowdata>=Lowthres]=np.nan
        #highdata[highdata<Lowthres]=np.nan

        mergedata=lowdata.copy()
        mergedata[:]=np.nan

        # mergedata[:]=np.nan
        # mergedata[lowdata<(Lowthres-buffer)]=lowdata[lowdata<(Lowthres-buffer)]
        # mergedata[(highdata>=(Lowthres+buffer))&(lowdata>=(Lowthres-buffer))]=highdata[(highdata>=(Lowthres+buffer))&(lowdata>=(Lowthres-buffer))]
        # medinds=((np.isnan(mergedata))&(lowdata<(Lowthres+buffer))).nonzero()
        # mergedata[medinds] = (lowdata[medinds]+highdata[medinds])/2.
        #mergedata[medinds]=(lowdata[medinds]*np.absolute(lowdata[medinds]-Lowthres)+highdata[medinds]*np.absolute(highdata[medinds]-Lowthres))/(np.absolute(lowdata[medinds]-Lowthres)+np.absolute(highdata[medinds]-Lowthres))

        mergedata[(lowdata<(Lowthres))&(highdata<(Lowthres))]=lowdata[(lowdata<(Lowthres))&(highdata<(Lowthres))]
        mergedata[(lowdata < (Lowthres - buffer)) & (highdata >= (Lowthres))] = lowdata[
            (lowdata < (Lowthres - buffer)) & (highdata >= (Lowthres))]

        # mergedata[(lowdata >= (Lowthres-buffer)) & (highdata >= (Lowthres + buffer))] = highdata[
        #     (lowdata >= (Lowthres-buffer)) & (highdata >= (Lowthres + buffer))]
        mergedata[(lowdata >= (Lowthres)) & (highdata >= (Lowthres))] = highdata[(lowdata >= (Lowthres)) & (highdata >= (Lowthres))] #(highdata >= (Lowthres - buffer)) &

        highwgt=(lowdata-(Lowthres-buffer))/buffer
        lowwgt=1.-highwgt
        mergedata[(lowdata >= (Lowthres-buffer))&(lowdata < Lowthres) & (highdata >= (Lowthres))]=(lowdata*lowwgt+highdata*highwgt)[(lowdata >= (Lowthres-buffer))&(lowdata < Lowthres) & (highdata >= (Lowthres))]


        # mergedata[(lowdata >= (Lowthres - buffer))&(lowdata<(Lowthres)) & (highdata >= (Lowthres)) & (highdata < (Lowthres + buffer))]=(0.5*(lowdata+highdata))[(lowdata >= (Lowthres - buffer))&(lowdata<(Lowthres)) & (highdata >= (Lowthres)) & (highdata < (Lowthres + buffer))]

        # mergedata[lowdata>=Lowthres]=np.nan
        # mergedata[((np.isnan(mergedata)) | (lowdata >= Lowthres)) & (highdata >= Lowthres)] = highdata[((np.isnan(mergedata)) | (lowdata >= Lowthres)) & (highdata >= Lowthres)]
        #
        # mergedata[((np.isnan(mergedata)) | ((lowdata >= Lowthres-buffer)&(lowdata < Lowthres))) & (highdata >= Lowthres)] = ((lowdata*np.absolute(lowdata-Lowthres)+highdata*np.absolute(highdata-Lowthres))/(np.absolute(lowdata-Lowthres)+np.absolute(highdata-Lowthres)))[((np.isnan(mergedata)) | ((lowdata >= Lowthres-buffer)&(lowdata < Lowthres))) & (highdata >= Lowthres)]

        #mergedata[((np.isnan(mergedata))|(lowdata>=(Lowthres-lowbuffer)))&(highdata>=Lowthres)]=highdata[((np.isnan(mergedata))|(lowdata>=(Lowthres-lowbuffer)))&(highdata>=Lowthres)]  #((np.isnan(mergedata))|(lowdata>(Lowthres-0.1)))&

        outdata = dso.createVariable(pfxVar, np.float32, ('y', 'x'))
        outdata.units = 'DU'
        outdata[:] = mergedata


    #merge clear and cloudy retrievals of column

    clrdata=dso['NO2_TOTCol_clear'][:]
    clddata = dso['NO2_TOTCol_cloudy'][:]
    clddata[clddata<=0.]=0.0001
    clrdata[clrdata<=0.] = 0.0001

    totdata=1./((1.-CFdata)/clrdata+CFdata/clddata)
    totdata[CFdata>=1.]=clddata[CFdata>=1.]
    totdata[CFdata<=0.]=clrdata[CFdata<=0.]
    totdata[(np.isnan(clrdata))|(np.isnan(clddata))]=np.nan

    outdata = dso.createVariable('NO2_TOTCol', np.float32, ('y', 'x'))
    outdata.units = 'DU'
    outdata[:] = totdata

    clrdata = dso['NO2_TOTCol_clear'][:]
    clddata = dso['NO2_TOTCol_cloudy'][:]+NO2gCol
    clddata[clddata <= 0.] = 0.0001
    clrdata[clrdata <= 0.] = 0.0001

    totdata = 1. / ((1. - CFdata) / clrdata + CFdata / clddata)
    totdata[CFdata >= 1.] = clddata[CFdata >= 1.]
    totdata[CFdata <= 0.] = clrdata[CFdata <= 0.]
    totdata[(np.isnan(clrdata)) | (np.isnan(clddata))] = np.nan

    outdata = dso.createVariable('NO2_TOTCol_Surf', np.float32, ('y', 'x'))
    outdata.units = 'DU'
    outdata[:] = totdata

    # clrdata = dso['NO2_TOTCol_clear_Strat'][:]
    # clddata = dso['NO2_TOTCol_cloudy_Strat'][:]
    #
    # clddata[clddata < 0.] = 0.0001
    # clrdata[clrdata < 0.] = 0.0001
    #
    # stratdata = 1. / ((1. - CFdata) / clrdata + CFdata / clddata)
    # stratdata[(stratdata/totdata<0.7)|(totdata>0.2)]=np.nan
    # stratdata[(np.isnan(clrdata)) | (np.isnan(clddata))] = np.nan
    # outdata = dso.createVariable('NO2_StratCol', np.float32, ('y', 'x'))
    # outdata.units = 'DU'
    # outdata[:] = stratdata

    # clrdata = dso['NO2_TOTCol_clear_Trop'][:]
    # clddata = dso['NO2_TOTCol_cloudy_Trop'][:]
    #
    # clddata[clddata < 0.] = 0.0001
    # clrdata[clrdata < 0.] = 0.0001
    #
    # tropdata = 1. / ((1. - CFdata) / clrdata + CFdata / clddata)
    # tropdata[(tropdata / totdata < 0.7) | (totdata < 0.4)] = np.nan
    # outdata = dso.createVariable('NO2_TropCol_ini', np.float32, ('y', 'x'))
    # outdata.units = 'DU'
    # outdata[:] = tropdata


    # try:
    #     intpstratdata=fill(stratdata,interp=True)
    # except:
    #     continue
    #
    # #intpstratdata=median_filter(intpstratdata,size=10)
    # #intpstratdata=interlin2d(Lons[np.isnan(stratdata)==False].flatten(),Lats[np.isnan(stratdata)==False].flatten(),stratdata[np.isnan(stratdata)==False].flatten(),fsize=(Lons.flatten(),Lats.flatten()))
    # outdata = dso.createVariable('NO2_StratCol_int', np.float32, ('y', 'x'))
    # outdata.units = 'DU'
    # outdata[:] = intpstratdata
    #
    # outdata = dso.createVariable('NO2_TropCol', np.float32, ('y', 'x'))
    # outdata.units = 'DU'
    # outdata[:] = totdata-intpstratdata



    #stratospheric separation,data1\2\3 corresponding to the initial (after subtracting high sources), interpolated, and smoothed one
    #[stradata1,stradata2, stradata3]=StratIntp(Lats,Lons,totdata,CPdata,CFdata,ClimTROPNO2,ClimSTRATNO2,ClimLat,ClimLon)#,outfile=outdir+'StratExample.nc')



    # MergeNO2 = intpdata[TOTinds[1]]
    # MergeNO2[np.isnan(intpdata[TOTinds[1]])] = (intpdata[TOTinds[0]])[np.isnan(intpdata[TOTinds[1]])]
    # outdata = dso.createVariable('NO2_TOTCol_Merge', np.float32, ('y', 'x'))
    # outdata.units = 'DU'
    # outdata[:] = MergeNO2

    dso.close()

# fig, axs = plt.subplots(1, 2, )
# maxy = 1.8*np.max([np.nanmax(lowclearNO2),np.nanmax(lowcloudyNO2)])
# axs[0].set_xlim(0.1, maxy)
# axs[1].set_xlim(0.1, maxy)
# axs[0].set_ylim(0.1, maxy)
# axs[1].set_ylim(0.1, maxy)
#
# valinds=np.array(((np.isnan(lowclearNO2)==False)&(np.isnan(highclearNO2)==False)).nonzero()).flatten()
# hist1=axs[0].hist2d(lowclearNO2[valinds], highclearNO2[valinds],range=[[0.1,maxy],[0.1,maxy]],bins=500,cmap='magma_r',norm=LogNorm())
# cbar1=plt.colorbar(hist1[3],ax=axs[0],cax=fig.add_axes([0.481,0.3,0.01,0.4]))
# cbar1.ax.tick_params(which='both',direction='in',pad=0.6,width=0.4,length=1)
# cbar1.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
# cbar1.ax.yaxis.set_minor_formatter(ticker.NullFormatter())
#
# valinds=np.array(((np.isnan(lowcloudyNO2)==False)&(np.isnan(highcloudyNO2)==False)).nonzero()).flatten()
# hist2=axs[1].hist2d(lowcloudyNO2[valinds], highcloudyNO2[valinds],range=[[0.1,maxy],[0.1,maxy]],bins=500,cmap='magma_r',norm=LogNorm())
# cbar2=plt.colorbar(hist2[3],ax=axs[1],cax=fig.add_axes([0.903,0.3,0.01,0.4]))
# cbar2.ax.tick_params(which='both',direction='in',pad=0.6,width=0.4,length=1)
# cbar2.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
# cbar2.ax.yaxis.set_minor_formatter(ticker.NullFormatter())
#
# #axs[0].plot([0.4-buffer,0.4-buffer],[0,maxy],color='black',linestyle=':')
# #axs[1].plot([0.4-buffer,0.4-buffer], [0, maxy], color='black', linestyle=':')
#
# axs[0].plot([0.4-buffer,0.4], [0.4, 0.4], color='black', linestyle=':')
# #axs[0].plot([0.4-buffer,0.4], [0.4, 0.4],  color='black', linestyle=':')
# axs[0].plot([0.4-buffer,0.4-buffer], [0.4, maxy], color='black', linestyle=':')
# axs[0].plot([0.4, 0.4],[0., 0.4],  color='black', linestyle=':')
# #axs[0].plot([0.4,0.4], [0., maxy], color='black', linestyle=':')
# #axs[1].plot([0.4, 0.4],[0., maxy],  color='black', linestyle=':')
#
# axs[1].set_yticks([])
# axs[0].set_ylabel('Retrieval from high NO'+r'$_2$'+' NN (DU)')
# axs[0].set_xlabel('Retrieval from low NO'+r'$_2$'+' NN (DU)')
# axs[1].set_xlabel('Retrieval from low NO'+r'$_2$'+' NN (DU)')
#
# axs[0].set_title('Clear (CF<0.2)')
# axs[1].set_title('Overcast (CF>0.8)')
#
# axs[0].set_aspect(1)
# axs[1].set_aspect(1)
# plt.savefig(outdir + Region+'.lowhigh.png', dpi=600)






