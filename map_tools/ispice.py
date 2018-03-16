# This file is from the polspice package and included in map_tools for
# convenience.  It includes bin_llcl() which used to be defined
# in a separate source file.

"""
  http://www2.iap.fr/users/hivon/software/PolSpice

  ispice defines tools to run Spice from Python,
  either in the Planck HFI DMC (aka piolib, objects managed by database)
  or using FITS files

Includes:
  Main routine:
        ispice()
  Support routines:
        flatten()
        is_fits()
        look_for_spice()
        thinc_exists()
        tools()
        bin_llcl()
  """


def flatten(x):
    """ flatten():  turns nested list or tuple into single list """
    result = []
    if isinstance(x, str):
        result = x
    else:
        for el in x:
            if hasattr(el, "__iter__") and not isinstance(el, str):
                result.extend(flatten(el))
            else:
                result.append(el)
    return result

def is_fits(inlist):
    """ test presence of FITS file in a list """
    import os
    flatlist = flatten(inlist)
    if isinstance(flatlist, str): flatlist = [ flatlist ]
    isFits = False
    for f in flatlist:
        if isinstance(f, str):
            if os.path.exists(f):
                isFits = os.path.isfile(f)
    return isFits

def thinc_exists():
    """ thinc_exists():  test existence of thinc module (DMC specific) """
    import imp
    try:
        imp.find_module('thinc')
        imp.find_module('piolib')
        myDMC = True
    except ImportError:
        myDMC = False
    return myDMC

def look_for_spice():
    """ look for spice code in neighbouring directories """
    import os
    from os.path import join

    lookfor = "spice"
    bdir = os.path.dirname(__file__)
    locs = (join(bdir,'.'), join(bdir,'..'), join(bdir,'../..'))
    path = ''
    for loc in locs:
        for root, dirs, files in os.walk(loc):
            if lookfor in files:
                path = join(root, lookfor)
                break
        if (path != ''): break
    return path

#----------------------------------------------------------------

class tools(object):
    """ tools() defines context dependent routines """
    def __init__(self, object):
        self.type = object
        in_thinc = thinc_exists()
        in_fits  = is_fits(object)
        if (not in_thinc and not in_fits):
            import sys
            print('Input files are not in FITS, and DMC not available')
            sys.exit('Aborting')

        self.inDMC = in_thinc and not in_fits
        self.command = ''
    # -------------------------------------------
    def myNewJob(self, code, label="", cast=False):
        if (self.inDMC):
            from thinc import NewJob
            return NewJob(code, label=label, cast=cast)
        else:
            if code == 'spice':
                mycom = look_for_spice()
            else:
                mycom = code
            self.command += mycom+' '
            return 0

    def mySet(self, jobid, name, value):
        if (self.inDMC):
            from thinc import Set
            Set(jobid, name, value)
        else:
            #print(str(name)+' = '+str(value))
            self.command += '-%s %s '%(str(name),str(value))

    def mySubmit(self, jobid, exportEnviron=None):
        if (self.inDMC):
            from thinc import Submit
            Submit(jobid, exportEnviron=exportEnviron)
        else:
            import os
            print('Submitting job %s'%(str(jobid)))
            print(self.command)
            os.system(self.command)
    # -------------------------------------------
    def inDMC(self):
        return self.inDMC

    def parse_mapfile(self, jobid, mapfile, case):
        import numpy as np
        """ parse_mapfile(jobid, mapfile, case)
        """
        if (self.inDMC):
            keywords=['mapfile','mapQfile','mapUfile']
        else:
            keywords=['mapfile']
        if (case==2): keywords = [        k+'2' for k in keywords]
        if (case==3): keywords = ['extra'+k     for k in keywords]
        if (case==4): keywords = ['extra'+k+'2' for k in keywords]

        ##print(case, mapfile,type(mapfile))
        if isinstance(mapfile, (list,tuple)): # list
            if (isinstance(mapfile[0], (list,tuple))  and (case==1 or case==2) ): # list of list
                ##print('==>List of list')
                if (self.inDMC):
                    if (case == 1):
                        kw = ["listmapweights1_", ["listmapfiles1_","listmapfilesQ1_","listmapfilesU1_"]]
                    if (case == 2):
                        kw = ["listmapweights2_", ["listmapfiles2_","listmapfilesQ2_","listmapfilesU2_"]]
                else:
                    if (case == 1):
                        kw = ["listmapweights1_", ["listmapfiles1_"]]
                    if (case == 2):
                        kw = ["listmapweights2_", ["listmapfiles2_"]]
                for i in range(len(mapfile)):
                    si = str(i+1)
                    if isinstance(mapfile[i][0], (type(1), type(1.), np.float32, np.float64)):
                        self.mySet(jobid, "%s%s"%(kw[0], si), mapfile[i][0])
                    else:
                        print('Warning: number was expected, got '+str(mapfile[i][0]))
                    if isinstance(mapfile[i][1], (list, tuple)):
                        nl = len(mapfile[i][1])
                        if (nl != 1 and nl != 3):
                            print('Warning: expected a 1 or 3-element list, found '+str(nl))
                            print(mapfile[i][1])
                        for j in range(nl):
                            self.mySet(jobid, "%s%s"%(kw[1][j], si), mapfile[i][1][j])
                    else:
                        self.mySet(jobid, "%s%s"%(kw[1][0], si), mapfile[i][1])
            else: # simple list
                ##print('==>List')
                if (len(mapfile[0]) > 0): # non-empty string in list
                    self.mySet(jobid, keywords[0], mapfile[0])
                    if (len(mapfile) == 3):# 3 element list
                        self.mySet(jobid, keywords[1], mapfile[1])
                        self.mySet(jobid, keywords[2], mapfile[2])
        else: #  not list -> string
            ##print('==>String')
            if (len(mapfile) > 0): # non-empty string
                self.mySet(jobid, keywords[0],  mapfile)


# ==============================================================================

def ispice(mapin1, clout, nlmax=-1,
           apodizetype=0,
           apodizesigma="NO",
           beam1="NO", beam2="NO",
           beam_file1="", beam_file2="",
           binpath="",
           covfileout="",
           corfile="",
           decouple="NO",
           extramapfile1="", extramapfile2="",
           fits_out="YES",
           mapfile2="",
           maskfile1="", maskfile2="", maskfilep1="", maskfilep2="",
           pixelfile="YES", pixelfile2="YES", polarization="NO",
           subav="NO",
           subdipole="NO",
           symmetric_cl="NO",
           tenormfilein="",tenormfileout="",
           tf_file="",
           thetamax="NO",
           tolerance="NO",
           weightfile1="", weightfilep1="",
           weightfile2="", weightfilep2="",
           weightpower1=1.0, weightpower2=1.0,
           weightpowerp1=1.0, weightpowerp2=1.0,
           windowfilein="",windowfileout="",
           label="spice", submit=None):
    """ ispice(mapin1, clout, nlmax=-1,
           apodizetype=0,
           apodizesigma="NO",
           beam1="NO", beam2="NO",
           beam_file1="", beam_file2="",
           binpath="",
           covfileout="",
           corfile="",
           decouple="NO",
           extramapfile1="", extramapfile2="",
           fits_out="YES",
           mapfile2="",
           maskfile1="", maskfile2="", maskfilep1="", maskfilep2="",
           pixelfile="YES", pixelfile2="YES", polarization="NO",
           subav="NO",
           subdipole="NO",
           symmetric_cl="NO",
           tenormfilein="", tenormfileout="",
           tf_file="",
           thetamax="NO" ,
           tolerance="NO" ,
           weightfile1="", weightfilep1="",
           weightfile2="", weightfilep2="",
           weightpower1=1.0, weightpower2=1.0, weightpowerp1=1.0, weightpowerp2=1.0,
           windowfilein="",windowfileout="",
           label="spice", submit=None):

           Python interface to F90 spice code

           Required:
           mapin1:    input I map, or list of [I,Q,U] maps,  DMC objects of type MAP
                      input I or IQU map,                    FITS files
                     or list of lists for on-the-fly weighted linear combination (LC) of maps:
                     eg:  [ [w1, I1 ],       [w2,  I2 ],       [w3,  I3 ],      ... ]
                     or   [ [w1,[I1]],       [w2, [I2]],       [w3, [I3]],      ... ]
                     or   [ [w1,[IQU1]],     [w2, [IQU2]],     [w3, [IQU3]]     ... ] (FITS only)
                     or   [ [w1,[I1,Q1,U1]], [w2, [I2,Q2,U2]], [w3, [I3,Q3,U3]] ... ] (DMC only)
                     so that mapin1_I = w1*I1 + w2*I2 + w3*I3 + ...
                       (and  mapin1_Q = w1*Q1 + w2*Q2 + w3*Q3 + ...)
                     where w* is a scalar number and I*,Q*,U* are MAPtype objects

           clout:    output C(l) (either auto or cross),   object  of type CL

           Optional:
           ...
           beam_file*: input B(l) file(s),                 FITS files or objects of type CL
           covfileout: output C(l)C(l) covariance matrix,  FITS file  or object  of type TAB3D
           mapfile2: map(s) or LC of maps for cross-spectrum
              see mapin1 for format
           extramapfile1: I or [I,Q,U] map to be added to mapin1
              will be ignored by Spice if mapin1 is a LC
           extramapfile2: I or [I,Q,U] map to be added to mapfile2
              will be ignored by Spice if mapfile2 is a LC
           fits_out: output files are in FITS instead of plain ASCII
           ...

           """

    mytools = tools(mapin1)

    command="spice"
    if (not mytools.inDMC and len(binpath)>0):
        command=binpath
    myJob = mytools.myNewJob(command, label = label, cast = True)

    mytools.mySet(myJob, "apodizesigma", apodizesigma)
    mytools.mySet(myJob, "apodizetype",  apodizetype)

    mytools.mySet(myJob, "beam",         beam1)
    if (len(beam_file1)>0):
        mytools.mySet(myJob, "beam_file",    beam_file1)
    mytools.mySet(myJob, "beam2",        beam2)
    if (len(beam_file2)>0):
        mytools.mySet(myJob, "beam_file2",   beam_file2)

    mytools.mySet(myJob, "clfile",       clout)
    if (len(covfileout)>0):
        mytools.mySet(myJob, "covfileout",   covfileout)
    if (len(corfile)>0):
        mytools.mySet(myJob, "corfile",  corfile)
    mytools.mySet(myJob, "decouple",     decouple)

    mytools.parse_mapfile(myJob, mapin1,        1)
    mytools.parse_mapfile(myJob, mapfile2,      2)
    mytools.parse_mapfile(myJob, extramapfile1, 3)
    mytools.parse_mapfile(myJob, extramapfile2, 4)

    if (not mytools.inDMC):
        mytools.mySet(myJob, "fits_out", fits_out)

    if (len(maskfile1)>0):
        mytools.mySet(myJob, "maskfile",      maskfile1)
    if (len(maskfilep1)>0):
        mytools.mySet(myJob, "maskfilep",     maskfilep1)
    if (len(maskfile2)>0):
        mytools.mySet(myJob, "maskfile2",     maskfile2)
    if (len(maskfilep2)>0):
        mytools.mySet(myJob, "maskfilep2",    maskfilep2)

    if (len(weightfile1)>0):
        mytools.mySet(myJob, "weightfile",      weightfile1)
    if (len(weightfilep1)>0):
        mytools.mySet(myJob, "weightfilep",     weightfilep1)
    if (len(weightfile2)>0):
        mytools.mySet(myJob, "weightfile2",     weightfile2)
    if (len(weightfilep2)>0):
        mytools.mySet(myJob, "weightfilep2",    weightfilep2)

    mytools.mySet(myJob, "nlmax",        nlmax)
    mytools.mySet(myJob, "normfac",      "1.00000")
    mytools.mySet(myJob, "npairsthreshold", "0.00000")
    mytools.mySet(myJob, "overwrite",       "YES")
    mytools.mySet(myJob, "polarization",    polarization)
    mytools.mySet(myJob, "pixelfile",       pixelfile)
    mytools.mySet(myJob, "pixelfile2",      pixelfile2)
    mytools.mySet(myJob, "subav",           subav)
    mytools.mySet(myJob, "subdipole",       subdipole)
    mytools.mySet(myJob, "symmetric_cl",    symmetric_cl)
    if (len(tenormfilein)>0):
        mytools.mySet(myJob, "tenormfilein", tenormfilein)
    if (len(tenormfileout)>0):
        mytools.mySet(myJob, "tenormfileout",tenormfileout)
    if (len(tf_file)>0):
        mytools.mySet(myJob, "tf_file",     tf_file)
    mytools.mySet(myJob, "thetamax",        thetamax)
    mytools.mySet(myJob, "tolerance",       tolerance)
    mytools.mySet(myJob, "verbosity",       "2")
    mytools.mySet(myJob, "weightpower",     weightpower1)
    mytools.mySet(myJob, "weightpower2",    weightpower2)
    mytools.mySet(myJob, "weightpowerp",    weightpowerp1)
    mytools.mySet(myJob, "weightpowerp2",   weightpowerp2)
    if (len(windowfilein)>0):
        mytools.mySet(myJob, "windowfilein",    windowfilein)
    if (len(windowfileout)>0):
        mytools.mySet(myJob, "windowfileout",   windowfileout)

    if (mytools.inDMC):
        mytools.mySet(myJob, "pbs_extraOption","-l place=scatter:excl -l select=1:ncpus=8")


    mysubmit=submit
    if (mysubmit==None):
        if (mytools.inDMC):
            mysubmit=False # default: do not submit in DMC
        else:
            mysubmit=True # default: do submit in FITS

    if mysubmit:
        mytools.mySubmit(myJob, exportEnviron={"OMP_NUM_THREADS":"8"})
    else:
        if (not mytools.inDMC):
            print(mytools.command)
            print()
            print('The command above was NOT RUN.')
            print('Select submit=True in ispice to run it, or copy and paste it in a terminal')
    return myJob


def bin_llcl (llcl_in, ubin, flatten=False, uniform=False):
    """ x, y, dx, dy = bin_llcl( llcl_in, bin, flatten=False, uniform=False)
    
    turns llcl_in (= continuous l*(l+1)*cl/2Pi) into a binned version with
       a constant or variable binwidth 'bin'


    INPUTS
      llcl_in : input l*(l+1)*Cl/2Pi, 1D vector, defined for each l from l=0
      bin : can be either a scalar = dl
        or a vector defining the bins boundaries : 
        [low0, low1, low2, ...,low(n-2), low(n-1)+1]
    
    OUTPUTS
        x:  center of bins
        y:  binned l*(l+1)*Cl/2Pi
        dx: width of each bin
        dy: returns on output the rms of C(l) for a full sky observation
           = C(l) * sqrt( 2/ 2l+1 / dl)
    
    
    KEYWORDS
      flatten: if set, the input C(l) is multiplied internally by l*(l+1)/2Pi before being
        binned. By default, the input C(l) is binned as is.
    
      uniform: if set, each l is given the same weight in the bin.
        By default, a weighting propto (2*l+1) (inverse cosmic
        variance) is applied to each l.
        In any cases, the output x is the same

    HISTORY
      2016-11-23: adapted from Healpix/IDL    bin_llcl.pro
      
    """

    bin = np.copy(ubin)
    nb  = np.size(bin)
    lmax_in = np.size(llcl_in)-1
    if (nb > 1):
        k = np.where(bin <= (lmax_in+1))[0]
        nk = np.size(k)
        if (nk == (nb-1)):
            bin = np.minimum( bin , lmax_in+1 ) # shorten last bin
        if (nk < (nb-1)): # shorten last valid bins, and drop the ones beyond lmax
            bin = np.concatenate((bin[k], [lmax_in+1]))
            nb = nk + 1
    

    if nb == 1:
        # regular binning
        nbins = int(lmax_in // np.int(bin))
        lmax  = nbins * bin  -1
        l     = np.arange(lmax+1, dtype=np.float)
        w     = 2*l + 1
        if (uniform):
            w = np.ones(lmax+1, dtype=np.float)
        y = np.copy(llcl_in[0:lmax+1])
        if (flatten):
            y *= l*(l+1.)/(2*np.pi)
        w1 = np.reshape(w,    (nbins, bin))
        y1 = np.reshape(y*w,  (nbins, bin))
        l1 = np.reshape(l,    (nbins, bin))
        n1 = np.ones(         (nbins, bin))
        
        llcl_out = np.sum(y1,1)/np.sum(w1,1)
        l_out    = np.sum(l1,1)/np.sum(n1,1)
        dl       = bin * np.ones(nbins, dtype=np.int)
        
    else:

        # irregular binning
        lmax  = int(np.amax(bin) - 1)
        nbins = int(nb-1)
        good  = np.where(bin < lmax)[0]
        ng    = np.size(good)
        if (ng == 0):
            print('l-range of binning does not intersect that of data')
            return -1, -1, -1, -1
    
        l  = np.arange(lmax+1, dtype=np.float)
        w  = 2*l + 1
        if (uniform):
            w = np.ones(lmax+1, dtype=np.float)
        y = np.copy(llcl_in[0:lmax+1])
        if (flatten):
            y *= l*(l+1.)/(2*np.pi)
        l_out    = np.zeros(nbins, dtype=np.float)
        llcl_out = np.zeros(nbins, dtype=np.float)
        dl       = np.zeros(nbins, dtype=np.int)
        for i in range(nbins):
            l_out[i] = np.mean(l[bin[i]:bin[i+1]])
            dl[i]    = bin[i+1]-bin[i]
            llcl_out[i] = np.sum( (y*w) [bin[i]:bin[i+1]] ) \
                        / np.sum(    w  [bin[i]:bin[i+1]] )
    

    dllcl = llcl_out * np.sqrt(2/(2*l_out+1.)/dl)
    
    return l_out, llcl_out, dl, dllcl


    
