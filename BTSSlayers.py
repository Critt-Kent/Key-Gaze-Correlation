import sys
import numpy as np
import pandas as pd
import os.path
import glob

   
def writeBTSS(studies, btssDir="BTSS", path='/data/critt/tprdb/TPRDB/', p=0, verbose=1): 
    '''
    studies: list of studies
    btssDir: local BTSS directory ignore if btssDir = ''
    user: default in TPRDB
    p: p=1 flag to print in TPRDB 
    
    '''
    if(btssDir != '') :
        try:
            os.mkdir(btssDir)
        except FileExistsError:
            print(f"Directory '{btssDir}' already exists.")
    
    for study in sorted(studies):
        if(verbose) : print(f"Writing: {study}")
        
        (AUdf, KBdf, PUdf, HOFdf, POLdf, PHdf) = writeBTSSstudy(study, path=path, p=p, verbose=verbose-1)

        # write session to BTSS directory
        if (btssDir != '') :
            # output the BTSS layers in the provided BTSS directory 
            AUdf.to_csv(f"{btssDir}/{study}.au", header=True, index=None, sep='\t', mode='w')
            KBdf.to_csv(f"{btssDir}/{study}.kb", header=True, index=None, sep='\t', mode='w')
            PUdf.to_csv(f"{btssDir}/{study}.pu", header=True, index=None, sep='\t', mode='w')
            HOFdf.to_csv(f"{btssDir}/{study}.hof", header=True, index=None, sep='\t', mode='w')
            POLdf.to_csv(f"{btssDir}/{study}.pol", header=True, index=None, sep='\t', mode='w')
            PHdf.to_csv(f"{btssDir}/{study}.ph", header=True, index=None, sep='\t', mode='w')


# read one study 
def writeBTSSstudy(study, path='/data/critt/tprdb/TPRDB/', p=0, verbose=0) :

    AU = pd.DataFrame()
    KB = pd.DataFrame()
    PU = pd.DataFrame()
    HOF = pd.DataFrame() 
    POL = pd.DataFrame() 
    PH = pd.DataFrame()

    files = glob.glob(f"{path}/{study}/Tables/*au")
    n = 0
    for fn in sorted(files):                
        if(verbose): 
            print(f"{n}\t{fn}")
            n+=1
            
        H = writeBTSSsession(fn.replace(".au", ""), p=p, verbose=verbose-1)
        if(len(H) < 1) :  continue

        AU = pd.concat([AU, H['au']], ignore_index=True)
        KB = pd.concat([KB, H['kb']], ignore_index=True)
        PU = pd.concat([PU, H['pu']], ignore_index=True)
        HOF =pd.concat([HOF, H['hof']], ignore_index=True)
        POL =pd.concat([POL, H['pol']], ignore_index=True)
        PH = pd.concat([PH, H['ph']], ignore_index=True)

    return (AU, KB, PU, HOF, POL, PH)

def writeBTSSsession(fn, p=0, verbose=0) :
    
    # read AU file, add features and HOF states
    A = HOF_states_heuristic(fn + '.au') 
    if(len(A) < 1) :
        print("Empty file:", fn)
        return A

    A['Dur_L'] = A.Dur_L.astype(int)
    A['Dur_R'] = A.Dur_R.astype(int)
    A['Dur_S'] = A.Dur_S.astype(int)
    A['Dur_N'] = A.Dur_N.astype(int)

    # generate BTSS layers 
    M = makeBTSSlayers(A, verbose=verbose)

    # classification of Policies
    PolicyClasses(M['pol'])
    
    # print file in TPRDB directory if p-flag
    if(p) :
        H = ['Id', 'Study', 'Session', 'SL', 'TL', 'Task', 'Text', 'Part', 
             'Time', 'Dur', 'LogDur', 'HOF', 'Ins', 'Del', 
            'KBseq', 'KBcount', 'Dur_L', 'Dur_R', 'Dur_S', 'Dur_N', 
            'RelDur_L', 'RelDur_R', 'RelDur_S', 'RelDur_N', 'LogDur_L', 'LogDur_R', 'LogDur_S', 'LogDur_N',
             'SfixDiff', 'SfixTot', 'TfixDiff', 'TfixTot',
           'Odur', 'Hdur', 'Fdur', 'Rdur'
         ]

        P = ['Id', 'Study', 'Session', 'StudySession', 'Tstyles', 'SL', 'TL', 'Task', 'Text', 'Part', 'Time', 
            'HOF', 'Ins', 'Del', 'InEff',
             'Dur', 'Odur', 'Hdur', 'Rdur', 'Fdur',
            'RelOdur', 'RelHdur', 'RelRdur', 'RelFdur',
            'LogDur', 'LogOdur', 'LogHdur', 'LogRdur', 'LogFdur',
            'Dur_L', 'Dur_R', 'Dur_S', 'Dur_N', 
             'RelDur_L', 'RelDur_R', 'RelDur_S', 'RelDur_N', 
            'D', 'O', 'H', 'R', 'F', 'I', 'DOHRFI'
             ]

        M['au'].to_csv(fn+'.au1', header=True, index=None, sep='\t', mode='w')        
        M['kb'].to_csv(fn+'.kb1', header=True, index=None, sep='\t', mode='w')
        M['pu'].to_csv(fn+'.pu1', header=True, index=None, sep='\t', mode='w')        
        M['ph'].to_csv(fn+'.ph1', header=True, index=None, sep='\t', mode='w')

#        M['hof'].to_csv(fn+'.hof', header=True, index=None, sep='\t', mode='w')        
#        M['pol'].to_csv(fn+'.pol', header=True, index=None, sep='\t', mode='w')
        M['hof'][H].to_csv(fn+'.hof', header=True, index=None, sep='\t', mode='w')        
        M['pol'][P].to_csv(fn+'.pol', header=True, index=None, sep='\t', mode='w')
     
    return M

##########################################################################
# read BTSS from 
def readBTSS(studies, btssDir='BTSS', layers = ['au', 'kb', 'pu', 'hof', 'pol','ph'], verbose=1) :
    
    BTSS = {}

    for study in sorted(studies):
        if(verbose) : print(f"Reading: {study}")

        for l in sorted(layers):                                
            fn = f"{btssDir}/{study}.{l}"
            if(verbose > 1) : print(f"Writing: {fn}")
            
            BTSS.setdefault(l, pd.DataFrame())
            BTSS[l] = pd.concat([BTSS[l], pd.read_csv(f"{fn}", sep="\t", dtype=None)], ignore_index=True)

    return BTSS

# read Tables for a list of sessions from TPR-DB, 
def readBTSSsessions(sessions, layers = ['au1', 'kb1', 'pu1', 'hof', 'pol','ph1'], 
                     path = "/data/critt/tprdb/TPRDB/", verbose=1) :
    
    BTSS = {}
    for s in sessions:
        if(verbose) : print(f"Reading: {s}")

        for l in sorted(layers):                                
            fn = f"{path}/{s}.{l}"
            
            BTSS.setdefault(l, pd.DataFrame())
            BTSS[l] = pd.concat([BTSS[l], pd.read_csv(f"{fn}", sep="\t", dtype=None)], ignore_index=True)

    return BTSS

############################################################################
# BTSS layers
############################################################################
# total duration 
#
def makeBTSSlayers(AU, verbose = 0) :

    # initialize Tstyles if not present
    if ('Tstyles' not in AU) : AU['Tstyles'] = 0

    # aggregation functions
    def firstLabel(x): return x.iloc[0]
        
    def UNITtypeSkip(lst):
        lst = list(lst)   
        result = []
        # there should be at least one KB in a PU
        result.append(str(lst[0]))
        for i in range(1, len(lst)):
            if(lst[i] == lst[i - 1]): continue
            result.append(str(lst[i]))
        new_list = [item for item in result if item not in ['0','P','K','-']]
        if(new_list == []): new_list = ['-']
        return ''.join(new_list)

    def UNITtype(lst):
        lst = list(lst)   
        result = []
        # there should be at least one KB in a PU
        result.append(str(lst[0]))
        for i in range(1, len(lst)):
            result.append(str(lst[i]))
        new_list = [item for item in result if item not in ['0','P','K','-']]
        if(new_list == []): new_list = ['-']
        return ''.join(new_list)

    def PUtype(lst):
        lst = list(lst)   
        KBtype = ['I', 'D', 'C']
        # maximum 9 KB per type
        return ''.join([str(min(lst.count(i), 9)) for i in KBtype])
    
    def HOFtype(lst):
        lst = list(lst)
#        print("I", lst)
        H = {0:0, 1:0, 2:0}
        for PUtype in lst:
            for o, n in  enumerate(list(PUtype)):
                H[o] += int(n)
#                print("\tX", o, n, H[o])
        new_list = [str(min(H[o], 9)) for o in H]
#        print("O", new_list, '\t', ''.join(new_list))
        return ''.join(new_list)

    def GazePath(lst):
        H = {'S' : {}, 'T' : {}}
        for Fpath in list(lst):
            if(Fpath == '---'): continue
            for fix in Fpath.split('+'):
                win, wrd = fix.split(':')
                H[win].setdefault(wrd, 0)
                H[win][wrd] += 1
        return f"{len(H['S'])},{sum(H['S'].values())},{len(H['T'])},{sum(H['T'].values())}"

    def SortSet(lst):
        lst = sorted(list(set(lst)))
        return ''.join(map(str, lst))

       
    #######################################################
    # Total Duration of sessions
    SS = AU.groupby(['StudySession']).agg({'Dur': 'sum'}).reset_index()
    SS.columns = ['StudySession', 'TotalDur']
    SS['TotalLogDur'] = np.log1p(SS.TotalDur + 1)
    
    #######################################################
    # AU layer

#    AU['LogAU_nbr'] = np.log1p(AU.AU_nbr)
    AU['LogDur'] = np.log1p(AU.Dur)
    AU['LogDur_L'] = np.log1p(AU.Dur_L)
    AU['LogDur_R'] = np.log1p(AU.Dur_R)
    AU['LogDur_S'] = np.log1p(AU.Dur_S)
    AU['LogIns'] = np.log1p(AU.Ins)
    AU['LogDel'] = np.log1p(AU.Del)
    AU['LogKBI'] = np.log1p(AU.KBI)
    AU['LogPUB'] = np.log1p(AU.PUB)
    
    #######################################################
    # KB layer
    # Duration of KBs   
    
    KBdf = AU.groupby(['StudySession', 'KBnbr']).agg(
        {'Time':firstLabel, 
         'Dur':'mean', 
         'Ins': 'sum', 
         'Del': 'sum', 
         'Tstyles': firstLabel,
         'KBtype': [firstLabel, UNITtype], 
         'PUnbr': firstLabel, 
         'Type': [UNITtype, 'count']}).reset_index()
    KBdf.columns = ['StudySession', 'KBnbr',  'Time', 'Dur', 'Ins', 'Del', 'Tstyles', 'KBtype', 'KBseq', 'PUnbr', 'AUseq',  'AUcount']
    KBdf['LogDur'] = np.log1p(KBdf['Dur']) 
    KBdf['Dur'] = KBdf['Dur'].astype(int)

    KBdf['Study'] = AU.Study[0]
    KBdf['Session'] = AU.Session[0]
    KBdf['Id'] = range(1, len(KBdf) +1)
    KBdf['SL'] = AU.SL[0]
    KBdf['TL'] = AU.TL[0]
    KBdf['Task'] = AU.Task[0]
    KBdf['Text'] = AU.Text[0]
    KBdf['Part'] = AU.Part[0]

    #######################################################
    # PU layer (PU)
    # Duration of each PUs (PUdur column shows PU durations)
       
    PUdf = AU.groupby(['StudySession', 'PUnbr']).agg(
        {'Time':firstLabel, 
         'Dur': 'sum', 
         'Ins': 'sum', 
         'Del': 'sum', 
         'Tstyles': firstLabel, 
         'KBtype': [PUtype, UNITtype, 'count'],
         'HOF': SortSet,
         'HOFnbr': "max",
    #    'HOFnbr': firstLabel
        }).reset_index()

    PUdf.columns = ['StudySession', 'PUnbr', 'Time', 'Dur', 'Ins', 'Del', 'Tstyles', 'PUtype', 'KBseq',  'KBcount', 'HOFset','HOFnbr']
    
    PUdf['LogDur'] = np.log1p(PUdf['Dur']) 
    
    PUdf['Study'] = AU.Study[0]
    PUdf['Session'] = AU.Session[0]
    PUdf['Id'] = range(1, len(PUdf) +1)
    PUdf['SL'] = AU.SL[0]
    PUdf['TL'] = AU.TL[0]
    PUdf['Task'] = AU.Task[0]
    PUdf['Text'] = AU.Text[0]
    PUdf['Part'] = AU.Part[0]

    
    #######################################################
    # HOF Layer (HOF)
  
    HOFdf = AU.groupby(['StudySession', 'HOFnbr']).agg(
        {'Time':firstLabel, 
         'Dur': 'sum', 
         'Ins': 'sum', 
         'Del': 'sum', 
         'Tstyles': firstLabel, 
         'HOF': firstLabel, 
         'KBtype': [UNITtype, 'count'],
         'GazePath':GazePath, 
         'Dur_L':'sum', 'Dur_R':'sum', 'Dur_S':'sum', 'Dur_N':'sum'}).reset_index()

    # nename columns
    HOFdf.columns = ['StudySession', 'HOFnbr', 'Time', 'Dur', 'Ins', 'Del', 'Tstyles', 'HOF', 'KBseq', 'KBcount', 'GazePath', 'Dur_L', 'Dur_R', 'Dur_S', 'Dur_N']

    HOFdf['Study'] = AU.Study[0]
    HOFdf['Session'] = AU.Session[0]
    HOFdf['Id'] = range(1, len(HOFdf) +1)
    HOFdf['SL'] = AU.SL[0]
    HOFdf['TL'] = AU.TL[0]
    HOFdf['Task'] = AU.Task[0]
    HOFdf['Text'] = AU.Text[0]
    HOFdf['Part'] = AU.Part[0]
    
    HOFdf['LogDur'] = np.log1p(HOFdf['Dur']) 
    HOFdf[['SfixDiff', 'SfixTot','TfixDiff', 'TfixTot']] = HOFdf['GazePath'].str.split(',', expand=True).astype(int)
    HOFdf.drop('GazePath', axis=1, inplace = True)

    # number of different fixated words /  total number of fixations
    HOFdf['SfixRel'] = HOFdf['SfixDiff']/HOFdf['SfixTot']
    HOFdf['TfixRel'] = HOFdf['TfixDiff']/HOFdf['TfixTot']
    HOFdf['RelDur_L'] = HOFdf['Dur_L']/HOFdf['Dur']
    HOFdf['RelDur_R'] = HOFdf['Dur_R']/HOFdf['Dur']
    HOFdf['RelDur_S'] = HOFdf['Dur_S']/HOFdf['Dur']
    HOFdf['RelDur_N'] = HOFdf['Dur_N']/HOFdf['Dur']
    
    HOFdf['LogDur_L'] = np.log1p(HOFdf['Dur_L'])
    HOFdf['LogDur_R'] = np.log1p(HOFdf['Dur_R'])
    HOFdf['LogDur_S'] = np.log1p(HOFdf['Dur_S'])
    HOFdf['LogDur_N'] = np.log1p(HOFdf['Dur_N'])

    # smooth probabilities = 1 to compute logIt
#    HOFdf.loc[(HOFdf.SfixRel == 1.0), 'SfixRel'] = 0.99999
#    HOFdf.loc[(HOFdf.TfixRel == 1.0), 'TfixRel'] = 0.99999

    HOFdf['SfixLogOdds'] = np.log1p(HOFdf['SfixRel']/(1-HOFdf['SfixRel']))
    HOFdf['TfixLogOdds'] = np.log1p(HOFdf['TfixRel']/(1-HOFdf['TfixRel']))

    HOF1 = PUdf.groupby(['StudySession', 'HOFnbr']).agg({'PUtype': [HOFtype, 'count'], 'HOFset' : set}).reset_index()
    HOF1.columns = ['StudySession', 'HOFnbr', 'HOFtype', 'PUcount','HOFset']
    HOFdf = pd.merge(HOFdf, HOF1, on=['StudySession', 'HOFnbr'], how='outer')

    
    ########################################################
    # Policy Index to HOF data
    HOFdf['Id'] = 0
    
    # set policy index = 0
    policy = 0
    start = -1
    for i, r in HOFdf.iterrows():
        if(start == -1) : start = i
    
        # A  new policy starts with 'O'
        if(r.HOF == 'O') :
            # all HOF states between successive O's get identical index
            HOFdf.loc[start:i, 'Id'] = policy
            start = i
            # increment policy index
            policy += 1
    
    # last policy
    HOFdf.loc[start:, 'Id'] = policy
    
    
    ###################################
    # relative duration 
    HOFdf['Odur'] = 0
    HOFdf.loc[HOFdf['HOF'] == 'O', 'Odur'] = HOFdf.loc[HOFdf['HOF'] == 'O', 'Dur'] 
    
    HOFdf['Hdur'] = 0
    HOFdf.loc[HOFdf['HOF'] == 'H', 'Hdur'] = HOFdf.loc[HOFdf['HOF'] == 'H', 'Dur'] 
    
    HOFdf['Fdur'] = 0
    HOFdf.loc[HOFdf['HOF'] == 'F', 'Fdur'] = HOFdf.loc[HOFdf['HOF'] == 'F', 'Dur'] 
    
    HOFdf['Rdur'] = 0
    HOFdf.loc[HOFdf['HOF'] == 'R', 'Rdur'] = HOFdf.loc[HOFdf['HOF'] == 'R', 'Dur'] 
    
    ###################################
    # create the POLicy dataframe
    def ListSet(lst):
        return ''.join(map(str, list(lst)))
    
    POLdf = HOFdf.groupby(['StudySession', 'Id']).agg(
        {'Time':firstLabel, 
         'Tstyles':firstLabel, 
         'Dur': 'sum', 
         'Odur': 'sum', 
         'Hdur': 'sum', 
         'Rdur': 'sum', 
         'Fdur': 'sum', 
         'Ins': 'sum', 
         'Del': 'sum', 
         'HOF': ListSet, 
         'Dur_L' : 'sum',
         'Dur_R' : 'sum',
         'Dur_N' : 'sum',
         'Dur_S' : 'sum',
    }).reset_index()
    

    POLdf['Study'] = AU.Study[0]
    POLdf['Session'] = AU.Session[0]
    POLdf['Id'] = range(1, len(POLdf) +1)
    POLdf['SL'] = AU.SL[0]
    POLdf['TL'] = AU.TL[0]
    POLdf['Task'] = AU.Task[0]
    POLdf['Text'] = AU.Text[0]
    POLdf['Part'] = AU.Part[0]

    # add more features
    # relation Insertions - Deletions
    POLdf['InEff']  = POLdf['Del'] / (POLdf['Ins'] + POLdf['Del'])
    
    POLdf['LogDur'] = np.log(POLdf['Dur'] + 1)
    POLdf['LogOdur']  = np.log(POLdf['Odur'] + 1)
    POLdf['LogHdur']  = np.log(POLdf['Hdur'] + 1)
    POLdf['LogRdur']  = np.log(POLdf['Rdur'] + 1)
    POLdf['LogFdur']  = np.log(POLdf['Fdur'] + 1)
    
    POLdf['RelOdur']  = POLdf['Odur'] / POLdf['Dur']
    POLdf['RelHdur']  = POLdf['Hdur'] / POLdf['Dur']
    POLdf['RelRdur']  = POLdf['Rdur'] / POLdf['Dur']
    POLdf['RelFdur']  = POLdf['Fdur'] / POLdf['Dur']
    
    POLdf['RelDur_L'] = POLdf['Dur_L'] / POLdf['Dur']
    POLdf['RelDur_R'] = POLdf['Dur_R'] / POLdf['Dur']
    POLdf['RelDur_S'] = POLdf['Dur_S'] / POLdf['Dur']
    POLdf['RelDur_N'] = POLdf['Dur_N'] / POLdf['Dur']
    
    POLdf.replace([np.inf, -np.inf], 0, inplace=True)
    POLdf.replace(np.nan, 0, inplace=True)


    #######################################################
    # Phase Layer (PH)
    
    PHdf = AU.groupby(['StudySession', 'Phase']).agg({'Time':firstLabel, 'Dur': 'sum', 'Tstyles': firstLabel,  'HOF': [UNITtypeSkip, 'count'], 'Ins': 'sum', 'Del': 'sum'}).reset_index()
    PHdf.columns = ['StudySession', 'Phase', 'Time', 'Dur', 'Tstyles', 'HOFseq', 'HOFnbr', 'Ins', 'Del']
    PHdf = PHdf.merge(SS, on='StudySession')

    PHdf['Study'] = AU.Study[0]
    PHdf['Session'] = AU.Session[0]
    PHdf['Id'] = range(1, len(PHdf) +1)
    PHdf['SL'] = AU.SL[0]
    PHdf['TL'] = AU.TL[0]
    PHdf['Task'] = AU.Task[0]
    PHdf['Text'] = AU.Text[0]
    PHdf['Part'] = AU.Part[0]

    PHdf['LogDur'] = np.log(PHdf['Dur'] +1) 
    PHdf['RelDur'] = PHdf['Dur'] / PHdf['TotalDur'] 

    H = {}
    H['au'] = AU
    H['kb'] = KBdf
    H['pu'] = PUdf
    H['hof'] = HOFdf
    H['pol'] = POLdf
    H['ph'] = PHdf
    
    return H


def PolicyClasses(P) :
    # insert columns for classification three classes
    # 0: values 0 1: 0 - mean 2: mean - max
    #PL1 = P[POLdf.StudySession.str.startswith('BML12-')].copy()

    # quantile: same number of instances in each class
    P['D'] = P['O'] = P['H'] = P['R'] = P['F'] = P['I'] = 0
    
    if(len(set(P.Dur)) > 4) :  P['D'] = pd.qcut(P.Dur, q=3, labels=False, duplicates='drop')
    if(len(set(P.RelOdur)) > 4) :  P['O'] = pd.qcut(P.RelOdur, q=3, labels=False, duplicates='drop')
    if(len(set(P.RelHdur)) > 4) :  P['H'] = pd.qcut(P.RelHdur, q=3, labels=False, duplicates='drop')
    if(len(set(P.RelRdur)) > 4) :  P['R'] = pd.qcut(P.RelRdur, q=3, labels=False, duplicates='drop')
    if(len(set(P.RelFdur)) > 4) :  P['F'] = pd.qcut(P.RelFdur, q=3, labels=False, duplicates='drop')
    if(len(set(P.InEff)) > 4) :  P['I'] = pd.qcut(P.InEff, q=3, labels=False, duplicates='drop')

    # add 'P' to make it stay a string
    P['DOHRFI'] = list('P:'+P.D.astype(str) + P.O.astype(str) + P.H.astype(str) + P.R.astype(str) + P.F.astype(str) + P.I.astype(str))

    
#############################################################
# ST_TT_id is one of {'TGid', 'SGid'}
# Insert the Min word Id in preceeding 
# Compute Word Id per AU
def WordIds(AU, ST_TT_id) :

    ST_TT_min = f'{ST_TT_id}Min'
    # each typing AU has one or more TT word 
    AU[ST_TT_min] = AU[ST_TT_id]
    AU[ST_TT_min] = [min(m.split('+')) for m in AU[ST_TT_id]]
    
    ## initialize non-typing AUs to 0
    AU.loc[AU[ST_TT_min] == '---', ST_TT_min] = 0
    
    ss = ''
    w = -1    
    for k in reversed(AU.index.tolist()):
        if(ss != AU.loc[k, 'StudySession']): 
            ss = AU.loc[k, 'StudySession']
    
            # the first AUs of the previous session
            # last AUs have TGidMin == 0
            if(w != -1) : AU.loc[k+1:w, ST_TT_min] = AU.loc[w, ST_TT_min]
            w = k
        
        
        if(AU.loc[k, ST_TT_min] == 0): continue
            
        AU.loc[k+1:w, ST_TT_min] = AU.loc[w, ST_TT_min]
        w = k
    
    # convert to integer
    AU[ST_TT_min] = AU[ST_TT_min].astype(int) 


# Add ST - TT Words
# TGidMin: TT word ID typed  
# SGwords: ST word ID typed
# TGwords: TT word ID typed
# TGids: for all AUs 
# SGids: for all AUS

# retrieve ST sords from st files
ST_TTwords = {}
def mapSTwords(studySession, STids, TTids):
    src = []
    tgt = []
    if(studySession not in ST_TTwords) :
        study, session = studySession.split('-')
        # Read TT words
        tt = pd.read_csv(f"/data/critt/tprdb/TPRDB/{study}/Tables/{session}.tt", sep="\t")
        for i,t in zip(tt['Id'].astype(str), tt['TToken']) : 
            ST_TTwords.setdefault(studySession, {})
            ST_TTwords[studySession].setdefault('tt', {})
            ST_TTwords[studySession]['tt'][i] = t

        # Read ST words
        st = pd.read_csv(f"/data/critt/tprdb/TPRDB/{study}/Tables/{session}.st", sep="\t")
        for i,t in zip(st['Id'].astype(str), st['SToken']) : 
            ST_TTwords.setdefault(studySession, {})
            ST_TTwords[studySession].setdefault('st', {})
            ST_TTwords[studySession]['st'][i] = t     
    if(studySession in ST_TTwords) :
#        print(Ids)
        for i in STids:
            if(i in ST_TTwords[studySession]['st']) :
                src.append(ST_TTwords[studySession]['st'][i])
            else:
                print(f"{studySession}: ST id:{i} not {STids}")
        for i in TTids:
            if(i in ST_TTwords[studySession]['tt']) :
                tgt.append(str(ST_TTwords[studySession]['tt'][i]))
            else:
                print(f"{studySession}: TT id:{i} not {TTids}")
    else: print(f"did not find: {studySession}")
#    print(f"List: {"+".join(src)}")
    return "+".join(src), "+".join(tgt)


###################################################
# Add TGid to previous rows
# ---> needs revision!
def ST_TTsegments(AU):
    # each typing AU has one or more TT word 
    AU['TGwords'] = AU['TGid']
    AU['SGwords'] = AU['SGid']
    AU['SGids'] = ''
    AU['TGids'] = ''
    AU['SSegIds'] = ''

    SS = ''  
    l = k = -1
    sg = '---'
    # assign ST words
    for i, r in AU.iterrows():
#        if (i > 40): break
        if (r['SGwords'] == '---') :
            if (r['StudySession'] != SS): 
                SS = r['StudySession']
                k = i
                if (sg != '---'): 
#                    AU.loc[l:i, 'TGwords'] = tg
                    AU.loc[l:i, 'SGwords'] = sg
                    AU.loc[l:i, 'SGids'] = sgId
                    k = -1
            elif (k == -1): k = i 
        elif (r['SGwords'] != '---'):
            l = i
            # set TTid to []
            sg, tg = mapSTwords(SS, r['SGid'].split('+'), [])
            sgId = r['SGid']
            
            # assign sg to current AU 
            AU.loc[i, 'SGwords'] = sg
            AU.loc[i, 'SGids'] = sgId
#            print("BBBB", r['SGid'], sgId, sg)
            
            # assign sg to previous type 1&2 AU 
            if (k != -1):
                AU.loc[k:i, 'SGwords'] = sg
                AU.loc[k:i, 'SGids'] = sgId
                k = -1
    AU.loc[l:i, 'SGwords'] = sg
    AU.loc[l:i, 'SGids'] = sgId

    SS = ''  
    l = k = -1
    tg = '---'
    # assign TT words
    for i, r in AU.iterrows():
#        if (i > 40): break
        if (r['TGwords'] == '---') :
            if (r['StudySession'] != SS): 
                SS = r['StudySession']
                k = i
                if (tg != '---'): 
#                    AU.loc[l:i, 'TGwords'] = tg
                    AU.loc[l:i, 'TGwords'] = tg
                    AU.loc[l:i, 'TGids'] = tgId
                    k = -1
            elif (k == -1): k = i 
        elif (r['TGwords'] != '---'):
            l = i
            # set STid to []
            sg, tg = mapSTwords(SS, [], r['TGid'].split('+'))
            tgId = r['TGid']
            # current AU
            AU.loc[i, 'TGwords'] = tg
            AU.loc[i, 'TGids'] = tgId
            
            # assign tg to previous AU 
            if (k != -1):
                AU.loc[k:i, 'TGwords'] = tg
                AU.loc[k:i, 'TGids'] = tgId
                k = -1
    AU.loc[l:i, 'TGwords'] = tg
    AU.loc[l:i, 'TGids'] = tgId

##############################################################
# AU annotation
##############################################################


###############################################################
# Heuristic/Rule-based HOF state learning
###############################################################

def HOF_states_heuristic(fn) :
    AUdf = pd.read_csv(fn, sep="\t", dtype=None)
    
    # some sessions don't have AUs
    if(len(AUdf) <1) : 
        print(fn, "empty") 
        return AUdf
    
    # add keystroke and gazepath features
    AUdf = add_HOF_Training_Features(AUdf, verbose=0)
    
    # add KD and PU features (KBtype, PUnbr, KBnbr, PUdur, KBdur)
    markProductionUnits(AUdf)
    
    # extract PU features
    PUdf = PUfeatures(AUdf)
    
    # mark the HOF states in AUf
    markHOFstates(AUdf, PUdf)

    # annotate AUs between two marked HOF states
    joinHOFstates(AUdf)

    #################################
    # Check remaining unassigned AUs
    A = AUdf[AUdf.HOF == '---']

    # Reverse the DataFrame
    Arev = A.iloc[::-1]
    
    # Iterate through the reversed DataFrame
    for index, row in Arev.iterrows():
        i = index
        while (AUdf.loc[i, 'HOF'] == '---'): i -= 1

        # assign last HOF state
        AUdf.loc[i:index, 'HOF'] = AUdf.loc[i, 'HOF']

    #################################
    # Assign running HOF number 
    AUdf['HOFnbr'] = (AUdf['HOF'] != AUdf['HOF'].shift()).cumsum()
    
    '''
    AUdf['HOFnbr'] = 0
    HOFnbr = 1
    HOF = ''
    start = 0
    for i in AUdf.index.tolist():
        if(HOF == '' ) : 
            HOF = AUdf.loc[i, 'HOF']
            start = i
            
        if(AUdf.loc[i, 'HOF'] != HOF) :
            k = i-1
            AUdf.loc[start:k, 'HOFnbr'] = HOFnbr
            HOF = AUdf.loc[i, 'HOF']
            HOFnbr += 1
            start = i
    k = AUdf.index.tolist()[-1]
    AUdf.loc[start:k, 'HOFnbr'] = HOFnbr  
    '''
    
    return AUdf

###################################################################################
# add additional features to AUdata in TPRDB
###################################################################################

def add_HOF_Training_Features(AUdata, verbose = 0) :
    
    # add Keystroke features to AUs
    # Add Effort/ Effect/ Relevance and TU features: 
    # probably not needed
#    AUdata = addKeystrokeData(AUdata)
    
    ### Takanori's gaze path method    
    AUss = AUdata[['Study', 'Session']].drop_duplicates()
    study = AUss['Study'].values[0]
    sessions = AUss['Session']
            
    # read the FD file
    FDdf = readSessions(sessions, '*fd',  path=f"/data/critt/tprdb/TPRDB/{study}/Tables/", verbose = 0)
        
    # add features fixation assign AIid to AU 
    AUgazePathFeatures(FDdf, AUdata)

    ###########
    # assign {L,R,S} label to every fixation
    FDdf = label_fixations_in_au(FDdf, line_break_dx_threshold=730, max_stid_increment=6, refix_dx=26, linear_chain=2, refix_chain=1)

    # collect {L,R,S, N} durations per AU 
    AUgazePath = mapFD_GPlabel2AUs(AUdata, FDdf)
    
    # merge gaze path label into AU
    AUdata = AUdata.merge(AUgazePath, on=['StudySession', 'Id'], how='left')
    
    # merge gaze path label and duration into AU
    AUdata = GPlabelDur(FDdf, AUdata)

    return AUdata



############################################################################
# Production Units
############################################################################

# Mark Tasks and Segments
# Add TGid to previous rows

def markProductionUnits(AU):
    # each typing AU has one or more TT word 
    AU['KBtype'] = '0' # type of KB / pause
    AU['PUnbr'] = 0 # number of PU
    AU['PUdur'] = 0
    AU['KBnbr'] = 0 # number of KB
    AU['KBdur'] = 0 # Duration of Break/Interruption
    
    KBdur = 0 # duration
    KBnbr = 1 # index of KB
    KIdur = 0  # duration of keystroke KIdur
    start = 0  # AU start index of Pause or KB
    delete = insert = 0
    
    # initialize values for KB features
    for i, r in AU.iterrows():
        if(KBdur == 0 and KIdur == 0): start = i

        # AUs can be keystrokes (4) or pause (~4)
        # pausing AUs
        if (not r.Type & 4) :
            
            # increase duration of pause 
            KIdur += r.Dur
            
#            print(start, i, KBdur, "\tdel", delete, 'ins:', insert)
                                   
            # keep values if this is first writing AU after pause
            if(KBdur > 0) :
                L = "I" # default: insert keystrokes
                if(delete > 0) :
                    if (insert > 0) : L = "C" # add and delete
                    else: L = "D" # only deletions

                # assign label to KBdur
                k = i-1
                AU.loc[start:k, 'KBtype'] = L
                AU.loc[start:k, 'KBdur'] = KBdur
                AU.loc[start:k, 'KBnbr'] = KBnbr

                # reset values for Key-Burst
                KBdur = delete = insert = 0
                KBnbr +=1 # increase unit counter
                start = i  # start of Pause
                            
        # writing AU
        if (r.Type & 4) :
            # the first typing AU
            # assign values for preceding Pause
            if(KIdur > 0) :

                # previous pause
                k = i-1
                AU.loc[start:k, 'KBdur'] = KIdur
                AU.loc[start:k, 'KBnbr'] = KBnbr   # make KIdur as KBdur:0
                
                if(KIdur > r.KBI) :
                    AU.loc[start:k, 'KBtype'] = 'K'
                    KBnbr +=1  # increase counter for KB
                    
                if(KIdur > r.PUB): 
                    AU.loc[start:k, 'KBtype'] = 'P'
                    
            # reset KIdur 
            # start: beginning of the KB
                KBdur = 0                 
                KIdur = 0
                start = i # start of Burst
                
            # duration of keystroke burst 
            KBdur += r.Dur
            delete += r.Del
            insert += r.Ins
    ### loop ends here
    
    # last piece should be pause
    if(KIdur > r.PUB): 
        AU.loc[start:i, 'KBtype'] = 'P'
    else: AU.loc[start:i, 'KBtype'] = 'K'

    AU.loc[start:i, 'KBdur'] = KIdur
    AU.loc[start:i, 'KBnbr'] = KBnbr

    ###########################################
    # find PUs
    PUdur = 0  # production Unit Duation 
    PBdur = 0  # production break Duation 
    PUnbr = 0  # 
    for i, r in AU.iterrows():
        
        if(PUnbr == 0): 
            start = i
            PUnbr = 1
        # production break
        if(r.KBtype == 'P') :
#            AU.loc[i, 'PUnbr'] = 0 # mark pause as 0
            PBdur += r.Dur
            
            if(PUdur > 0):
                AU.loc[start:last, 'PUdur'] = PUdur
                AU.loc[start:last, 'PUnbr'] = PUnbr
                PUnbr += 1
                PUdur = 0
                start = i #reset beginning of chunk
        # inside PU 
        else : 
            if(PBdur > 0):
                AU.loc[start:last, 'PUdur'] = PBdur
                AU.loc[start:last, 'PUnbr'] = PUnbr
                PUnbr += 1
                PBdur = 0
                start = i

            PUdur += r.Dur

        # the last AU
        last = i
        
    AU.loc[start:i, 'PUdur'] = PBdur
    AU.loc[start:i, 'PUnbr'] = PUnbr


def PUfeatures(AU) :
    PUfeat = { 'Dur' :["sum"], 
                'Ins' :["sum"],
                'Del' :["sum"],
#                'KBdur' :["sum"],
                'Dur_L' :["sum"],
                'Dur_R' :["sum"],
                'Dur_S' :["sum"],
                'Dur_N' :["sum"]
              
    }
    
    PU = AU.groupby(['PUnbr']).agg(PUfeat).reset_index()
    
    flat_cols = [] 
    # iterate through tuples and join them as single string
    for i in PU.columns:
        flat_cols.append(i[0]+i[1])
    
    PU.columns = flat_cols
    return PU


###################################################################

def readSessions(sessions, ext,  path="/data/critt/tprdb/TPRDB/", verbose = 0):
    
    df = pd.DataFrame()
    for sess in sessions:
        if(verbose > 1) :
            for fn in glob.glob(path + sess + ext): 
                print("\t", sess, "\t", fn)
        # list of tables (dataframes) 
        l = [pd.read_csv(fn, sep="\t", dtype=None) for fn in glob.glob(path + sess + ext)]
        # print filename # sessions rows
        if(verbose) :
            row = 0
            for d in l: row += len(d.index)
            print(f"{sess}\t#sessions:{len(l)}\t{ext}:{row}")
        l.insert(0, df)
        df = pd.concat(l, ignore_index=True)
    return(df)


#################################################################
# GAZE PATH COMPUTATION
#################################################################

def AUgazePathFeatures(FDdf, AUdf, verbose = 0):

    FDdf["StudySession"] = FDdf["Study"] + '-' + FDdf["Session"]
    AUdf["StudySession"] = AUdf["Study"] + '-' + AUdf["Session"]

# distance to next fixations
    FDdf["Xnext1"] = FDdf["X"].shift(-1)
    FDdf["Ynext1"] = FDdf["Y"].shift(-1)

    FDdf['Xnext1'] = FDdf['Xnext1'].fillna(0)
    FDdf['Ynext1'] = FDdf['Ynext1'].fillna(0)
    
    FDdf['XN_Dist1'] = FDdf["Xnext1"] - FDdf["X"]
    FDdf['YN_Dist1'] = FDdf["Ynext1"] - FDdf["Y"]

    # z-score of X and Y distances
    FDdf['XN_Dist1Z'] = (FDdf.XN_Dist1 - FDdf.XN_Dist1.mean()) / FDdf.XN_Dist1.std()
    FDdf['YN_Dist1Z'] = (FDdf.YN_Dist1 - FDdf.YN_Dist1.mean()) / FDdf.YN_Dist1.std()

    X2d1 = FDdf['XN_Dist1'] * FDdf['XN_Dist1']
    Y2d1 = FDdf['YN_Dist1'] * FDdf['YN_Dist1']
    
    # fixation distances
    FDdf['FixN_Dist1'] = np.sqrt(X2d1 + Y2d1)
    FDdf['FixN_Dist1Log'] = np.log(FDdf['FixN_Dist1'] + 1)
    
    # z-score of fixation distances
    FDdf['FixN_Dist1Z'] = (FDdf.FixN_Dist1 - FDdf.FixN_Dist1.mean())/ FDdf.FixN_Dist1.std()

# distance to previous fixations
    FDdf["Xprev1"] = FDdf["X"].shift(1)
    FDdf["Yprev1"] = FDdf["Y"].shift(1)

    FDdf['Xprev1'] = FDdf['Xprev1'].fillna(0)
    FDdf['Yprev1'] = FDdf['Yprev1'].fillna(0)
    
    FDdf['XP_Dist1'] = FDdf["X"] - FDdf["Xprev1"]
    FDdf['YP_Dist1'] = FDdf["Y"] - FDdf["Yprev1"] 
    
    FDdf['XP_Dist1Z'] = (FDdf.XP_Dist1 - FDdf.XP_Dist1.mean()) / FDdf.XP_Dist1.std()
    FDdf['YP_Dist1Z'] = (FDdf.YP_Dist1 - FDdf.YP_Dist1.mean()) / FDdf.YP_Dist1.std()

    X2d1 = FDdf['XP_Dist1'] * FDdf['XP_Dist1']
    Y2d1 = FDdf['YP_Dist1'] * FDdf['YP_Dist1']
    
    # fixation distances
    FDdf['FixP_Dist1'] = np.sqrt(X2d1 + Y2d1)
    FDdf['FixP_Dist1Log'] = np.log(FDdf['FixP_Dist1'] + 1)
    FDdf['FixP_Dist1Z'] = (FDdf.FixP_Dist1 - FDdf.FixP_Dist1.mean())/ FDdf.FixP_Dist1.std()


# distance over 2 fixations, not used
#    FDdf["Xnext2"] = FDdf["X"].shift(-2)
#    FDdf["Ynext2"] = FDdf["Y"].shift(-2)
#    FDdf['Xnext2'] = FDdf['Xnext2'].fillna(0)
#    FDdf['Ynext2'] = FDdf['Ynext2'].fillna(0)
#    FDdf['Xdist2'] = FDdf["Xnext2"] - FDdf["X"]
#    FDdf['Ydist2'] = FDdf["Ynext2"] - FDdf["Y"]
#    FDdf['Xdist2Z'] = (FDdf.Xdist2 - FDdf.Xdist2.mean()) / FDdf.Xdist2.std()
#    FDdf['Ydist2Z'] = (FDdf.Ydist2 - FDdf.Ydist2.mean()) / FDdf.Ydist2.std()
   
#    X2d2 = FDdf['Xdist2'] * FDdf['Xdist2']
#    Y2d2 = FDdf['Ydist2'] * FDdf['Ydist2']
    
#    FDdf['FixDist2'] = np.sqrt(X2d2 + Y2d2)
#    FDdf['FixDist2Log'] = np.log(FDdf['FixDist2'] + 1)

#    FDdf['FixDist2Z'] = (FDdf.FixDist2 - FDdf.FixDist2.mean())/ FDdf.FixDist2.std()

    ##########################################################
    # Angle between successive fixations currently not used
    # angle between 0 -- 360 degree
    def FixAngle(x2, y2):        
        angle = []
        for i in range(len(x2) - 1) :
            a = np.arctan2(y2[i], x2[i]) # - np.arctan2(y1[i], x1[i])
            angle.append(np.degrees(a))
        angle.append(np.nan)
        return angle

    FDdf['Angle1'] = FixAngle(list(FDdf["XN_Dist1"]), list(FDdf["YN_Dist1"])) 
#    FDdf['Angle2'] = FixAngle(list(FDdf["Xdist2"]), list(FDdf["YN_Dist2"])) 
    ##########################################################

    # new feature 
    FDdf['AUid'] = 0
    AUdf["End"] = AUdf["Time"] + AUdf["Dur"]


    ############################################################
    # ADD AUid to FDdf 
    Sessions = set(AUdf["StudySession"])
    for session in Sessions:
        
        AUs = AUdf[(AUdf["StudySession"] == session)]
        FDs = FDdf[(FDdf["StudySession"] == session)]

        if(verbose): print(f"AUgaze:{session} #AUs:{len(AUs)}")
            
        # loop over start-end AUs
        for start, end in zip(list(AUs["Time"]), list(AUs["End"])):

            AUid = AUs[AUs.Time == start]["Id"].values[0]            
            FDdf.loc[((FDdf["StudySession"] == session) & (FDdf["Time"] >= start) & (FDdf["Time"] < end)), 'AUid'] = AUid
            
    return 


#########################
#########################################################
# Gaze-Path computation 1
# Label fixations with L, R, S

##################################################################################
# Takanori Gaze Path
##################################################################################
def label_fixations_in_au(fixations_df, 
                          line_break_dx_threshold=730, 
                          max_stid_increment=6,
                          linear_dx=300, 
                          linear_dy=150,
                          refix_dx=26, 
                          refix_dy=150,
                          linear_chain=2, 
                          refix_chain=1,
                          max_refix_skip=2):
    """
    GPlabel a group of fixations belonging to one AU (Activity Unit) in chunks.
    
    Parameters:
    -----------
    fixations_df : pd.DataFrame
        A time-sequential fixation dataset (with the same AUid).
        Columns: [X, Y, STid, (others)... ] etc. are assumed
    line_break_dx_threshold: int
        Threshold for determining a line break ("if dx moves back beyond this value" e.g., 1000)
    max_stid_increment: int
        Threshold for determining a line break ("if STid increment is less than or equal to this value" e.g., 3)
    linear_dx: int
        Maximum dx considered for Linear (e.g., 300px)
    linear_dy: int
        Maximum dy considered for Linear (e.g., 150px)
    refix_dx: int
        dx range considered for Re-fixation (e.g., within 50px)
    refix_dy: int
        Maximum dy considered for Re-fixation (e.g., 150px)
    linear_chain: int
        The number of consecutive pairs needed to qualify as Linear (3 consecutive pairs → 4 fixations in total)
    refix_chain: int
        The number of consecutive pairs needed to qualify as Re-fixation (2 consecutive pairs)
    max_refix_skip: int
        The maximum number of small regressions allowed in a row within a Linear chain (e.g., 2)
        Here it is set to 2, so up to two consecutive pairs are acceptable.
    
    Returns:
    --------
    pd.DataFrame
        Returns a DataFrame with the same number of rows as the input, adding a 'GPlabel' column.
        Each fixation is assigned either 'Linear', 'Re-fixation', or 'Scattered'.
    """
    
    df = fixations_df.copy()
    df['GPlabel'] = None 

    index_list = df.index.tolist()
    N = len(index_list)

    i = 0
    while i < N - 1:
        # Acquire the pair of fixations[i] and fixations[i+1]
        idx_i  = index_list[i]
        idx_i1 = index_list[i+1]

        xi     = df.at[idx_i,  'X']
        yi     = df.at[idx_i,  'Y']
        stidi  = df.at[idx_i,  'STid']
        ttidi  = df.at[idx_i,  'TTid']

        xi1    = df.at[idx_i1, 'X']
        yi1    = df.at[idx_i1, 'Y']
        stidi1 = df.at[idx_i1, 'STid']
        ttidi1 = df.at[idx_i1, 'TTid']

        # get reading direction from the “TL” column
        _rtl_langs = {'ar'}  # right-to-left language codes (extend if needed)
        tl_code = str(df.iloc[0]['TL']).lower() if 'TL' in df.columns else ''
        dir_fac = -1 if tl_code in _rtl_langs else 1   # 1 = LTR, –1 = RTL
        
        # Calculate the saccade
#        dx = xi1 - xi
        # make dx positive for “forward” in either direction
        dx  = (xi1 - xi) * dir_fac

        dy = yi1 - yi

        # Line break detection
        line_break_saccade = False
        if (dx < -line_break_dx_threshold) and (
            (1 <= (stidi1 - stidi) <= max_stid_increment)
            or
            (1 <= (ttidi1 - ttidi) <= max_stid_increment)
        ):
            line_break_saccade = True
# commented out
#            print("There might be a line break.") 

        # -----------------------------
        # (A) Linear detection
        # -----------------------------
        linear_candidate = ((dx > refix_dx and dx <= linear_dx and abs(dy) <= linear_dy)
                            or line_break_saccade)

        if linear_candidate:
            # Starting from this pair, compute how far the "Linear chain" can continue
            linear_count = 1  # First, we have one pair
            chain_start = i
            chain_end_idx = i + 1  # The first fix after forming a pair is i+1
            j = i + 1

            # A counter to track how many consecutive "small regressions (Re-fixation conditions)" pairs occur
            # If this exceeds max_refix_skip (2), stop the Linear determination
            refix_in_a_row = 0  

            while j < N - 1:
                # Get the next pair
                idx_j  = index_list[j]
                idx_j1 = index_list[j+1]
                xj     = df.at[idx_j,  'X']
                yj     = df.at[idx_j,  'Y']
                stidj  = df.at[idx_j,  'STid']
                ttidj  = df.at[idx_j,  'TTid']
                xj1    = df.at[idx_j1, 'X']
                yj1    = df.at[idx_j1, 'Y']
                stidj1 = df.at[idx_j1, 'STid']
                ttidj1 = df.at[idx_j1, 'TTid']

#                dx2 = xj1 - xj
                dx2 = (xj1 - xj) * dir_fac
                dy2 = yj1 - yj

                lb2 = False
                if (dx2 < -line_break_dx_threshold) and (
                    (1 <= (stidj1 - stidj) <= max_stid_increment)
                    or
                    (1 <= (ttidj1 - ttidj) <= max_stid_increment)
                ):
                    lb2 = True

# commented out
                    #print("There might be a line break.")

                # Check if this is a "Linear pair"
                next_linear_candidate = ((dx2 > refix_dx and dx2 <= linear_dx and abs(dy2) <= linear_dy)
                                         or lb2)
                # Check if this is a "small regression (equivalent to Re-fixation) pair"
                next_ref_candidate = (abs(dx2) <= refix_dx and abs(dy2) <= refix_dy)

                if next_linear_candidate:
                    # If it's a fully Linear pair, increment linear_count and reset refix_in_a_row
                    linear_count += 1
                    refix_in_a_row = 0
                    j += 1
                    chain_end_idx = j + 1  # The "back side" fix of j pair is j+1
                elif next_ref_candidate:
                    # In the case of a small regression pair (abs(dx2)<=refix_dx, etc.)
                    refix_in_a_row += 1
                    if refix_in_a_row <= max_refix_skip:
                        # If up to two consecutive pairs, treat it as continuing the Linear chain
                        # However, do not increment linear_count; consume the pair (advance j)
                        j += 1
                        chain_end_idx = j + 1  
                    else:
                        # If more than two consecutive pairs (i.e. 3 or more), then terminate the Linear chain
                        break
                else:
                    # If neither (e.g., a big jump), break the chain
                    break

            # Check how many pairs the chain continued
            if linear_count >= linear_chain:
                # Example: fix(1.. chain_end_idx-1) all labeled Linear
                for fix_idx in range(chain_start, chain_end_idx):
                    df.at[index_list[fix_idx], 'GPlabel'] = 'L' # Linear
                # Since it's already processed collectively, skip the chunk all at once
                i = chain_end_idx  
                # Return to the top of the outer while loop
                continue  
            else:
                # If fewer than three in a row, proceed to Re-fixation determination with the same i
                # Without incrementing i, we'll move on to Re-fixation
                pass  

        # -----------------------------
        # (B) Re-fixation determination
        # -----------------------------
        # If Linear wasn't considered or the 3-consecutive chain failed, check Re-fixation here
        re_candidate = (abs(dx) <= refix_dx and abs(dy) <= refix_dy)

        if re_candidate:
            count = 1
            j = i + 1
            while j < N - 1:
                idx_j  = index_list[j]
                idx_j1 = index_list[j+1]
                xj     = df.at[idx_j,  'X']
                yj     = df.at[idx_j,  'Y']
                xj1    = df.at[idx_j1, 'X']
                yj1    = df.at[idx_j1, 'Y']
                
                dx2 = xj1 - xj
                dy2 = yj1 - yj

                if (abs(dx2) <= refix_dx and abs(dy2) <= refix_dy):
                    count += 1
                    j += 1
                else:
                    break
            
            if count >= refix_chain:
                # 2 or more consecutive → definitely Re-fixation
                for offset in range(count+1):
                    if (i + offset) < N:
                        df.at[index_list[i + offset], 'GPlabel'] = 'R' # Re-fixation
                i += count
                continue
            else:
                # Fewer than 2 consecutive → do not assign a label, just increment i by 1
                i += 1
                continue
        else:
            # Not Re-fixation → increment i
            i += 1

    # *At the end of the loop, if the last fixation is unlabeled, label it as Scattered
    # =============================
    # Finally, assign Scattered all at once
    # =============================
    df.loc[df['GPlabel'].isna(), 'GPlabel'] = 'S' # Scattered

    return df

# End Takonori Gaze Path
#####################################################


# map FD gaze-path GPlabel into AUs:
# -- total gaze duration <= AU duration
# -- overlapping fixations are  

def mapFD_GPlabel2AUs(AU1, FD1, verbose = 0) :
    AU1["End"] = AU1["Time"] + AU1["Dur"]
    FD1["End"] = FD1["Time"] + FD1["Dur"]
    
    Sessions = set(AU1["StudySession"])
    
    H = {}
    for session in Sessions:
            
        AUs = AU1[(AU1["StudySession"] == session)]
        FDs = FD1[(FD1["StudySession"] == session)]
        
        # Initiaalize left over of fixation from previous AU
        O = {'Dur_L':0, 'Dur_R':0, 'Dur_S':0, 'Dur_N':0} 
        
        # loop over AUs 
        for start, end, Id in zip(list(AUs["Time"]), list(AUs["End"]), list(AUs["Id"])):

            # duration of AU
            AUdur = AUs.loc[AUs.Id == Id, 'Dur'].item()

            # initialize Dur of AU Id            
            H.setdefault(Id,{})
            H[Id] = {'StudySession': session, 
                     'Id': Id, 
                     'Dur_L':0, 'Dur_R':0, 'Dur_S':0, 'Dur_N':0}

            end0 = start
            
            # loop over GP GPlabel: hang-over from previous AU
            for o in O.keys():
                    
                # if there is an End timestamp > 0
                if(O[o] > 0) :
                    Fdur = sum([H[Id][lab] for lab in ['Dur_L','Dur_R','Dur_S','Dur_N']])
                    if(AUdur <= Fdur): break
                    
                    # fixation ends in current AU
                    if (O[o] < end) : 
                        if (O[o] > end0) : 
                            H[Id][o] = min(O[o] - start, AUdur-Fdur) 
                            end0 = O[o]
                        O[o] = 0
                        
#                        Fdur = sum([H[Id][lab] for lab in ['Dur_L','Dur_R','Dur_S','Dur_N']])
#                        print(f"\tO1: {o} {Id} AUdur/Fdur:{AUdur}/{Fdur} = {AUdur - Fdur}")
                    else:
                        # previous Fixation covers entire AU
                        H[Id][o] = end - start - Fdur
#                        Fdur = sum([H[Id][lab] for lab in ['Dur_L','Dur_R','Dur_S','Dur_N']])
#                        print(f"\tO2: {o} {Id} AUdur/Fdur:{AUdur}/{Fdur} = {AUdur - Fdur}")
            
#            Fdur = sum([H[Id][lab] for lab in ['Dur_L','Dur_R','Dur_S','Dur_N']])
#            print(f"AU1 {Id} AUdur/Fdur:{AUdur}/{Fdur}={AUdur - Fdur}")
            
            # loop over fixation data
            F = FD1.loc[((FD1["StudySession"] == session) & (FD1["Time"] >= start) & (FD1["Time"] < end))]
            for i, fix in F.iterrows():              
                # duration of accumulated fixation
                Fdur = sum([H[Id][lab] for lab in ['Dur_L','Dur_R','Dur_S','Dur_N']])

                # fixation label 
                GP_lab = f"Dur_{fix.GPlabel}"

                # some fixations seem to overlap / be embedded in others: to be fixed
                if(fix.End <= end0) : continue
                
                # fixation inside AU
                if(fix.End <= end) : 
                    # there may be several concurrent fixations in AU ...
                    H[Id][GP_lab] += min(fix.End-end0, AUdur-Fdur)

                # fixation crosses AU boundary: fill remaining AU duration
                else :
                    H[Id][GP_lab] += AUdur - Fdur                    
                    O[GP_lab] = fix.End
                    break

                # memorize end of previous fixation
                if(fix.End > end0) : end0 = fix.End

#                print(f"\tB: {fix.Id}:{GP_lab} Overlap:{end - fix.End}\tf:{fix.Dur}\t{H[Id][GP_lab]}\tAUdur/Fdur:{AUdur}/{Fdur} = {AUdur - Fdur}")
#                print(GP_lab, 'AU:', end, fix.Time, fix.Dur, fix.End, 'H:', H[Id][GP_lab], 'O:', O[GP_lab])

            Fdur = sum([H[Id][lab] for lab in ['Dur_L','Dur_R','Dur_S','Dur_N']])
            H[Id]['Dur_N'] =  AUdur - Fdur
            
#            print("AU2", Id,  "Dur", AUdur, "FixDur", Fdur, "Dur_N", AUdur - Fdur)
#            print(H[Id])

    # convert into Dataframe and return
    return pd.DataFrame(H).T

# Assign GazePath label (GPlabel) 
def GPlabelDur(FU, AU):

    # count number of different words in Gaze Path

    AU['STs_TTs'] = 0
    AU['NextTime'] = AU['Time'].shift(-1)
    
    # go through all AUs
    for i, r in AU.iterrows() :
        t = r.Time    # beginning of AU 
        n = r.NextTime # end of AU 
        m = 0

        # ST reading
        if(r.Type & 1) :
            m = len(set(FU[(FU.Time >= t) & (FU.Time < n)].STid))
        # TT reading
        elif(r.Type & 2) :
            m = len(set(FU[(FU.Time >= t) & (FU.Time < n)].TTid))

        AU.loc[i, 'STs_TTs'] = m
    
    ###########################################################
    # Different AU GPlabel 
    # Calculate relative durations
    GP_features = ['Dur_L', 'Dur_R', 'Dur_S', 'Dur_N']
    x = AU[['Dur_L', 'Dur_R', 'Dur_S', 'Dur_N']].sum(axis=1)
    AU['Total_Dur'] = AU[GP_features].sum(axis=1)

    rel_columns = []
    AU['One'] = 1
    # add relative duration values
    for col in GP_features:
        rel_col = f'Rel{col}'
        rel_columns.append(rel_col) 
    
        # make sure sum rel is not > dur
        AU['MaxDur'] = AU[['Dur', 'Total_Dur', 'One']].max(axis=1)
        AU[rel_col] = AU[col] / AU['MaxDur']
#        AU[rel_col] = AU[rel_col].replace(np.nan, 0.0) 
#        AU[rel_col] = AU[rel_col].fillna(0.0) 

    # Assign GPlabel based on dur
    AU['GPlabel'] = 'N' 
    for index, row in AU.iterrows():
        # Annotate GPlabel if Relative Dur is 0.3 (30%) or higher
        GPlabel = [col.split('_')[-1] for col in rel_columns if row[col] >= 0.3]
        # Combine GPlabel or set 'N'
        AU.at[index, 'GPlabel'] = ''.join(GPlabel) if GPlabel else 'N'

    return AU



###################################################################################
# HOF states
###################################################################################

# 1) find sure HOF states
# 2) assign states between two sure states of same type
# 3) assign states between two sure states of different type
def markHOFstates(AU, PU) :

    # Initialize HOF annotations
    AU['HOF'] = '---'
    AU['WperFix'] = AU.STs_TTs / (AU.FixS + AU.FixT + 1)

    # find HOF state anchors
    i = 0
    while i < len(AU):
        row = AU.loc[i]

        #######
        # Long Pauses (KBtype == Pause)
        if ((row.KBtype == 'P') 
             & (row.PUdur >= row.PUB * 2) # long Pause
             & (row.Dur >= row.PUB) # long AU
             & (row.FixS + row.FixT >= 3)) : # ST readung or TT reading

    # Orientation during PUB 
            if ((row.Type == 1) # ST reading
                & (row.WperFix > 0.3)  # more than 30% new words
#                & (row.RelDur_L > 0.3) # linear reading more than 30%
                ):           
                AU.loc[i, 'HOF'] = 'O'

    # Reviewing during PUB 
            elif ((row.Type == 2) # TT reading
                & (row.WperFix > 0.3)  # more than 30% new words
                ):
                AU.loc[i, 'HOF'] = 'R'
        
    # Hesitation during PUB 
            elif ((row.WperFix < 0.3) 
                ):            
                AU.loc[i, 'HOF'] = 'H'

        #######
        # Long Pauses
        # Hesitation during long PUB type 8
        elif ((row.KBtype == 'P')  # 
            & (row.Type == 8) 
            & (row.Phase != 'O')
            & (row.Dur > row.PUB * 2))  :
                AU.loc[i, 'HOF'] = 'O'
                if (row.Dur > row.PUB * 4) : AU.loc[i, 'HOF'] = 'H'

        # Flow / Hesitation in long PU (KBtype != P(ause)
        elif ((row.KBtype != 'P') & (row.PUdur >= row.PUB)) :
            
            # extract PU data row and squeeze into Series
            pu = PU[(PU.PUnbr == row.PUnbr)].squeeze()
            # Flow PU
            if (pu.Inssum >= pu.Delsum): AU.loc[i,  'HOF'] = 'F'
            # Hesitation
            else: AU.loc[i,  'HOF'] = 'H'
        
            i = completePU_hof(AU, i)                
        i += 1
        
    return AU

# make sure all AUs in a PU have the same hof state
# return id of last AU in PU
def completePU_hof(AU, t) :

    hof = AU.loc[t, 'HOF']

    t0 = max(0, t-1)
    # complete to right while in PU (!= PUB)
    while((t0 > 0) & (AU.loc[t0, 'KBtype'] != 'P')) :
#        print("\tLR0:", t0, t, AU.loc[t0, 'KBtype'], AU.loc[t0, 'HOF'])        
        t0 -= 1

    if((AU.loc[t0, 'KBtype'] == 'P') | (AU.loc[t0, 'HOF'] == hof)) :
        t0 += 1
        AU.loc[t0:t, 'HOF'] = hof
#        print("LR1:", t0, t, hof)
    else :    
        print("LRx:", t0, t, AU.loc[t0, 'KBtype'], AU.loc[t0, 'HOF'])

    ####################
    t1 = min(t+1, len(AU))
    # complete to right while in PU (!= PUB)
    while((t1 < len(AU)) & (AU.loc[t1, 'KBtype'] != 'P')) :
            t1 += 1
        
    if(AU.loc[t1, 'KBtype'] == 'P') :
        t1 -= 1
        AU.loc[t:t1, 'HOF'] = hof
#        print("LR2:", t, t1, hof)

    return t1

# annotate AUs between two sure HOF states
def joinHOFstates(AU) :

    # set first AU 'O' 
    if(AU.loc[0, 'HOF'] == '---'): AU.loc[0, 'HOF'] = 'O'

    # loop through set of states
    i = 0
    while i < len(AU) -1 :
        
        # first AU label
        hof = AU.loc[i, 'HOF']
        
        # skip leading unannotated AUs
        if(hof == '---') :
            i +=1
            continue
            
        k = i+1
        h = AU.loc[k, 'HOF']
        
        # continue if next AU is annotated
        if(h != '---') : 
            i = k         
            continue
                    
        # loop through sequence of un-annotated states
        while(h == '---') :
            k += 1
            if(k >= len(AU)): 
                k -= 1
                break
            h = AU.loc[k, 'HOF']

        # if flanked by identical HOF label assign same label to in-between AUs
        if (h == hof) :
            AU.loc[i:k,'HOF'] = hof
        else :
        # interpolate if flanked by two different HOF labels
            interpolateHOFstates(AU, i, k)
        i = k
        
    return AU


def interpolateHOFstates(AU, t1, t2) :

    # left side label 
    hof1 = AU.loc[t1, 'HOF']
    # right side label 
    hof2 = AU.loc[t2, 'HOF']
    # lower gap
    t1 += 1
    t2 -= 1
    t3 = t1
    t0 = t2

    # Flow extend only with typing AUs
    if(hof1 == 'F'):
        h = AU.loc[t3, 'HOF']
        
        # all AUs until last typing get F label
        while (h == '---') : 
            # last F-AU is typing
            if(AU.loc[t3,  'Type'] & 4) :            
                AU.loc[t1:t3,  'HOF'] = 'F'
                t1 = t3
            t3 += 1
            if (t3 >= len(AU)): break
            h = AU.loc[t3, 'HOF']
            
        # remaining AUs get hof2 label
        AU.loc[t1:t2, 'HOF'] = hof2
        return

    if(hof2 == 'F'):
        h = AU.loc[t0, 'HOF']

        # all AUs until last typing get F label
        while ((t0 >= 0) & (h == '---')) : 
            # first F-AU is typing
            if(AU.loc[t0,  'Type'] & 4) :            
                AU.loc[t0:t2,  'HOF'] = 'F'
                t2 = t0
            t0 -= 1
            h = AU.loc[t0, 'HOF']
            
        # remaining AUs get hof1 label
        AU.loc[t1:t2, 'HOF'] = hof1
        return
    
        
    ########################
    # Hesitation: cover all types of AUs
    if(hof1 == 'H'):
        h = AU.loc[t1, 'HOF']
        while (h == '---'): 
            AU.loc[t1,  'HOF'] = 'H'
            t1 += 1
            if (t1 >= len(AU)): break
            h = AU.loc[t1, 'HOF']
        return

    if(hof2 == 'H'):
        h = AU.loc[t2, 'HOF']
        while (h == '---') : 
            AU.loc[t2,  'HOF'] = 'H'
            t2 -= 1
            if (t2 < 0): break
            h = AU.loc[t2, 'HOF']
        return


    ########################
    if(hof1 == 'O'):
        r = AU.loc[t1].squeeze()
        while ((r.HOF == '---') & 
               ((r.Type == 8) | (r.Type == 4) | (r.Type == 1) | (r.Dur < r.PUB))) : 

            AU.loc[t1,  'HOF'] = 'O'
            t1 += 1
            if (t1 >= len(AU)): break
            r = AU.loc[t1].squeeze()

    if(hof2 == 'O'):
        r = AU.loc[t2].squeeze()
        while ((r.HOF == '---') &  (t1 <= len(AU)) 
               & ((r.Type == 8) | (r.Type == 4) | (r.Type == 1) | (r.Dur < r.PUB))): 
            AU.loc[t2,  'HOF'] = 'O'
            t2 -= 1
            if (t2 < 0): break
            r = AU.loc[t2].squeeze()
    
    ########################
    if(hof1 == 'R'):
        r = AU.loc[t1].squeeze()
        while ((r.HOF == '---') & 
               ((r.Type == 8) | (r.Type == 4) | (r.Type == 2) | (r.Dur < r.PUB))) : 

            AU.loc[t1,  'HOF'] = 'R'
            t1 += 1
            if (t1 >= len(AU)): break
            r = AU.loc[t1].squeeze()

    if(hof2 == 'R'):
        r = AU.loc[t2].squeeze()
        while ((r.HOF == '---') & (t1 <= len(AU)) 
                & ((r.Type == 8) | (r.Type == 4) | (r.Type == 2) | (r.Dur < r.PUB))): 
            AU.loc[t2,  'HOF'] = 'R'
            t2 -= 1
            if (t2 < 0): break
            r = AU.loc[t2].squeeze()
        
    ########################
    # most likely the beginning of session
    if(hof1 == '---'):
        h = AU.loc[t1, 'HOF']
        while (h == '---'): 
            AU.loc[t1,  'HOF'] = hof2
            t1 += 1
            if (t1 >= len(AU)) : break
            h = AU.loc[t2, 'HOF']
            
    # most likely the end of session: assign last HOF state
    if(hof2 == '---'):
        h = AU.loc[t2, 'HOF']
        while (h == '---') : 
            AU.loc[t2,  'HOF'] = hof1
            t2 -= 1
            if (t2 < 0): break
            h = AU.loc[t2, 'HOF']
            
