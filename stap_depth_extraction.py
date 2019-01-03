#! /usr/bin/env python
# -*- coding: utf-8 -*

##########################################################################
# @author: yann.leguen@cea.fr
# Copyright (C) CEA, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html for details.
##########################################################################


import os
import json
import numpy as np
import pandas as pd
import nibabel.gifti.giftiio as gio
from multiprocessing import Pool
from multiprocessing import cpu_count

# Note: the shortestpath method is only available in Brainvisa
# trunk development branch (as of March 2018)
from soma import aims
from soma import aimsalgo as algo

from detect_peaks import detect_peaks


def build_dict_areals(labels, sulci, letters):
    """
    Build dictionary containing areal references

    Parameters
    labels: file path to dataframe containing areal/number matching
    sulci: list of the sulci names considered in the following analysis
    letters: list of letters that can be concatenated to each sulcus name
    """
    df = pd.read_csv(labels)
    df['Num'] = df.index+1
    df.index = df['Name']
    dict_ind = {}
    for sulc in sulci:
        # Only keep areal from main sulci, not labelled as junction or bis
        names = [j for j in df['Name']
                 if sulc in j and 'junct' not in j and 'bis' not in j]
        dict_ind[sulc] = {}
        dict_ind[sulc]['Name'] = df.loc[names]['Name']
        dict_ind[sulc]['Areal'] = df.loc[names]['Areal']
        dict_ind[sulc]['Num'] = df.loc[names]['Num']
    return dict_ind

def build_dict_pits_vertex(database, s_ids, sulci, letters, dict_ind):
    """
    Build dictionary containing vertex position
    of the sulcal pits for each subject
    
    Parameters
    database: path to the database directory containing the sulcal pits
    s_ids: list of of considered subject id
    sulci: list of considered

    """
    
    pits_vertex = {}
    for s_id in s_ids:
        print s_id
        pits_vertex[s_id] = {}
        path_s = os.path.join(database, s_id, 't1mri', 'BL',
                              'default_analysis', 'segmentation', 'mesh')
        for sd in ['L', 'R']:
            pits_vertex[s_id][sd] = {}
            # Subject sulcal pits texture
            pits = os.path.join(path_s, 'surface_analysis',
                                s_id+'_'+sd+'white_pits.gii')

            # Subject back-projected group-clusters of pits
            # Path to updated parcellation HCP S1200
            clusters = os.path.join(path_s, 'surface_analysis_update',
                                    s_id+'_'+sd+'clusters_sym_lh.gii')
            """# Path to parcellation HCP S900
            clusters = os.path.join(path_s, 'surface_analysis',
                                    s_id+'_'+sd+'white_clusters_sym_lh.gii')
            """
            # Subject DPF
            DPF = os.path.join(path_s, 'surface_analysis',
                               s_id+'_'+sd+'white_DPF.gii')

            pits_ar = gio.read(pits).darrays[0].data
            clusters_ar = gio.read(clusters).darrays[0].data
            #clusters_ar = clusters_ar.astype('int')
            DPF_ar = gio.read(DPF).darrays[0].data

            for sulc in sulci:
                pits_vertex[s_id][sd][sulc] = {}
                for k, A in enumerate(letters):
                    # Check if the areal exists
                    if  sulc+'_'+A in dict_ind[sulc]['Name']:
                        # Label number for the areal A
                        sect_A = dict_ind[sulc]['Areal'][sulc+'_'+A]
                        # Vertex belonging to areal A
                        ind_sectA = np.where(float(sect_A) == clusters_ar)[0]
                        # Index of pits in areal A
                        ind_pits_sectA = np.nonzero(pits_ar[ind_sectA])[0]

                        # Choose the sulcal pit in the areal that will be
                        # used as extremity of the sulcal depth geodesic path
                        """
                        In the UK Biobank and to avoid using wrongly
                        labelled deep sulcal pits, lying in sulcal wall,
                        we selected the deepest point in the areal.
                        This corresponds in most cases, for areals
                        sup_temporal b, c, d, to select the true sulcal pit.
                        
                        **** 
                        Note
                        The following code should be updated to use a fixed
                        threshold (cf paper on the sulcal pits) to determine
                        if the sulcal pit is deep or not (in term of DPF and
                        geodesic depth).
                        ****
                        
                        """
                        
                        
                        # If no pit in areal A,
                        if len(ind_pits_sectA) == 0:
                            # Then we just take the deepest point
                            # NB: This could be the best option,                    
                            # because sometimes pits are "misplaced"
                            # pits are sometimes identified in sulcal wall
                            # (see Supplementary Fig 1 paper Neuroimage 2018)
                            pit_sectA = ind_sectA[np.argmax(DPF_ar[ind_sectA])]
                        # If only one pit, then we select it
                        elif len(ind_pits_sectA) == 1:
                            pit_sectA = ind_sectA[ind_pits_sectA[0]]
                        # If several pits, then we take the deepest
                        else: 
                            pit_sectA = ind_sectA[ind_pits_sectA
                                                  [np.argmax(
                                                      DPF_ar[
                                                      ind_sectA[
                                                          ind_pits_sectA]])]]

                        # Assign the vertex position in the dictionary
                        pits_vertex[s_id][sd][sulc][A] = pit_sectA
    return pits_vertex


def build_depth_profil_STAP(parameters):
    """
    Extract the geodesic path along the bottom of the STAP
    between sulcal pits sup_temporal_b and sup_temporal_d
    and truncates this path at the border between areal c and d

    Parameters
    
    database: path to the database directory containing the sulcal pits
    dict_ind: dictionary containing areal name/numero correspondence
    s_id: subject id
    pits_vertex: dictionary containing vertex position of pits for each s_id
    outdir: output directory that will contain .npy file of the DPF, geodesic
            depth and position [mm] along the STAP
    """
    database, dict_ind, s_id, pits_vertex, outdir = parameters
    # Get areal index for sup temporal b and c
    sulc, B, C, D = 'sup_temporal', 'b', 'c', 'd'
    iB = dict_ind[sulc]['Areal']['sup_temporal_b']
    iC = dict_ind[sulc]['Areal']['sup_temporal_c']

    # Set the subject input directory
    path_s = os.path.join(database, s_id, 't1mri', 'BL',
                          'default_analysis', 'segmentation', 'mesh')

    for sd in ['L', 'R']:
        # Set the output directories for each files
        outdir2 = os.path.join(outdir, 'STAP_specific', 'DPF', sd)
        outdir3 = os.path.join(outdir, 'STAP_specific', 'GeoDepth', sd)
        outdir4 = os.path.join(outdir, 'STAP_specific', 'positions_profil', sd)

        output2 = os.path.join(outdir2, '.'.join([s_id, sd, 'npy']))
        output3 = os.path.join(outdir3, '.'.join([s_id, sd, 'npy']))
        output4 = os.path.join(outdir4, '.'.join([s_id, sd, 'npy']))
        if (not os.path.isfile(output2) or
            not os.path.isfile(output3) or
            not os.path.isfile(output4)):
            # path to the geodesic depth texture on the white mesh
            depth = os.path.join(path_s, 'surface_analysis',
                                 s_id+'_'+sd+'geodesic_depth.gii')
            depth_ar = gio.read(depth).darrays[0].data

            # path to the DPF (Depth Potential Function) on the white mesh
            DPF = os.path.join(path_s, 'surface_analysis',
                               s_id+'_'+sd+'white_DPF.gii')
            DPF_ar = gio.read(DPF).darrays[0].data

            # Subject back-projected group-clusters of pits
            # Path to updated parcellation HCP S1200
            clusters = os.path.join(path_s, 'surface_analysis_update',
                                    s_id+'_'+sd+'clusters_sym_lh.gii')
            # Path to updated parcellation HCP S900
            """
            clusters = os.path.join(path_s, 'surface_analysis',
                                    s_id+'_'+sd+'white_clusters_sym_lh.gii')
            """
            clusters_ar = gio.read(clusters).darrays[0].data
            #clusters_ar = clusters_ar.astype('int')
            white = os.path.join(database, s_id, 't1mri', 'BL',
                                 'default_analysis', 'segmentation',
                                 'mesh', s_id+'_'+sd+'white.gii')
            r=aims.Reader()
            # load the native white mesh
            mesh = r.read(white)
            # load the DPF texture corresponding to this mesh
            tex = r.read(DPF)
            # GeodesicPath(AimsSurfaceTriangle surface ,
            #              TimeTexture<float> texCurv,
            #              int method, int strain);
            g=aims.GeodesicPath(mesh, tex, 2, 1)

            # Extract the shortest geodesic path btwn pits sup_temporal b and c
            sulcus1=np.array(g.shortestPath_1_1_ind(
                pits_vertex[s_id][sd][sulc][B],
                pits_vertex[s_id][sd][sulc][C]))
            # Extract the shortest geodesic path btwn pits sup_temporal c and d
            sulcus2=np.array(g.shortestPath_1_1_ind(
                pits_vertex[s_id][sd][sulc][C],
                pits_vertex[s_id][sd][sulc][D]))

            sulcus = np.asarray(list(sulcus1)+list(sulcus2))
            areals =  np.asarray([int(ar) for ar in clusters_ar[sulcus]])
            areals = areals.astype('int')

            # In practice no need for first iB if start from pit sup_temporal_b
            # But might be useful if start from pit sup_temporal_a
            first_iB = np.where(areals == iB)[0][0]
            inds_iC = np.where(areals == iC)[0]
            last_iC = inds_iC[len(inds_iC)-1]
            # truncate the STAP depth profile btwn areals sup_temporal b and c
            sulcus = sulcus[first_iB:last_iC]

            positions = [0]
            vert=np.array(gio.read(white).darrays[0].data)
            length=0
            # Concatenate the length btwn each vertex along the extracted STAP
            for i in range(sulcus.size - 1):
                length=length + np.sqrt(
                    (vert[sulcus[i]][0] - vert[sulcus[i+1]][0])*
                    (vert[sulcus[i]][0] - vert[sulcus[i+1]][0])+
                    (vert[sulcus[i]][1] - vert[sulcus[i+1]][1])*
                    (vert[sulcus[i]][1] - vert[sulcus[i+1]][1])+
                    (vert[sulcus[i]][2] - vert[sulcus[i+1]][2])*
                    (vert[sulcus[i]][2] - vert[sulcus[i+1]][2]) )
                positions.append(length)
            # Save the extracted depth profils to binary files
            output2 = os.path.join(outdir2, '.'.join([s_id, sd, 'npy']))
            np.save(output2, DPF_ar[sulcus])
            output3 = os.path.join(outdir3, '.'.join([s_id, sd, 'npy']))
            np.save(output3, depth_ar[sulcus])
            output4 = os.path.join(outdir4, '.'.join([s_id, sd, 'npy']))
            np.save(output4, positions)

def build_pheno_STAP(parameters):
    """
    Identify plis de passage (PPs) in the STAP depth profile
    Convert this to a 0/1 (case/control) phenotype

    Parameters:
    indir: path to the saved depth profiles
    s_ids: subjects id
    outdir: output directory that will contain the phenotypes
    thr_DPF: threshold constraint on the DPF to identify PP
    thr_depth: threshold constraint on the geodesic depth
    """
    indir, s_ids, outdir, thr_DPF, thr_depth = parameters
    outdir = os.path.join(outdir, ('thrDPF'+str(thr_DPF)+
                                   '_thrDepth'+str(thr_depth)))
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    for sd in ['R', 'L']:
        df = pd.DataFrame()
        df['ID'] = s_ids
        df_depth = pd.DataFrame()
        df_depth['ID'] = s_ids
        count_pp =  []
        pp_depth = []
        indir2 = os.path.join(indir, 'STAP_specific', 'DPF', sd)
        indir3 = os.path.join(indir, 'STAP_specific', 'GeoDepth', sd)
        for s_id in s_ids:
            #print s_id
            # Load previously saved depth profils
            input2 = os.path.join(indir2, '.'.join([s_id, sd, 'npy']))
            DPF_profil = np.load(input2)
            input3 = os.path.join(indir3, '.'.join([s_id, sd, 'npy']))
            depth_profil = np.load(input3)
            # Identify the peaks in the depth profils

            """if np.amin(depth_profil)  < 13 and np.amin(DPF_profil) <= 0.42:
                count_pp.append(1)
            else:
                count_pp.append(0)"""
            
            # This condition is actually equivalent to the first one with
            # these set of parameters for detect_peaks
            indexes = detect_peaks(DPF_profil, mph=-thr_DPF, mpd=0,
                                   threshold=0, dist_bd=0, edge='both',
                                   valley=True, show=False)
            #count_pp.append(len(indexes))
            if len(indexes) > 0 and np.amin(depth_profil[indexes])  < thr_depth:
                count_pp.append(1)
            else:
                count_pp.append(0)            
            pp_depth.append(DPF_profil[indexes])  

        pp_depth = [np.min(pp) if len(pp) != 0 else np.nan for pp in pp_depth]
        df_depth[sd+'_pp'] = pp_depth
        df[sd+'_pp'] = (np.asarray(count_pp) >= 1)* 1
        output = os.path.join(outdir, 'PP_S1200_'+sd+'_STAP.csv')
        #df.to_csv(output, header=True, index=False)
        output = os.path.join(outdir, 'DPF_PP_'+sd+'_STAP.csv')
        df_depth.to_csv(output, header=True, index=False)
        output = os.path.join(outdir, 'PP_S1200_'+sd+'_STAP.phe')
        df0 = pd.DataFrame()
        df0['FID'] = df['ID']
        df0['IID'] = df['ID']
        df0[sd+'_pp'] = df[sd+'_pp']+1
        df0.to_csv(output, sep=' ', header=True, index=False)

def build_pheno_avg_STAP(parameters):
    indir, s_ids, outdir = parameters
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    for sd in ['R', 'L']:
        df = pd.DataFrame()
        df['ID'] = s_ids
        df_dpf = pd.DataFrame()
        df_dpf['ID'] = s_ids
        depth_avg = []
        DPF_avg = []
        indir2 = os.path.join(indir, 'STAP_specific', 'DPF', sd)
        indir3 = os.path.join(indir, 'STAP_specific', 'GeoDepth', sd)
        for s_id in s_ids:
            #print s_id
            # Load previously saved depth profils
            input2 = os.path.join(indir2, '.'.join([s_id, sd, 'npy']))
            DPF_profil = np.load(input2)
            input3 = os.path.join(indir3, '.'.join([s_id, sd, 'npy']))
            # Add 2 mm due to the morphological erosion and closing
            depth_profil = np.load(input3)+2
            depth_avg.append(np.mean(depth_profil))  
            DPF_avg.append(np.mean(DPF_profil))
            
        df[sd+'_avg_depth'] = depth_avg
        df_dpf[sd+'_avg_DPF'] = DPF_avg
        
        output = os.path.join(outdir, 'pits_'+sd+'_avg_depth_STAP.csv')
        df.to_csv(output, header=True, index=False)
        output = os.path.join(outdir, 'pits_'+sd+'_avg_DPF_STAP.csv')
        df_dpf.to_csv(output, header=True, index=False) 

        output = os.path.join(outdir, 'pits_'+sd+'_avg_depth_STAP.phe')
        df0 = pd.DataFrame()
        df0['FID'] = df['ID']
        df0['IID'] = df['ID']
        df0[sd+'_avg_depth'] = df[sd+'_avg_depth']
        df0.to_csv(output, sep=' ', header=True, index=False)
        
        output = os.path.join(outdir, 'pits_'+sd+'_avg_DPF_STAP.phe')
        df0 = pd.DataFrame()
        df0['FID'] = df_dpf['ID']
        df0['IID'] = df_dpf['ID']
        df0[sd+'_avg_DPF'] = df_dpf[sd+'_avg_DPF']
        df0.to_csv(output, sep=' ', header=True, index=False)
        

    for depth in ['depth', 'DPF']:
        inL = os.path.join(outdir, 'pits_L_avg_'+depth+'_STAP.csv')
        dfL = pd.read_csv(inL)
        dfL.index = dfL['ID']
        inR = os.path.join(outdir, 'pits_R_avg_'+depth+'_STAP.csv')
        dfR = pd.read_csv(inR)
        dfR.index = dfR['ID']    
        dfR = dfR.loc[dfL.index]
        dfR = dfR.dropna()
        dfL = dfL.loc[dfR.index]
        df = pd.DataFrame()
        # Compute the asymmetry index for all subjects
        df['asym_avg_'+depth] = (2*(dfL['L_avg_'+depth]-dfR['R_avg_'+depth])/
                                (dfL['L_avg_'+depth]+dfR['R_avg_'+depth]))
        if IMAGEN:
            df['ID'] = ["%012d" % p  for p in dfL['ID']]
        else:
            df['ID'] = dfL['ID']
        df.index = df['ID']
        

        output = os.path.join(outdir, 'pits_asym_avg_'+depth+'_STAP.csv')
        df.to_csv(output, header=True, index=False)
        
        output = os.path.join(outdir, 'pits_asym_avg_'+depth+'_STAP.phe')
        df0 = pd.DataFrame()
        df0['FID'] = df['ID']
        df0['IID'] = df['ID']
        df0['asym_avg_'+depth] = df['asym_avg_'+depth]
        print df0.head()
        df0.to_csv(output, sep=' ', header=True, index=False)


        
def build_pp_depth_dict(indir, s_ids):
    pp_depth = {}
    pp_DPF = {}
    for sd in ['L', 'R']:
        pp_depth[sd] = []
        pp_DPF[sd] = []
        indir2 = os.path.join(indir, 'STAP_specific', 'DPF', sd)
        indir3 = os.path.join(indir, 'STAP_specific', 'GeoDepth', sd)
        for s_id in s_ids:
            #print s_id
            # Load previously saved depth profils
            input2 = os.path.join(indir2, '.'.join([s_id, sd, 'npy']))
            DPF_profil = np.load(input2)
            input3 = os.path.join(indir3, '.'.join([s_id, sd, 'npy']))
            depth_profil = np.load(input3)
            # Identify the peaks in the depth profils
            indexes = detect_peaks(DPF_profil, mph=None, mpd=15, threshold=0,
                                   dist_bd=0, edge='both', valley=True,
                                   show=False)
            id_geoD = detect_peaks(depth_profil, mph=None, mpd=10, threshold=0,
                                   dist_bd=0, edge='both', valley=True,
                                   show=False)
            pp_DPF[sd].append(DPF_profil[id_geoD])
            pp_depth[sd].append(depth_profil[indexes])

    return pp_depth, pp_DPF

if __name__ == '__main__':
    ROOT_DIR = ''
    # Directory to Freesurfer database
    DIR_F = ''
    s_ids = os.listdir(DIR_F)

    # Set boolean for the IMAGEN cohort subject ids
    IMAGEN = False
    outdir = os.path.join(ROOT_DIR,
                          '3T_sulcal_pits/Freesurfer/depth_profiles')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    
    # Build dictionary containing parcel references
    # Labels HCP S1200 parcellation
    labels = os.path.join(os.getcwd(), 'labels_areals.csv')
    letters = ['a', 'b', 'c', 'd', 'e', 'f']
    sulci = ['sup_temporal']
    
    dict_ind = build_dict_areals(labels, sulci, letters)
    dict_vertex = os.path.join(outdir, 'sulcal_pits_STAP_vertex_pos_dict.json')

    # It takes quite a long time to build the dictionary of all subjects
    # and all sulci in a large database such as UK Biobank
    if not os.path.isfile(dict_vertex):
        pits_vertex = build_dict_pits_vertex(database, s_ids, sulci, letters,
                                             dict_ind)
        encoded = json.dumps(pits_vertex)
        with open(dict_vertex, 'w') as f:
            json.dump(encoded, f)
    else:
        with open(dict_vertex, 'r') as f:
            data = json.load(f)
            pits_vertex = json.loads(data)

    

    # Create the outdir to avoid conflict in parallel process at first call
    for sd in ['L', 'R']:
        outdir2 = os.path.join(outdir, 'STAP_specific', 'DPF', sd)
        if not os.path.isdir(outdir2):
            os.makedirs(outdir2)
        outdir3 = os.path.join(outdir, 'STAP_specific', 'GeoDepth', sd)
        if not os.path.isdir(outdir3):
            os.makedirs(outdir3)
        outdir4 = os.path.join(outdir, 'STAP_specific', 'positions_profil', sd)
        if not os.path.isdir(outdir4):
            os.makedirs(outdir4)

    # Create a list of parameters to run in parallel on each s_id
    parameters = []
    for s_id in s_ids:
            parameters.append([database, dict_ind, s_id, pits_vertex, outdir])
    
    number_CPU = cpu_count()-2
    pool = Pool(processes = number_CPU)
    pool.map(build_depth_profil_STAP, parameters)
    pool.close()
    pool.join()
    

    # path to the saved depth profiles
    indir = outdir
    # set output directory for the phenotypes
    outdir = os.path.join(ROOT_DIR, '3T_sulcal_pits/Freesurfer/phenotypes')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    # Set the threshold on DPF and geodesic depth as in the paper
    thr_DPF, thr_depth = 0.42, 13
    parameters = indir, s_ids, outdir, thr_DPF, thr_depth 
    build_pheno_STAP(parameters)
    outdir = os.path.join(outdir, "thrDPF0.42_thrDepth13")
    parameters = indir, s_ids, outdir
    build_pheno_avg_STAP(parameters)
