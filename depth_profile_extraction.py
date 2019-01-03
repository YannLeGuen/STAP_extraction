#! /usr/bin/env python
# -*- coding: utf-8 -*

##########################################################################
# @author: yann.leguen@cea.fr
# Copyright (C) CEA, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html for details.
##########################################################################


# Script that generalizes the extraction of geodesic path between any
# pair of pits belonging to a same sulcus

import os
import json
import numpy as np
import pandas as pd
import nibabel.gifti.giftiio as gio
from multiprocessing import Pool
from multiprocessing import cpu_count

from stap_depth_extraction import build_dict_areals
from stap_depth_extraction import build_dict_pits_vertex
from detect_peaks import detect_peaks

# Note: the shortestpath method is only available in Brainvisa
# trunk development branch (as of March 2018)
from soma import aims
from soma import aimsalgo as algo

def get_depth_profil(parameters):
    """
    Extract the geodesic path along the bottom of a sulcus
    between any pair of sulcal pits

    Parameters
    
    database: path to the database directory containing the sulcal pits
    dict_ind: dictionary containing areal name/numero correspondence
    s_id: subject id
    pits_vertex: dictionary containing vertex position of pits for each s_id
    outdir: output directory that will contain .npy file of the DPF, geodesic
            depth and position [mm] along the STAP
    """
    database, sulc, A, B, s_id, pits_vertex, outdir = parameters
    pits = '_'.join([sulc, A, sulc, B])

    # Set the subject input directory
    path_s = os.path.join(database, s_id, 't1mri', 'BL',
                          'default_analysis', 'segmentation', 'mesh')
    
    for sd in ['L', 'R']:
        # Set the output directories for each files
        outdir2 = os.path.join(outdir, sulc+'_'+A+'_'+B,
                               'DPF', sd)
        outdir3 = os.path.join(outdir, sulc+'_'+A+'_'+B,
                               'GeoDepth', sd)
        outdir4 = os.path.join(outdir, sulc+'_'+A+'_'+B,
                               'positions_profil', sd)

        # path to the geodesic depth texture on the white mesh
        depth = os.path.join(path_s, 'surface_analysis',
                             s_id+'_'+sd+'geodesic_depth.gii')
        depth_ar = gio.read(depth).darrays[0].data

        # path to the DPF (Depth Potential Function) texture on the white mesh
        DPF = os.path.join(path_s, 'surface_analysis',
                           s_id+'_'+sd+'white_DPF.gii')
        DPF_ar = gio.read(DPF).darrays[0].data

        # Subject back-projected group-clusters of pits
        # Path to updated parcellation HCP S1200
        clusters = os.path.join(path_s, 'surface_analysis_update',
                                s_id+'_'+sd+'clusters_sym_lh.gii')

        clusters_ar = gio.read(clusters).darrays[0].data
        #clusters_ar = clusters_ar.astype('int')
        white = os.path.join(database, s_id, 't1mri', 'BL', 'default_analysis',
                             'segmentation', 'mesh', s_id+'_'+sd+'white.gii')
      
        r=aims.Reader()
        # load the native white mesh
        mesh = r.read(white)
        # load the DPF texture corresponding to this mesh
        tex = r.read(DPF)
        # GeodesicPath(AimsSurfaceTriangle surface ,TimeTexture<float> texCurv,
        #              int method, int strain);
        g=aims.GeodesicPath(mesh, tex, 2, 1)

        # Extract the shortest geodesic path between pits sulc A and B
        sulcus=np.array(g.shortestPath_1_1_ind(pits_vertex[s_id][sd][sulc][A],
                                                pits_vertex[s_id][sd][sulc][B]))

        positions = [0]
        vert=np.array(gio.read(white).darrays[0].data)
        length=0
        # Concatenate the distance between each vertex along the extracted STAP
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

def build_pheno(parameters):
    sulc, A, B, s_ids, indir, outdir, th_DPF, th_depth = parameters
    pits = '_'.join([sulc, A, sulc, B])
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    for sd in ['L', 'R']:
        df = pd.DataFrame()
        df['ID'] = s_ids
        df_depth = pd.DataFrame()
        df_depth['ID'] = s_ids
        count_pp =  []
        pp_depth = []
        indir2 = os.path.join(indir, sulc+'_'+A+'_'+B, 'DPF', sd)
        indir3 = os.path.join(indir, sulc+'_'+A+'_'+B, 'GeoDepth', sd)
        for s_id in s_ids:
            # Load previously saved depth profils
            input2 = os.path.join(indir2, '.'.join([s_id, sd, 'npy']))
            DPF_profil = np.load(input2)
            input3 = os.path.join(indir3, '.'.join([s_id, sd, 'npy']))
            depth_profil = np.load(input3)

            indexes = detect_peaks(DPF_profil, mph=-th_DPF, mpd=0, threshold=0,
                                   dist_bd=0, edge='both',
                                   valley=True, show=False)
            #count_pp.append(len(indexes))
            if len(indexes) > 0 and np.amin(depth_profil[indexes])  < th_depth:
                count_pp.append(1)
            else:
                count_pp.append(0)
                
        df[sd+'_pp'] = (np.asarray(count_pp) >= 1)* 1
        print sd+' '+str(np.sum(df[sd+'_pp']))
        output = os.path.join(outdir, 'PP_S1200_'+sd+'_'+pits+'.csv')
        df.to_csv(output, header=True, index=False)

    """pp_depth = [np.min(pp) if len(pp) != 0 else np.nan for pp in pp_depth]
    df_depth[sd+'_pp'] = pp_depth
    output = os.path.join(outdir, 'GeoDepth_pits_S1200_'+sd+'_'+pits+'.csv')
    df_depth.to_csv(output, header=True, index=False)"""


if __name__ == '__main__':
    ROOT_DIR = ''
    # Cohort name
    COHORT = ''
    # Directory to the sulcal pits database
    database = os.path.join(ROOT_DIR,  '3T_sulcal_pits',
                            'Freesurfer', COHORT)
    s_ids = os.listdir(database)
    
    # Build dictionary containing parcel references
    # Labels HCP S1200 parcellation
    labels = os.path.join(os.getcwd(), 'labels_areals.csv')
    letters = ['a', 'b', 'c', 'd', 'e', 'f']
    sulci = ['sup_frontal', 'inf_frontal', 'postcentral', 'intraparietal',
             'sup_temporal', 's_central', 'inf_temporal', 'calcarine',
             'collateral', 'cingulate', 'precentral', 'occipito_temporal']
    
    dict_ind = build_dict_areals(labels, sulci, letters)

    outdir = os.path.join(ROOT_DIR, '3T_sulcal_pits'
                          'Freesurfer', 'depth_profiles', COHORT)
    dict_vertex = os.path.join(outdir, 'sulcal_pits__vertex_pos_dict.json')

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

    
    for sulc in sulci:
        for k, A in enumerate(letters):
            if len(dict_ind[sulc]['Areal']) < k+2 or len(letters) < k+2:
                break
            else:
                B = letters[k+1]
                # Create the outdir to avoid conflict in parallel
                # process at first call
                for sd in ['L', 'R']:
                    outdir2 = os.path.join(outdir,  sulc+'_'+A+'_'+B,
                                           'DPF', sd)
                    if not os.path.isdir(outdir2):
                        os.makedirs(outdir2)
                    outdir3 = os.path.join(outdir,  sulc+'_'+A+'_'+B,
                                           'GeoDepth', sd)
                    if not os.path.isdir(outdir3):
                        os.makedirs(outdir3)
                    outdir4 = os.path.join(outdir,  sulc+'_'+A+'_'+B,
                                           'positions_profil', sd)
                    if not os.path.isdir(outdir4):
                        os.makedirs(outdir4)

    # Create a list of parameters to run in parallel on each s_id
    parameters = []
    for s_id in s_ids:
        for sulc in sulci:
            for k, A in enumerate(letters):
                if len(dict_ind[sulc]['Areal']) < k+2 or len(letters) < k+2:
                    break
                else:
                    B = letters[k+1]
                    parameters.append([database, sulc, A, B,
                                       s_id, pits_vertex, outdir])

    number_CPU = cpu_count()-2
    pool = Pool(processes = number_CPU)
    #pool.map(get_depth_profil, parameters)
    pool.close()
    pool.join()


    parameters = []
    # path to the saved depth profiles
    indir = outdir
    # set output directory for the phenotypes
    outdir = os.path.join(ROOT_DIR, '3T_sulcal_pits', 'Freesurfer'
                          'phenotypes', COHORT)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    # Set the threshold on DPF and geodesic depth as in the paper
    thr_DPF, thr_depth = 0.42, 13
    for sulc in sulci:
        for k, A in enumerate(letters):
            if len(dict_ind[sulc]['Areal']) < k+2 or len(letters) < k+2:
                break
            else:
                B = letters[k+1]
                parameters.append([sulc, A, B, s_ids, indir,
                                   outdir, thr_DPF, thr_depth])


    number_CPU = cpu_count()-2
    pool = Pool(processes = number_CPU)
    pool.map(build_pheno, parameters)
    pool.close()
    pool.join()
