#! /usr/bin/env python
# -*- coding: utf-8 -*

##########################################################################
# Copyright (C) CEA, 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html for details.
##########################################################################

import os
from multiprocessing import cpu_count
from multiprocessing import Pool


# Method to project the group areals of pits on the native mesh
def texture_to_subj(s_id):
    # Filename containing the areals
    clusters = 'clusters_total_average_pits_smoothed0.7_60_sym_lh.gii'
    # Full path
    tex_atlas = os.path.join(os.getcwd(), 'pits_density_08_2017', clusters)
    # Path to the sphere reg of the template
    # in this case left hemisphere of fsaverage_sym
    sph_atlas = os.path.join(DIR_FSAVERAGE,
                             'folder_gii', 'sym', 'lh.sphere.reg.gii')

    for sd in ['R', 'L']:
        # Group areals projected on the native mesh
        subj_tex = os.path.join(DIR_BV, s_id,
                                't1mri/BL/default_analysis/segmentation/mesh',
                                'surface_analysis_update',
                                s_id+'_'+sd+'clusters_sym_lh.gii')

        # Sphere native R/L to fsaverage_sym left hemisphere
        subj_sph = os.path.join(DIR_BV, s_id,
                                't1mri/BL/default_analysis/segmentation/mesh',
                                'surface_analysis',
                                s_id+"."+sd+
                                ".sphere.reg.fsaverage_sym.surf.gii")

        cmd = " ".join(['python -m brainvisa.axon.runprocess',
                        'ProjectTextureOntoAtlas',
                        tex_atlas,
                        sph_atlas,
                        subj_sph,
                        subj_tex])

        print cmd
        os.system(cmd)



# Method to project Freesurfer texture on fsaverage
def texture_to_atlas(s_id):
    # Path to the sphere reg of the template
    sph_atlas = os.path.join(DIR_BV, 'folder_gii', 'sym', 'lh.sphere.reg.gii')

    for sd in ['R', 'L']:
        subj_sph = os.path.join(DIR_BV, 'HCP', s_id,
                                't1mri/BL/default_analysis/segmentation/mesh',
                                'surface_analysis',
                                s_id+"."+sd+
                                ".sphere.reg.lh.fsaverage_sym.surf.gii")
        for surf in ['.area', '.sulc', '.thickness', '.curv']:
            subj_tex = os.path.join(DIR_BV, 'HCP', s_id,
                                    'surf', sd.lower()+'h'+surf+'.gii')
            tex_atlas = os.path.join(DIR_BV, 'HCP', s_id,
                                    'surf',
                                     (sd.lower()+'h'+surf+
                                      '.lh.fsaverage_sym.surf.gii'))


            cmd = " ".join(['python -m brainvisa.axon.runprocess',
                            'ProjectTextureOntoAtlas',
                            subj_tex,
                            subj_sph,
                            sph_atlas,
                            tex_atlas])
            print cmd
            os.system(cmd)

if __name__ == '__main__':

    DIR_FSAVERAGE = ''
    DIR_BV = ''
    s_ids = os.listdir(DIR_BV)

    number_CPU = cpu_count()-1
    pool = Pool(processes = number_CPU)
    pool.map(texture_to_subj, s_ids)
    pool.close()
    pool.join()
