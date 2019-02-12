#!/usr/bin/env python

"""
Python wrapper scripts to setup SUBJECTS_DIR for source localization.

1. Create / setup directories for source localization.
2. Construct brain surfaces from MR.
3. Setup the source space.
4. Perform coregistration
- http://www.slideshare.net/mne-python/mnepythyon-coregistration-28598463

Setting up source space done, start forward and inverse model computation.
"""

import os
import os.path as op
import numpy as np
import mne
import subprocess
import errno

# using subprocess to run freesurfer commands (other option os.system)
# e.g. call(["ls", "-l"])
from subprocess import call

###############################################################################
# Subject directory
###############################################################################

data_path = mne.datasets.sample.data_path()
subjects_dir = data_path + '/subjects'

os.environ['SUBJECTS_DIR'] = subjects_dir
os.environ['DISPLAY'] = ':0'

###############################################################################
# Subject lists
###############################################################################

subject_list = ['sample']

###############################################################################
# Script configuration
###############################################################################

grade = '5'
pos = 5.0  # subject_spacing
template_spacing = 5.0
mindist = 5.0

mne_bin_path = '/Users/kiefer/mne/MNE-2.7.3-3268-MacOSX-i386/bin/'
freesurfer_home = '/Users/kiefer/mne/freesurfer/'
freesurfer_bin = '/Users/kiefer/mne/freesurfer/bin/'

mri_dir = '/mri/orig/001.mgz'
nii_fname = '_1x1x1mm_orig.nii.gz'

do_recon_all = False
do_vsrc = True

###############################################################################
# Helper functions
###############################################################################


def mksubjdirs(subjects_dir, subj):
    """
    Create the directories required by freesurfer.

    Parameters:
    -----------
    subjects_dir : str
        Path to the subjects directory.
    subj : str
        ID of the subject.

    Returns:
    --------
    None
    """

    # make list of folders to create

    folders_to_create = ['bem', 'label', 'morph', 'mpg', 'mpg', 'mri', 'rgb',
                         'scripts', 'stats', 'surf', 'tiff', 'tmp', 'touch']

    mri_subfolders = ['aseg', 'brain', 'filled', 'flash', 'fsamples', 'norm',
                      'orig', 'T1', 'tmp', 'transforms', 'wm']

    for count in range(0,len(mri_subfolders)):
        mri_subfolders[count] = os.path.join('mri', mri_subfolders[count])

    folders_to_create.extend(mri_subfolders)

    # create folders
    for folder in folders_to_create:
        dirname_prep = os.path.join(subjects_dir, subj, folder)

        try:
            os.makedirs(dirname_prep)
        except OSError as exc:
            if exc. errno == errno.EEXIST and os.path.isdir(dirname_prep):
                pass
            else:
                raise


def copyWithSubprocess(cmd):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


###############################################################################
# Setup directories + recon-all + make watershed bem
###############################################################################

# to convert from dicom to nii use mri_convert:
# mri_convert -it siemens_dicom -i path_to_dicom -ot nii -o subjects_dir/subjid/subjid_1x1x1mm_orig.nii.gz

if do_recon_all:
    for subj in subject_list:

        print 'Setting up freesurfer surfaces and source spaces for %s' % (subj)

        # # Makes subject directories and dir structures
        # # Be sure to be in subjects_dir before starting ipython
        # # It is best to run this step first, then place the *.nii/*.mgz file in the
        # # respective subject directory and then run the rest of the commands.
        #
        #
        # call([freesurfer_bin + 'mksubjdirs', subj])
        #
        # cmd = ['rsync','-av','--ignore-existing', mri_dir_pre + subj + mri_dir,
        #              subjects_dir + '/' + subj +  mri_dir]
        #
        # copyWithSubprocess(cmd)

        # alternative to using freesurfer, which is kind of buggy
        mksubjdirs(subjects_dir, subj)

        # Convert NIFTI files to mgz format that is read by freesurfer
        # do not do this step if file is already in 001.mgz format

        call([freesurfer_bin + 'mri_convert',
              op.join(subjects_dir, subj, subj + nii_fname),
              op.join(subjects_dir, subj + mri_dir)])

        # Reconstruct all surfaces and basically everything else.
        # Computationally intensive!
        call([freesurfer_bin + 'recon-all', '-autorecon-all', '-subjid', subj])

        # Set up the MRI data for forward model.
        call([mne_bin_path + 'mne_setup_mri', '--subject', subj])
        # $MNE_BIN_PATH/mne_setup_mri --subject $i --overwrite (if needed)

        # Setting up of Triangulation files
        call([mne_bin_path + 'mne_watershed_bem', '--subject', subj])
        # $MNE_BIN_PATH/mne_watershed_bem --subject $i --overwrite (available in python, use the python version)
        # alternatively use the python script
        # make_watershed_bem() might need to be exectued in a python console
        bem = mne.bem.make_watershed_bem(subject=subj, subjects_dir=subjects_dir, overwrite=True)

        # if we do get "blurry"/ low resolution heads in mne coreg use mne make_scalp_surfaces -s subj -d subjects_dir
        # call(['mne', 'make_scalp_surfaces', '-s', subj, '-d', subjects_dir])


###############################################################################
# Create volume source spaces
###############################################################################

if do_vsrc:
    for subj in subject_list:

        # Setting up the surface files
        watershed_dir = op.join(subjects_dir, subj, 'bem', 'watershed', subj)
        surf_dir = op.join(subjects_dir, subj, 'bem', subj)
        call(['ln', '-s', watershed_dir + '_brain_surface',
              surf_dir + '-brain.surf'])
        call(['ln', '-s', watershed_dir + '_inner_skull_surface',
              surf_dir + '-inner_skull.surf'])
        call(['ln', '-s', watershed_dir + '_outer_skin_surface',
              surf_dir + '-outer_skin.surf'])
        call(['ln', '-s', watershed_dir + '_outer_skull_surface',
              surf_dir + '-outer_skull.surf'])

        # from jumeg.jumeg_volmorpher import make_indiv_spacing
        # indiv_spacing = make_indiv_spacing(subj, 'fsaverage', template_spacing=template_spacing,
        #                                    subjects_dir=subjects_dir)

        bem_dir = op.join(subjects_dir, subj, 'bem')
        bem_name = op.join(bem_dir, subj + '-5120-5120-5120-bem.fif')
        bem_sol_name = op.join(bem_dir, subj + '-5120-5120-5120-bem-sol.fif')

        if not op.exists(bem_sol_name):
            model = mne.make_bem_model(subject=subj, ico=4, conductivity=[0.3], subjects_dir=subjects_dir)
            mne.write_bem_surfaces(bem_name, model)
            bem = mne.make_bem_solution(model)
            mne.write_bem_solution(bem_sol_name, bem)

        fname_mri = op.join(subjects_dir, subj, 'mri', 'aseg.mgz')
        fname_bem = bem_sol_name

        volume_labels = mne.get_volume_labels_from_aseg(op.join(subjects_dir, subj, 'mri', 'aseg.mgz'))

        vsrc = mne.setup_volume_source_space(subject=subj, pos=pos, mri=fname_mri,
                                             volume_label=volume_labels,
                                             bem=fname_bem, mindist=mindist,
                                             subjects_dir=subjects_dir,
                                             add_interpolator=True,
                                             verbose=None)

        volume_source_space_label_dictionary = {}
        for label, vol_src in zip(volume_labels, vsrc):
            vertnos = vol_src['vertno']
            volume_source_space_label_dictionary.update({label: vertnos})
        del vsrc

        vsrc = mne.setup_volume_source_space(subject=subj, pos=pos, mri=fname_mri,
                                             bem=fname_bem, mindist=mindist,
                                             subjects_dir=subjects_dir,
                                             add_interpolator=True,
                                             verbose=None)

        vol_source_space_name = op.join(subjects_dir, subj, subj + '_vol-%.2f-src.fif' % pos)
        np.save(vol_source_space_name[:-4] + '_vertno_labelwise', volume_source_space_label_dictionary)
        mne.write_source_spaces(vol_source_space_name, vsrc, overwrite=True)


# Align the coordinate frames using either mne coreg in terminal (recommended)
# or mne.gui.coregistration() in python - results in a -trans.fif file

